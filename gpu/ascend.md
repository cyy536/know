# soc

```
海思-昇腾是 SoC
├── NPU
│   ├── DaVinci AI Core
├── ARM CPU：部分作为 AI CPU，部分作为 Ctrl CPU
```

AI Core 和 AI CPU 都是可编程的

[抽象硬件架构-CANN社区版8.5.0-昇腾社区](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_10_0015.html)

SIMD：每个核心执行一条指令，处理多个数据

Scalar 执行地址计算、循环控制等标量计算工作，并把向量计算、矩阵计算、数据搬运、同步指令发射给对应单元执行。

DMA 负责数据搬运，包括Global Memory和Local Memory之间的数据搬运以及不同层级Local Memory之间的数据搬运。

向量计算单元、矩阵计算单元、数据搬运单元异步的并行执行接收到的指令

向量编程使用流水线；矩阵编程有高阶api可以调用的，类似tensor core；支持Vector与Cube混合计算（Flash Attention）

AI Core 可以拆分成 Cube Core 和 Vector Core，可以独立运行

```
SoC：310P 910 910B
__NPU_ARCH__ 是 NPU 架构版本号：2000 2200 3000
__CCE_AICORE__ 是 AI Core 微架构版本号：200 220 310
不是一一对应
```

# 算子

编写算子的原型定义json文件，用于生成算子开发工程

```
├── op_host                    
│   ├── add_example_def.cpp
│   ├── add_example_tiling.cpp    
├── op_kernel               
│   ├── add_example.cpp  
│   ├── add_example_tiling_key.h
```

op_kernel下存放kernel侧算子实现。op_host下存放host侧代码实现，包括算子原型定义、host侧tiling实现。

```
namespace ops {
class AddExample : public OpDef
{
public:
    explicit AddExample(const char* name) : OpDef(name)
    {
    }
};
OP_ADD(AddExample); 
}
```

```
template <uint32_t schMode>
__global__ __aicore__ void add_example(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling){}
```

大多数情况下，Local Memory的存储，无法完整的容纳算子的输入与输出，需要每次搬运一部分输入进行计算然后搬出，再搬运下一部分输入进行计算，直到得到完整的最终结果，这个数据切分、分块计算的过程称之为Tiling。

开发者可以从上下文结构中获取算子的输入、输出以及属性信息，也就是Tiling实现的输入，经过Tiling计算后，获取到TilingData数据结构（切分算法相关参数）、blockDim变量、用于选择不同的kernel实现分支的TilingKey、算子workspace的大小，也就是Tiling实现的输出，并将这些输出设置到上下文结构中。

```
namespace optiling {
BEGIN_TILING_DATA_DEF(MyAddTilingData)  // 声明tiling结构名字
  TILING_DATA_FIELD_DEF(uint32_t, field1);   // 结构成员的类型和名字
  TILING_DATA_FIELD_DEF(uint32_t, field2);
  TILING_DATA_FIELD_DEF(uint32_t, field3);
END_TILING_DATA_DEF;
 
REGISTER_TILING_DATA_CLASS(MyAdd, MyAddTilingData)  // tiling结构注册给算子
}
```

```
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    if (condition) {
        context->SetTilingKey(1);
    } else {
        context->SetTilingKey(2);
    }
    return ge::GRAPH_SUCCESS;
}
```

workspace内存分为两部分：Ascend C API需要的workspace内存和算子实现使用到的workspace内存（按需）。单算子API执行场景，可以通过单算子API调用第一段接口获取workspace的大小，然后由开发者申请对应大小的Global Memory；

```
namespace optiling {
IMPL_OP_OPTILING(AddExample).Tiling(AddExampleTilingFunc).TilingParse<AddExampleCompileInfo>(TilingParseForAddExample);
}
```

单算子API调用：

```
aclnnStatus aclnnXxxGetWorkspaceSize(const aclTensor *src, ..., aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
aclnnStatus aclnnXxx(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
```

# 图

算子入图场景下，需要提供shape推导等算子入图适配函数的实现。

在图模式下用户首先将模型的计算过程构造成一张图，然后通过GE将图下发到昇腾硬件执行。

通过算子原型构建Graph

借助GE能力，PyTorch、TensorFlow、MindSpore、PaddlePaddle等主流AI框架的算法模型可以统一转换为使用Ascend IR（Ascend Intermediate Representation）表示的计算图（Ascend Graph），并通过GE的图编译加速技术，显著提升计算图在昇腾硬件上的执行效率。

一种是将模型文件转成Ascend IR，一种是将PyTorch的FX图转换为Ascend IR（扩展torch.compile）

GE提供了两种常规的模型调度模式（下沉调度和Host调度）用于实现Host和Device之间的高效协同。

下沉调度模式通常适用于静态shape模型，由于输入tensor shape固定不变，可以在编译阶段完成内存编排和Tiling计算，因此可以将模型中的算子以整图形式下发到Device上，执行时，只需在Host侧下发一个模型执行的Task即可触发模型在Device上调度执行，从而提升模型调度性能。

Host调度模式通常适用于动态shape模型，由于输入tensor的shape不确定，必须在上一个算子完成shape推导后，才能确定下一个算子的输入shape等信息，因此整个模型无法下沉执行，通常需要将每个算子Kernel逐一下发到Device执行。

Tiling下沉优化方案：将Tiling计算下沉至Device侧的AI CPU上执行，从而实现计算全程在Device侧高效完成。

可以将Graph编译为om离线模型

单算子模型执行：将该算子描述文件编译成单算子模型文件（*.om文件），再分别调用acl接口加载om模型文件、执行算子

Kernel直调（用<<<>>>接口调用核函数）：通过适配torch.library或Pybind注册自定义算子，可以实现PyTorch框架调用算子Kernel程序。注册后在Python侧可以通过torch.ops.namespace.op_name方式进行调用。

单算子API调用：支持前向和反向，torch_npu.npu_add_custom()

图模式调用：。。。

通算融合算子统称为MC²算子：HCCL为集合通信任务客户端，运行在AI Core上，且不执行通信任务，而是由用户调用Prepare接口将对应类型的通信任务信息发送给AI CPU服务端

# GroupedMatmul

[Installation — vllm-ascend](https://docs.vllm.ai/projects/ascend/en/latest/installation.html)

```
for i :
	C[mi,ni]=A[mi,ki]XB[ki,ni]
```

```
import torch
import torch_npu
import random

def random_nonnegative_list(n=2, total=10):
    cuts = sorted(random.sample(range(total + n - 1), n - 1))
    # print(cuts)
    cuts = [-1] + cuts + [total + n - 1]
    return [cuts[i+1] - cuts[i] - 1 for i in range(n)]

# print(random_nonnegative_list())

num_tokens_top_k,hidden_size,num_experts,I_2,I=65536,2048,128,1536,768

hidden_states = torch.randn(num_tokens_top_k, hidden_size).npu()
w1 = torch.randn(num_experts, hidden_size, I_2).npu()
w2 = torch.randn(num_experts, I, hidden_size).npu()
group_list = torch.tensor(random_nonnegative_list(num_experts,num_tokens_top_k), dtype=torch.int64).npu()
group_list_type = 1#分组轴上每组大小

gate_up_out = torch_npu.npu_grouped_matmul(
    x=[hidden_states],
    weight=[w1],
    split_item=2,#2/3代表输出为单tensor
    group_list_type=group_list_type,
    group_type=0,#m轴分组
    group_list=group_list,
)[0]

gate_up_out = torch.randn(num_tokens_top_k,I).npu()

hidden_states = torch_npu.npu_grouped_matmul(
    x=[gate_up_out],
    weight=[w2],
    split_item=2,
    group_list_type=group_list_type,
    group_type=0,
    group_list=group_list,
)[0]
```

[aclnnGroupedMatmulV5-CANN商用版8.5.0-昇腾社区](https://www.hiascend.com/document/detail/zh/canncommercial/850/API/aolapi/context/ops-transformer/aclnnGroupedMatmulV5.md)

```
gdb -ex "break aclnnGroupedMatmul" \
    -ex "break aclnnGroupedMatmulV2" \
    -ex "break aclnnGroupedMatmulV3" \
    -ex "break aclnnGroupedMatmulV4" \
    -ex "break aclnnGroupedMatmulV5" \
    -ex "run" \
    --args python example1.py
(gdb) bt
#0  0x0000fffee432a38c in aclnnGroupedMatmulV5 () from /usr/local/Ascend/cann-8.5.0/lib64/libopapi.so
#1  0x0000ffff311ff7f0 in ?? ()
   from /root/autodl-tmp/vllm-ascend-env/lib/python3.10/site-packages/torch_npu/lib/libtorch_npu.so
#2  0x0000ffff32768144 in ?? ()
   from /root/autodl-tmp/vllm-ascend-env/lib/python3.10/site-packages/torch_npu/lib/libtorch_npu.so
#3  0x0000ffff30550670 in ?? ()
   from /root/autodl-tmp/vllm-ascend-env/lib/python3.10/site-packages/torch_npu/lib/libtorch_npu.so
#4  0x0000ffff3055124c in ?? ()
   from /root/autodl-tmp/vllm-ascend-env/lib/python3.10/site-packages/torch_npu/lib/libtorch_npu.so
#5  0x0000ffff3054f168 in ?? ()
   from /root/autodl-tmp/vllm-ascend-env/lib/python3.10/site-packages/torch_npu/lib/libtorch_npu.so
#6  0x0000ffffa50a1d98 in execute_native_thread_routine () from /root/miniconda3/bin/../lib/libstdc++.so.6
#7  0x0000ffffb1430398 in start_thread (arg=0x80e960) at ./nptl/pthread_create.c:442
#8  0x0000ffffb1499e9c in thread_start () at ../sysdeps/unix/sysv/linux/aarch64/clone.S:79
```

[docs/common/GroupedMatmul算子设计介绍.md · Ascend/cann-ops-adv - Gitee.com](https://gitee.com/ascend/cann-ops-adv/blob/master/docs/common/GroupedMatmul算子设计介绍.md)

把单个矩阵分块到不同core上执行，所有的core是负载均衡的

CANN 8.5.0为首次开源开放架构升级的版本

[ops-transformer/gmm/grouped_matmul/op_kernel · CANN/ops-transformer - AtomGit | GitCode](https://gitcode.com/cann/ops-transformer/blob/8.5.0/gmm/grouped_matmul/op_kernel/grouped_matmul.cpp)

```
git checkout 8.5.0
bash install_deps.sh
chmod +x Ascend-cann-toolkit_${cann_version}_linux-${arch}.run
./Ascend-cann-toolkit_${cann_version}_linux-${arch}.run --install
pip3 install -r requirements.txt
bash build.sh --pkg --jit --soc=ascend910b -j1 2>&1 | tee build.log
        // OpAICoreConfig config_kirin = GetKirinCoreConfig();
        // this->AICore().AddConfig("kirinx90", config_kirin);
./cann-${soc_name}-ops-transformer_${cann_version}_linux-${arch}.run --full 
bash build.sh --run_example grouped_matmul eager
```

算子库的源码编译有error和卡住的现象

```
        modified:   attention/prompt_flash_attention/op_host/prompt_flash_attention_tiling.cpp
        modified:   ffn/swin_attention_ffn/op_host/swin_attention_ffn_def.cpp
        modified:   ffn/swin_attention_ffn/op_host/swin_attention_ffn_tiling.cpp
        modified:   ffn/swin_transformer_ln_qkv/op_host/swin_transformer_ln_qkv_def.cpp
        modified:   ffn/swin_transformer_ln_qkv/op_host/swin_transformer_ln_qkv_tiling.cpp
        modified:   install_deps.sh
        modified:   moe/moe_compute_expert_tokens/op_host/moe_compute_expert_tokens_def.cpp
        modified:   moe/moe_finalize_routing/op_host/moe_finalize_routing_def.cpp
        modified:   moe/moe_finalize_routing_v2/op_host/moe_finalize_routing_v2_def.cpp
        modified:   moe/moe_re_routing/op_host/moe_re_routing_def.cpp
        modified:   moe/moe_token_unpermute/op_host/moe_token_unpermute_def.cpp
        modified:   moe/moe_token_unpermute_with_ep/op_host/moe_token_unpermute_with_ep_def.cpp
        modified:   posembedding/apply_rotary_pos_emb/op_host/apply_rotary_pos_emb_def.cpp
        modified:   posembedding/dequant_rope_quant_kvcache/op_host/dequant_rope_quant_kvcache_def.cpp
        modified:   posembedding/interleave_rope/op_host/interleave_rope_def.cpp
        modified:   posembedding/rope_quant_kvcache/op_host/rope_quant_kvcache_def.cpp
        modified:   posembedding/rotary_position_embedding/op_host/rope_interleaved_tiling.h
        modified:   posembedding/rotary_position_embedding/op_host/rope_rotate_half_tiling.h
        modified:   posembedding/rotary_position_embedding/op_host/rotary_position_embedding_def.cpp
```

单个算子有几千行代码，没有具体文档

```
namespace optiling {
REGISTER_TILING_DATA_CLASS(GroupedMatmul, GMMTilingData)
class GMMTiling {
public:
    GMMTilingData tilingData;
};
}  // namespace optiling

ASCENDC_EXTERN_C ge::graphStatus TilingGMM(gert::TilingContext* context) {
  GMMTiling tiling;
  OP_CHECK_IF(tiling.Init(context) != ge::GRAPH_SUCCESS,
  return tiling.RunFusionKernelTiling(context);
}

ge::graphStatus GMMTiling::Init(const gert::TilingContext* context) {
  OP_CHECK_IF(PrepareTilingData(context) != ge::GRAPH_SUCCESS,
  tilingData.gmmArray.set_mList(mList_);

ge::graphStatus GMMTiling::PrepareTilingData(const gert::TilingContext* context) {
  OP_CHECK_IF(GMMGetAttrs(context) != ge::GRAPH_SUCCESS,

ge::graphStatus GMMTiling::GMMGetAttrs(const gert::TilingContext* context) {
  auto attr = context->GetAttrs();
  groupListType_ = groupListTypePtr != nullptr ? *groupListTypePtr : 0;
  
ge::graphStatus GMMTiling::RunFusionKernelTiling(gert::TilingContext* context) {
  OP_CHECK_IF(CalMMTiling(context, compileInfoPtr) != ge::GRAPH_SUCCESS,
  OP_CHECK_IF(GMMSetMMTiling(context, compileInfoPtr) != ge::GRAPH_SUCCESS,
  OP_CHECK_IF(DynamicTilingSingleN(context, usedCoreNum_, compileInfoPtr) != ge::GRAPH_SUCCESS,
  GMMSetTplTilingKey(context);

ge::graphStatus GMMTiling::CalMMTiling(const gert::TilingContext* context, const GMMCompileInfo* compileInfoPtr) {
  baseN_ = BEST_BASEN; // init

ge::graphStatus GMMTiling::GMMSetMMTiling(const gert::TilingContext* context, const GMMCompileInfo* compileInfoPtr) {
  matmul_tiling::MultiCoreMatmulTiling mm(platformInfo);
  OP_CHECK_IF(mm.GetTiling(tilingData.mmTilingData) == -1,

ge::graphStatus GMMTiling::DynamicTilingSingleN(gert::TilingContext* context, const uint32_t& aicNum, const GMMCompileInfo *compileInfoPtr) {

void GMMTiling::GMMSetTplTilingKey(gert::TilingContext* context) {
    aivAicRatio = static_cast<uint32_t>(GROUPED_MATMUL_CUBE_ONLY);
  context->SetTilingKey(tilingKey);
```

