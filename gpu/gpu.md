# gpu

产品线
```
GeForce：消费级/游戏/创作者
Quadro：是旧专业工作站品牌，现已由 NVIDIA RTX / RTX PRO 取代
Titan：消费级和专业/创作者之间的混合定位，已停用
Tesla：旧的数据中心计算品牌，已停用
A/L/H/B：数据中心 GPU 
Jetson：边缘 AI/机器人/嵌入式
```

微架构-compute capability

按微架构发布时间排序
```
pascal-6:1080ti,Tesla p100
volta-7:Tesla v100
turing-7.5:2080ti,T4
ampere-8:a100,a800,a30
      -8.6:3090,RTX A6000
hopper-9:gh200,h200,h100,h800,h20
ada lovelace-8.9:4090,RTX 6000 Ada,L40s,L20
blackwell-10:B100,B200,GB200,B300,GB300
         -11: Jetson T5000
         -12: RTX 50
```

```
PCIe 显卡:
    1 个 GPU 芯片
    HBM
    PCIe 金手指
    少量型号可能有 NVLink Bridge

SXM 模块:
    1 个 GPU 芯片
    HBM
    压接到 HGX baseboard

8 个 SXM GPU 模块通过 HGX baseboard 上的 NVLink 走线连接 NVSwitch 芯片

HGX baseboard：
    SXM GPU 模块插座
    NVSwitch 芯片
    NVLink 高速走线
    与 CPU 主板/PCIe switch 的连接
```

HGX (Hyperscale Graphics eXtension) 是 NVIDIA 提供给厂商集成的 GPU 服务器平台。DGX 是 NVIDIA 自己卖的完整 AI 服务器产品。

gh200=h200+grace cpu

gb200=2*b200+grace cpu

```
GB200 NVL72 Rack

Compute Tray x18
  Tray 0:  2 个 GB200 Grace Blackwell Superchip
  ...

NVLink Switch Tray x9
  NVSwitch fabric
  连接全部 72 个 Blackwell GPU

Scale-out Network
  ConnectX / InfiniBand / Ethernet
```

GB200 superpod (NVL576): 多 Racks，两级 NVLink Switch Tray 互联

xai有10万卡集群

tensor core首次引入
```
FP16  -> Volta
INT8  -> Turing
INT4  -> Turing
BF16  -> Ampere
FP8   -> Hopper
FP4   -> Blackwell
```

```
fp32:1+8+23
fp16:1+5+10，不能表示太小的值，梯度溢出，要放大loss
bf16:1+8+7，精度低
fp8:1+4+3,1+5+2
```

# nvidia-smi

设备监控和管理

```
nvidia-smi --query-gpu=compute_cap --format=csv
nvidia-smi -q | grep -i -A 10 "PCIe Generation"
nvidia-smi topo -m
--matrix
```
全双工，PCIe3单向带宽1GB/s

全双工，NVLink3单向带宽25GB/s


# cuda

硬件层级：GPU -> GPC -> SM -> warp -> thread

任务层级: grid -> cluster -> block/CTA -> cta_tile -> hmma_tile -> tensor_core_tile

一个 kernel 启动后，grid 里的 block 会被调度到各个 SM。只有当某个 block 需要的资源满足 SM 限制时，它才能成为该 SM 上的 resident block。没有被分配到 SM 的 block 会等待，直到有 SM 资源释放。一个 SM 可以同时驻留多个 block，只要资源够。如果单个 block 的资源需求超过单个 SM 能力，那么这个 kernel 无法合法启动。

SM 以 warp 为单元调度执行，可并发、并行执行多个 warp。这个切换基本没有 CPU 那种上下文切换开销，因为 warp 的寄存器状态已经常驻在 SM 上。

```
每个 SM 有:
  register file
  shared memory
  L1 / texture cache

整个 GPU 共享:
  L2 cache
  global memory
```
H100 combined L1 + texture + shared memory = 256 KB / SM。H100 可配置 shared memory 容量最高 = 228 KB / SM

register 放不下产生 spill 时，会 spill 到 local memory。local memory 物理上在 global memory 中，可能经过 L1/L2 cache。

shared memory 通常划分为 32 个 bank。
常见模式下每个 bank 每周期服务 32b。
如果一个 warp 内多个线程访问同一个 bank 的不同地址，会发生 bank conflict，需要序列化。
如果多个线程访问同一个地址，可能走 broadcast，不一定冲突。

global memory 访问会按 warp 合并。现代 CUDA 架构通常以 32B 为基本内存事务单位。

```
Tensor Core 操作通常由整个 warp 协作完成。
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 是 warp-level synchronous 操作。
同一个 warp 内所有参与线程必须以一致方式调用，否则行为未定义。
线程会等待该 MMA 操作完成后继续。

Hopper 引入 WGMMA。
一个 warp group = 4 个 warp = 128 个线程，通常来自同一个 SM 中连续的 warps。
wgmma.mma_async 是 warp-group 级别的异步矩阵乘操作。

TMA = Tensor Memory Accelerator。
Hopper 引入，用于 global memory <-> shared memory 的大块、多维 tensor 异步搬运。也支持 cluster 内 shared memory 之间的搬运。
通常由少量线程，甚至一个线程，发起 TMA copy。
TMA 避免让整个 warp 手写地址计算和循环搬运，减少寄存器和指令开销。

WGMMA 和 TMA 都被设计成异步操作。
发起后线程可以执行其它不依赖结果的任务；
但在消费 WGMMA 结果或 TMA 搬运的数据前，必须显式等待/同步。

cluster 是 Hopper 引入的一个新的 CUDA 线程层级。cluster 是一组 block。这些 block 会被保证同时调度到同一个 GPU Processing Cluster 的一组 SM 上，从而可以做更强的协作。支持 cluster-level synchronization，访问彼此的 distributed shared memory。
```

## api

CUDA 把昂贵的初始化推迟到第一次实际使用 GPU 的时候做，warmup很重要

一旦检测到错误，在下次cuda api调用时返回

### context

CUDA context 是某个进程在某张 GPU 上的 CUDA 执行环境。包含该 GPU 上的内存分配、模块、kernel、stream、event、状态等

一个进程内的线程可以共享context，不同context时分复用gpu，上下文切换会有较大的开销

mps server代理clients的任务，只使用一个context，不同clients可以并行执行，可以控制client使用的sm比例。

MIG在Ampere架构引入，物理GPU虚拟化成多个独立的GPU实例

Green Context 的创建和切换开销更小，适合频繁创建/销毁或大量并发使用的场景。在创建 Green Context 时，可以显式传入它应使用的资源集合，从而实现更精细的资源调度和管理。Only SM type is supported today.

### stream

Hyper-Q 就是把 GPU 的硬件工作队列从一个扩展到多个，让多个 stream 提交的独立任务更容易并发执行

默认流会与所有其他流（非阻塞流（需通过cudaStreamNonBlocking标志显式创建）除外）强制同步，任何默认流中的操作需等待其他流中的任务完成，反之亦然。

cudaStreamAddCallback由驱动在host上执行

### event

stream 绑定创建它的 context，只能在该 GPU 上提交工作；cudaStreamWaitEvent 允许另一个 GPU 的 stream 等待该 event，从而实现跨 device 同步。

### memory

在支持 UVA 的 CUDA 程序里，CPU host memory 和多个 GPU 的 device memory 会被放进同一个虚拟地址空间里管理。虚拟地址区间由 OS 和 CUDA driver 协调，保证不会冲突。

cudaMalloc 是高层 API：一次调用完成地址分配和物理显存 allocation，返回可直接给 GPU 使用的 device pointer。

CUDA VMM 是低层 API：地址分配、物理显存 allocation、map/unmap、访问权限设置都由用户显式控制。

cudaMemcpyAsync: 可以指定 stream，异步 API

cudaMemcpy: cudaMemcpyAsync 提交到默认流，然后 host 等它完成。

cpu page memory不能直接传到GPU memory，需要先传到cpu pin memory，H2D/D2H 可能会发生隐式阻塞

```
通过 cudaMallocManaged 或 __managed__ 分配 Unified Memory，CPU 和 GPU 使用同一个指针访问，由 CUDA driver 负责数据迁移。

cudaMallocManaged 分配统一内存
host 初始化数据
kernel 使用同一个 pointer
如果数据不在可访问的位置，触发 page fault
驱动可能分配目标端物理页、迁移数据、更新 CPU/GPU 页表
cudaMemPrefetchAsync 可提前把页面迁移到目标 device，减少运行时 page fault 和迁移抖动
```

### ipc

获取其他进程的资源：Memory IPC、Event IPC

进程间传 handle 需要你自己用 OS IPC 完成

CUDA memory IPC 的 handle 机制又可分：
```
legacy CUDA IPC: 同一台机器内进程间共享 GPU memory
fabric CUDA IPC: GPU 位于同一个 NVSwitch fabric
```
NVLink fabric = 用 NVLink 做链路、NVSwitch 做交换、由系统软件管理起来的 GPU 高速互联网络/访问域。

### ...

cuda graph：将动态的、由 CPU 逐条提交的 GPU 操作序列，转换为静态的、可一次性提交的依赖图，从而减少 CPU 驱动开销

CUDA 驱动自动分析操作间的数据依赖：两个无关 kernel 可以并行执行

with torch.cuda.graph() 内部必须是“纯异步 CUDA 操作流”，任何 CPU 交互都会破坏捕获

根据 CTA 的硬件资源使用情况，计算每个 SM 最多可以同时驻留多少个 CTA，和任务量无关，任务量决定驻留时间
```
cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &ctas_per_sm, kernel, Kernel_traits::THREADS, smem_size);
```
## GPU Direct P2P

允许同一节点内 GPU 之间直接访问或传输数据，避免通过主机内存作为中转缓冲区

Direct Access：GPU0 reads or writes GPU1 memory (load/store)

Direct Transfers：cudaMemcpy() initiates DMA copy from GPU0 memory to GPU1 memory

如果 GPU 之间有 NVLink / NVSwitch，通常优先走 NVLink P2P。如果没有 NVLink，则可能走 PCIe P2P。如果 P2P 不可用，则退化为经过 host memory 的中转路径。PCIe P2P 只能在同一个 RC 下，如果连在一个PCIe switch 下，传输不需要经过 RC。如果跨 NUMA 的话，还要经过 QPI/UPI。

## GPU Direct RDMA

允许第三方设备（如RDMA网卡）直接访问GPU显存，通过PCIe总线实现RDMA网卡与GPU显存的直接DMA传输。其核心是注册GPU显存为RDMA可访问区域，使网卡能直接读写显存

the only theoretical requirement for GPUDirect RDMA to work between a third-party device and an NVIDIA GPU is that they share the same root complex

## GDRCopy

GDRCopy 允许 CPU 通过 BAR mapping 直接映射并读写 GPU memory。

While GPUDirect RDMA is meant for direct access to GPU memory from third-party devices, it is possible to use these same APIs to create perfectly valid CPU mappings of the GPU memory.

依赖 PCIe BAR 将 GPU 显存暴露给 CPU。绕过 GPU DMA 引擎的调度开销，小数据延迟可低于 cudaMemcpy。

```
GDRCopy flush:
  在 GPUDirect RDMA 中，NIC completion 不一定等价于 GPU kernel 可见。
  NIC 写 GPU memory 的路径是 NIC -> PCIe -> GPU PCIe interface；
  kernel 读 GPU memory 的路径是 SM -> cache -> HBM。
  通过 PCIe read transaction 强制刷新，让之前的 PCIe writes 在 GPU 内存系统里变得可见。
```

## zero-copy
GPU通过PCIe总线访问pin memory，不需要经显存

## kernel

```
PTX = 虚拟指令集 / 中间表示
SASS: NVIDIA GPU 的真实机器指令 / 汇编指令
cubin: 里面包含 SASS 机器码和相关元数据
fatbin: 一个容器，可以同时装多个 cubin 和/或 PTX
  
__global__：cpu调用gpu执行
__device__：gpu调用gpu执行
__host__：cpu调用cpu执行

如果只是使用 CUDA api，只用 g++ 即可，但 g++ 无法处理 __global__ 等
nvcc会先对 .cu 文件进行预处理，分离出主机代码和设备代码。
主机代码会被交给主机编译器编译成目标文件（.o）。
设备代码会被编译成 fatbin，设备代码会被打包到主机目标文件中
不加-c的话，nvcc会调用主机编译器进行链接，生成可执行文件。

nvcc
  ├── host code -> 交给 host compiler
  └── device code
        -> ...
        -> PTX
        -> ptxas
        -> cubin
        -> fatbin

Clang
  ├── host code
  └── device code
        -> LLVM IR
        -> LLVM optimizer
        -> NVPTX backend
        -> PTX
        -> ...
```

```
-arch=compute_xx:
选择虚拟架构，影响 PTX 生成。

-code=compute_xx:
生成 PTX。

-code=sm_xx:
生成 cubin。

-arch=compute_80 -code=sm_80:
只生成 sm_80 cubin。

-arch=compute_80 -code=sm_80,compute_80:
同时生成 sm_80 cubin 和 compute_80 PTX。

-arch=sm_80:
简写，等价于 -arch=compute_80 -code=sm_80,compute_80。
```

PTX 可以在更高 compute capability 的 GPU 上通过 driver JIT 编译运行。编译结果会进入 ~/.nv/ComputeCache。PTX 内容变化或 driver 升级通常会导致重新 JIT。
```
1. 如果 fatbin 里有当前 GPU 对应的 sm_xx cubin
   -> 直接加载 cubin，不 JIT PTX

2. 如果没有对应 sm_xx cubin，但有兼容的 compute_xx PTX
   -> driver JIT，把 PTX 编译成当前 GPU 的 SASS

3. 如果既没有可用 cubin，也没有兼容 PTX
   -> 启动失败，常见报错 no kernel image is available
```

__shfl_down_sync(0xffffffff, sum, delta): 一个线程从 warp 内的另一个线程 (+delta) 读取数据，不需要使用共享内存

atomicAdd 是 CUDA 原子加法操作，可用于 global memory 或 shared memory 中的变量

### cccl

CUDA Core Compute Libraries: Thrust, CUB, libcudacxx 

CUB is a lower-level, CUDA-specific library designed for speed-of-light parallel algorithms across all GPU architectures. In addition to device-wide algorithms, it provides cooperative algorithms like block-wide reduction and warp-wide scan, providing CUDA kernel developers with building blocks to create speed-of-light, custom kernels.

```
float m = BlockReduce(temp_storage).Reduce(m_partial, max_op);
float d = BlockReduce(temp_storage).Sum(d_partial);
```
### cuBLAS

cuBLAS：主机端 API，支持所有 152 个标准 BLAS 例程，适用于通用线性代数计算。

cuBLASLt：轻量级库，专注于 GEMM，支持偏置和激活函数。

cuDNN provides highly tuned implementations for standard routines, such as forward and backward convolution, attention, matmul, pooling, and normalization.

cuDNN Frontend is NVIDIA's modern, open-source entry point to the cuDNN library and a growing collection of high-performance open-source kernels — scaled dot-product attention (SDPA / Flash Attention), grouped GEMM fusions for Mixture-of-Experts (MoE) training, fused normalization + activation, and more.

apex(A PyTorch Extension): Tools for easy mixed precision and distributed training in Pytorch

### cutlass

CUTLASS is a collection of abstractions for implementing high-performance matrix-matrix multiplication (GEMM) and related computations at all levels and scales within CUDA.

```
    using Shape = cutlass::gemm::GemmShape<16 * M, 16 * N, 16>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
    using WarpMma = typename cutlass::gemm::warp::DefaultMmaTensorOp<
        Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
        cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAdd, 1, true>::Type;
    constexpr int kIters = Shape::kK / InstructionShape::kK;
    WarpMma mma_op;
    #pragma unroll
    for (int iter = 0; iter < kIters; iter++) {
        mma_op(c_cl, reinterpret_cast<const typename WarpMma::FragmentA (&)>(a_cl[iter]),
               reinterpret_cast<const typename WarpMma::FragmentB (&)>(b_cl[iter]), c_cl);
    }
```

CUTLASS 3.0 introduced a new core library, CuTe, to describe and manipulate tensors of threads and data. CuTe is a collection of C++ CUDA template abstractions for defining and operating on hierarchically multidimensional layouts of threads and data.

用 Shape 描述逻辑大小，用 Stride/Layout 描述坐标到内存 offset 的映射，用 Tensor 把 pointer + layout 绑定起来，再用 tile / partition / copy / mma 操作它。

```
auto tensor = make_tensor(
    make_gmem_ptr(ptr),
    make_layout(
        make_shape(Int<4>{}, Int<8>{}),
        make_stride(Int<8>{}, Int<1>{})
    )
);
auto tile = local_tile(
    tensor,
    make_shape(Int<16>{}, Int<16>{}),
    make_coord(block_m, block_n)
);
auto thr_layout = make_layout(
    make_shape(Int<128>{})
);
auto thr_tensor = local_partition(
    tile,
    thr_layout,
    threadIdx.x
);
```

CUTLASS 4 adds CUTLASS DSLs. These are Python native interfaces for writing high-performance CUDA kernels based on core CUTLASS and CuTe concepts without any performance compromises. With the release of 4.0, we are releasing the first of these in CuTe DSL. 

CuTe DSL 是 Python 版本的 CuTe 风格编程模型；CuTe DSL 属于 CUTLASS DSLs。

### triton

Triton 里写的是 cta_tile 级逻辑，抽象层高，编译器负责翻译

```
grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch * nheads)

_fwd_kernel[grid](
    ...,
    BLOCK_M=BLOCK,
    BLOCK_N=BLOCK,
    num_warps=num_warps,
    num_stages=1,
)

@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _fwd_kernel(
    q = tl.load(q_ptrs)
    qk += tl.dot(q, k, trans_b=True)
    tl.store(out_ptrs, acc_o)
    
@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "SEQUENCE_PARALLEL": False},
            num_warps=8,
            num_stages=1,
            pre_hook=init_to_zero("DQ"),
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "SEQUENCE_PARALLEL": True},
            num_warps=8,
            num_stages=1,
            pre_hook=init_to_zero("DQ"),
        ),
    ],
    key=["CACHE_KEY_SEQLEN_Q", "CACHE_KEY_SEQLEN_K", "BIAS_TYPE", "IS_CAUSAL", "BLOCK_HEADDIM"],
)
@triton.heuristics(
)
@triton.jit
def _bwd_kernel(
    tl.atomic_add(dq_ptrs, dq)

autotune自动生成实参
new key：
    分别运行configs
    测量耗时
    选择最快的config
    缓存key:config
old key：
    key:config
heuristics根据实参补充实参
kernel<<<grid(实参),num_warps>>>(实参)

num_stages: 它会影响 load 和计算重叠方式，以及 shared memory 使用量
num_warps，num_stages有默认值

调用 Kernel 时，Triton 的 Python 包装层会从 x 中取出类似 x.data_ptr() 的设备指针，将它作为参数传给 GPU Kernel
```
### softmax

$$
\frac{exp(x-x_m)}{\sum exp(x-x_m)}
$$

```
一个block处理一行，每个thread处理len/blockDim.x个元素
safe_softmax：减去最大值，避免数值上溢
通过一次reduce获得xmax和sum，再exp(x-xmax)/sum
online_softmax：分块softmax
update xmax
new_sum=(big_sum+small_sum*exp(small-big))
```

### mm

```
每个线程计算C的一个元素，重复读取AB(MNK+MNK)
读取到shared mem，分块矩阵乘法(MNK/B_D+MNK/B_D)
一个线程计算4*4个元素
TO_FLOAT4(x) (*((float4*)(&(x)))) 使用向量指令，减少了内存指令数量
读取到reg，分块矩阵乘法(2B_DB_DB_D->2B_DB_DB_D/dpt)
A先取到reg再转置存于shared mem:也使用float4
双缓冲，减少同步:read0|read1,comp0|read0,comp1|...|compx
tensor core
```

### presum

分块求前缀和（块内部分warp求前缀和）

块和再求前缀和（单线程）

分块加前缀和

### transpose

让读写连续：先确定block begin，按行优先读global，按列优先读shared(按行优先写global)

防止列读造成bank冲突：sm[blockSize,blockSize+1]

## Triton Inference Server

Triton Inference Server：同时管理多个模型，处理并发请求，监控

TensorRT-LLM：pagedattention

TensorRT Inference Server是Triton Inference Server的前身，FasterTransformer是TensorRT-LLM的前身

# nsys

CUPTI（CUDA Profiling Tools Interface）是 NVIDIA 提供的 CUDA 工具接口，专用于 GPU 性能分析和调试。Nsight Systems/Compute，PyTorch Profiler，它们基于 CUPTI 但提供更友好的界面。
```
Nsight Systems:时间轴数据
Nsight Compute:CUDA内核性能
Nsight Graphics:图形渲染
NVTX在应用程序中插入自定义标记和注释，会在时间轴中显示
```

[Developer Tools Download Center | NVIDIA Developer](https://developer.nvidia.com/tools-downloads)

```
nsys_args="/opt/nvidia/nsight-systems/2023.2.1/bin/nsys profile --stats true -w true -t cuda,nvtx,osrt,cudnn,cublas --capture-range=cudaProfilerApi -x true --force-overwrite true -o ${nsys_dir}"
```

```
--stats 总结
-w --show-output 把命令的输出输出到终端
-t Select the API(s) to be traced. 
--capture-range profiling will start only when an appropriate start API or hotkey is invoked
-x true stop collecting automatically when the launched process has exited
-cudabacktrace=true 调用栈
--sample cpu数据
```

SM利用率

```
https://docs.nvidia.com/nsight-systems/UserGuide/index.html#gpu-metrics

nsys profile --gpu-metrics-devices=help
查看是否支持

https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters#TrblShootTag

container must be started with the appropriate permissions by passing --cap-add=SYS_ADMIN as an admin user.

nsys profile --gpu-metrics-devices=1 --gpu-metrics-frequency=500 --capture-range=cudaProfilerApi --capture-range-end=stop --trace=cuda,nvtx,osrt,cudnn,cublas --sample=none --cudabacktrace=none --force-overwrite=true --show-output=true --output= ./my-app

https://dev-discuss.pytorch.org/t/using-nsight-systems-to-profile-gpu-workload/59

torch.cuda.cudart().cudaProfilerStart()    
torch.cuda.nvtx.range_push("start")
torch.cuda.nvtx.range_pop()
torch.cuda.cudart().cudaProfilerStop() 

nsys export --force-overwrite true -t sqlite eevee_nsys.nsys-rep

SELECT timestamp, value FROM ... where ... SMs Active [Throughput %]
The ratio of cycles SMs had at least 1 warp in flight (allocated on SM) to the number of cycles as a percentage. A value of 0 indicates all SMs were idle (no warps in flight). A value of 50% can indicate some gradient between all SMs active 50% of the sample period or 50% of SMs active 100% of the sample period.
```

```
import sqlite3
import matplotlib.pyplot as plt

def get_data(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    # tables = cursor.fetchall()
    # for table in tables:
    #     print(table[0])
    # TARGET_INFO_GPU_METRICS
    # GPU_METRICS

    # cursor.execute("SELECT DISTINCT metricName FROM TARGET_INFO_GPU_METRICS")
    # results = cursor.fetchall()
    # for row in results:
    #     print(row)  
    # ('SMs Active [Throughput %]',)

    # cursor.execute('SELECT COUNT(*) FROM TARGET_INFO_GPU_METRICS')
    # row_count = cursor.fetchone()
    # print(f"The table 'TARGET_INFO_GPU_METRICS' has {row_count[0]} rows.")
    # 37 per gpu

    # cursor.execute("SELECT timestamp, value FROM GPU_METRICS JOIN TARGET_INFO_GPU_METRICS USING (metricId) WHERE value != 0 and metricName == 'SMs Active [Throughput %]'")
    cursor.execute("SELECT timestamp, value FROM GPU_METRICS JOIN TARGET_INFO_GPU_METRICS USING (metricId) WHERE metricName == 'SMs Active [Throughput %]'")
    results = cursor.fetchall()

    cursor.close()
    conn.close()

    return results

def find_first_geq_idx(nums,gate):
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] < gate:
            left = mid + 1
        elif nums[mid] > gate:
            right = mid - 1
        else:
            return mid
    return right

if __name__=="__main__":
    # for name in ["time","space","decouple"]:
    for name in ["blip3o",]:
        t_v=get_data(f'{name}.sqlite')
        t_v = sorted(t_v, key=lambda pair: pair[0],reverse=False)
        t=[t for t,v in t_v]
        v=[v for t,v in t_v]
        print(name,t[0],t[-1])
        start_time=2.9
        end_time=3.6
        start=find_first_geq_idx(t,start_time*1e9)
        end=find_first_geq_idx(t,end_time*1e9)
        plt.plot(t[start:end],v[start:end], label=f"{name}")
        sum_u=sum([v[i]*0.002 for i in range(start,end+1)])
        print(f"avg_util:{sum_u/(end_time-start_time)}")
        
    plt.xlabel('time(ns)')
    plt.ylabel('SMs Active(%)')
    plt.title(f"SMs Active")
    plt.grid(True)
    plt.legend()
    plt.savefig("SMs_Active"+'.png')
    
    '''
    time 421976616 16593890684
    avg_util:27.965000000000025
    space 77979221 9233914229
    avg_util:53.60833333333337
    decouple 243975828 6257920830
    avg_util:89.93666666666691
    '''
```
