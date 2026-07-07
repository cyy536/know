# sample

```
https://huggingface.co/blog/zh/how-to-generate
确定性：
贪心：每个时间步保留最可能的 1 个词
beam：每个时间步保留最可能的 num_beams 个词
n-grams 惩罚是确保每个 n-gram 都只出现一次

随机性：
采样
降低“温度”F.softmax(x/t)：增加高概率单词的似然并降低低概率单词的似然
Top-K：概率最大的 K 个词会被选出，在这重新被归一化概率后的 K 个词中采样
Top-p：在累积概率超过概率 p 的最小单词集中进行

num_return_sequences：beam的最高几个或多次采样
```

# backward

```
zero optimizer:16参数，basic optimizer
basic optimizer:32参数，32*2优化器

前向：16激活，16参数
反向：16激活，16参数，16梯度
更新：16梯度->32，32*2优化器，32参数->16
```

# model

```
res-ln-attn-drop-add-(res-ln-cross-add-)res-ln-ffn-drop-add

llama
https://zhuanlan.zhihu.com/p/632102048
1:RoPE,RMSNorm,SwiGLU
2:GQA

crossattn:
mtm(blip)
vit+bert+bert

append:
mllm(blip2,llava)
vit+bert/ffn+llm

deepstack:
mllm(qwen3vl)
vit+ffn+llm
隔一定层，hidden加上高分辨率token，高分辨率如何获得，看图
H[vis_pos] += Xstack[(idx − lstart)//n]

推理：input,step->diffusion->noise
训练：input+noise,step->diffusion->noise'

diffusion: unet, diffusion transformer
step和条件通过crossattn或append和input交互

BLIP3o
文本(+图像)->(vit+)Qwen3ForCausalLM自回归->[图像特征,...]
for{噪声+step+条件->diffusion->}->vae.decode
```

## RoPE(Q,K)

```
q_embed = (q * cos) + (rotate_half(q) * sin)
```

## RMSnorm

$$
y_i=\gamma_i\frac{x_i}{\text{RMS}(x)}\\
\text{RMS}(x)=\sqrt{\sum x^2_i/n+\epsilon}
$$

## SwiGLu

$$
\text{Swish}(X1)*(X2)
$$

## ffn

$$
\text{Relu}(XW_u+b_u)W_d+b_d\\
\text{SwiGLu}(XW1+b1,XW2+b2)W_d+b_d\\
$$

## moe

打分(N)topk(N)，扩展(K)重排(K)，a2a(X)，ffn(X)，a2a(O)，重排(BL)sum(K)

```
G(X):[BL,N]
每个token对每个专家打分
top_k_values,expert_idx=top_k_values(G):[BLK]
W=softmax(top_k_values):[BLK]

sorted_expert_ids,sorted_indices=expert_idx.sort()
排序后的值在原始张量中的位置
X.expand(k)[sorted_indices]:[BLK,H]
W[sorted_indices]:[BLK]

expert_counts = torch.bincount(sorted_expert_ids, minlength=num_experts)
删去多的，padding少的
X:[N,C,H]
W:[N,C]

all2all(X,p):[N/p,C*p,H]
O=compute(X)
all2all(O)

O:[BLK,H]
W:[BLK]
sum(O*W)=[BL,H]
```

## PatchEmbed

```
img_size=256, patch_size=8, in_chans=3, embed_dim=512
输入一幅256^2通道数为3的图像，每个patch的大小分别为8^2，总patch数目就相当于(256/8)^2=1024个，相当于把图像分为1024个patch，每个patch编码为一个512维的向量。
self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1,
```

# dp

ddp多进程，各自加载数据，各自计算loss，allreduce梯度，各自更新参数

[浅谈Tensorflow分布式架构：ring all-reduce算法 - 知乎](https://zhuanlan.zhihu.com/p/69797852)

```
ring all-reduce(N,p)=reduce-scatter(N,p)+all-gather(N/p,p)
每个进程有数据N，p个进程，接收量[N/p(p-1)]2
接收量：all2all(接收rank)==reduce-scatter(接收rank，累积)==all-gather(接收别人的)
allreduce(O=[B,L,N/p*H][N/p*H,D]=[B,L,D])
all2all(X=[BLKD])
接收量：2/k
```

Pytorch DDP选择在backward的时候进行梯度更新，并且为此引入bucket的概念。一个bucket里通常有不超过一定大小的几个layer的参数组成（默认是25MB），当一个bucket内部的梯度准备好了之后，则这个bucket就可以开始进行ring-all-reduce取平均操作，而不需要等到计算完整个模型的反向传播过后才进行一次统一的ring-all-reduce操作，相当与把多卡通讯的时间隐藏在了backward计算的耗时中，大大提升了资源利用效率。

# megatron

```
tp
attn
X[W,W]=[?,?]
按head划分：每个GPU计算自己的flashattn
[O,O][W;W]
mlp
X[W,W][W;W]
all-reduce*2
大参数，缩短时延，通信量高

pp
大层数，通信量小，气泡，调度microbatch，不能缩短时延

tp+sp，不能解决长序列
https://blog.csdn.net/qinduohao333/article/details/131629428
减少LayerNorm，Dropout计算和显存：切分X的len维度,完整的W
linear：完整的X,切分W的head维度
(all-gather+SA+FFA+reduce-scatter)*2
```

# deepspeed

```
zero
https://zhuanlan.zhihu.com/p/513571706
1:os,反向allreduce/[reduce]梯度,正向all-gather参数
2:os+grad
3:os+grad+param16,反向+all-gather参数
infinity:zero3,卸载到具有足够容量的最快内存
offload:grad+param16,cpuadam

sp
https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/deepspeed-ulysses/chinese/README.md
除QK^TV外，都切分X的len，计算attn前从序列划分变成头划分
[B,L/p,N,H]->[B,L/p*p,N/p,H]
concat(all2all(Q.split(p,dim=2)),dim=1)
all2all*2
接收量BL/pNH*6
head num%P==0
ulysses 整体高于 ring attention，这是因为 ulysses 的 all2all 通信方式优于 ring attention 的 P2P 通信

USP
在外循环使用 DeepSpeed Ulysess 这样在计算 Attention 时序列仍然是完整的，因此在内循环使用 Ring-Attention
```

FSDP 是PyTorch团队受ZeRO-3启发，在PyTorch原生框架内实现的一种具体实现。

# sp

```
cp(ring attn)
https://zhuanlan.zhihu.com/p/689067888
https://zhuanlan.zhihu.com/p/14665512019
切分X的len维度
q for k,v(flashattn2)
p2p传输kv,计算通信重叠,接收量(BLNH*2)
如果将总长度增加到 65536*8，则计算时间远大于通信时间
不同子序列需要atten的kv长度不同
Striped Attention(单向电梯) 
| l0,l4 | l1,l5 | l2,l6 | l3,l7 |
zigzag ring attention(z字分配)
| l0,l7 | l1,l6 | l2,l5 | l3,l4 |
实现完美的平衡仍然具有挑战性
```

# ep

```
all2all*2
[B/p,L*K,H]->[B/p*p,L*K/p,H]

https://zhuanlan.zhihu.com/p/466363675
Hierarchical All-to-all
intra node all2all(内部任务分组)
inter node per gpu all2all
[B/p1p2,L*K,H]->[B/p1p2*p1p2,L*K/p1p2,H]
==
[B/p1,L*K,H]->[B/p1*p1,L*K/p1,H]
[B/p2,L*K,H]->[B/p2*p2,L*K/p2,H]

attn(dp=ep*tp)->expert(ep*tp)
inter node per gpu all2all(ep)
intra node allgather(tp)

attn(tp)->expert(ep)
reduce scatter(B/tp)->all2all

attn(tp1=ep*tp2)->expert(ep*tp2)
reduce scatter(B/tp1)->all2all->all-gather

当k比较大时：all2all*2换成allgather+scatter+full+reducescatter
原因：ring-based communication pattern with only neighboring workers.
MegaScale-MoE
```

# orca

FasterTransformer

请求级调度，static batching

迭代级调度

选择批处理(Continuous Batching)：

```
非attn[sum(L),H]
attn[B,N,L,H],可以有prefill和decode
```

# vllm

算力 7.0 及以上

依赖pytorch

吞吐量比 HuggingFace Transformers 高出 24 倍

按需分配，减少内外碎片

物理分块，逻辑不连续

物理可共享，写入时引用数不为1则分配新的块

[兼容性矩阵 — vLLM](https://docs.vllm.ai/en/latest/features/compatibility_matrix.html)

s-lora

通常，这种调整只应用于self-attention中的以Q、K、V和output的矩阵运算，FFN不使用

作者用OpenAI Triton实现MBGMM计算XAB batch，adapter和kv共同管理

自动前缀缓存（简称 APC）缓存现有查询的 KV 缓存，只减少了处理查询的时间（预填充阶段）

[Automatic Prefix Caching — vLLM](https://docs.vllm.ai/en/latest/design/v1/prefix_caching.html)

```
The block hash will be assigned when the block is full
block_hash=hash(Parent hash value,Block tokens)
Cache blocks: Mapping from hash key to block IDs.
Request blocks: Mapping from request ID to allocated block IDs.
空闲块组成双向链表；块引用为0时，反向插入队尾，有可能恢复使用；取出时，引用加1
prefill：查找Cache blocks；如果在空闲队列中，取出；分配空闲块，清除hash；添加Cache blocks
decode：分配空闲块，清除hash；添加Cache blocks
```

预填充和解码阶段放入不同的 vLLM 实例中，可以灵活地分配不同的并行策略、硬件，控制尾部 ITL 

[LLM PD 分离背后的架构问题 - 知乎](https://zhuanlan.zhihu.com/p/27836625742?utm_psn=1892545875911364697)

DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving

Splitwise: Efficient Generative LLM Inference Using Phase Splitting

推测解码

tp，pp

Chunked Prefill：

默认情况下，vLLM 计划程序会优先考虑预填充，并且不会将预填充和解码批量解码为同一批次。 此策略优化了 TTFT （第一个令牌的时间），但会导致 ITL （令牌间延迟）变慢和 GPU 利用率低下。启用分块预填充后，将更改策略以优先处理解码请求。 在计划任何预填充之前，它会将所有待处理的解码请求批处理到批处理中。 当有可用的 token_budget 时，它会安排待处理的预填充。 如果最后一个待处理的预填充请求无法放入 ，则会对其进行分块。它通过将计算绑定（预填充）和内存绑定（解码）请求定位到同一批次，帮助实现更好的 GPU 利用率。

DeepSpeed-FastGen: High-throughput Text Generation for LLMs via MII and DeepSpeed-Inference

SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills

[架构概述 — vLLM](https://docs.vllm.ai/en/latest/design/arch_overview.html)

## pagedattn

[vLLM皇冠上的明珠：深入浅出理解PagedAttention CUDA实现 - 知乎](https://zhuanlan.zhihu.com/p/673284781)

prefill: 修改了 FlashAttention 的源码来支持 block-table

decode: V1改编自FasterTransformers的MHA实现。V2是参考FlashDecoding方式进行实现。

```
        use_v1 = max_seq_len <= 8192 and (
            max_num_partitions == 1 or num_seqs * num_heads > 512
        )
```

从 vLLM-0.5.x 开始，官方把 FlashInfer 作为可选/默认后端编译进来

prefill和decode不batch，prefill优先

[VLLM Paged Attention 实现 - 峰子的乐园](https://dingfen.github.io/2024/08/02/2024-5-9-vllm/)

```
q=[num_seqs, num_heads, head_size]
k_cache=[num_blocks, num_kv_heads, head_size/x, block_size, x]
v_cache=[num_blocks, num_kv_heads, head_size, block_size]
block_tables映射表，表示每个sequence映射到哪几个block上
context_lens=[num_seqs]//用于变长，已经生成的 tokens
grid=[num_heads,num_seqs](粒度和flashattn一样)
每个thread block的任务量不同？context len不同
PA只有一层循环
for k
	s=qk^T
shared memory:p=softmax(s)
for v
	o=pv
```

BLOCK_SIZE 指的是每个块中 token 的数量

每个 warp 一次处理一个 query token 和一整块（16）的 key tokens 之间的计算（可能会在多次迭代中处理多个块）

Thread group 一次计算一个 query 与一个 key token， THREAD_GROUP_SIZE 为 2

每个 thread group 会 load 一个完整的 query token 张量

一个thread group一次加载16B，一个线程一次加载的Q或K元素数量VEC_SIZE 将为 16/2/2=4，x=2*4=8

[vLLM V1: A Major Upgrade to vLLM’s Core Architecture | vLLM Blog](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html)

vLLM V1 was integrating FlashAttention 3. Given the high level of dynamism in V1—such as combining prefill and decode within the same batch—a flexible and high-performance attention kernel was essential. 

缺乏对请求优先级的智能调度

根据请求长度分布动态调整分页大小，一个block要启动一次卸载

# FlashInfer

FlashInfer 是一个用于大型语言模型的库和内核生成器，提供 LLM GPU 内核（如 FlashAttention、SparseAttention、PageAttention、Sampling 等）的高性能实现。FlashInfer 专注于 LLM 服务和推理，并在不同场景中提供最先进的性能。

[用FlashInfer加速大语言模型推理中的自注意力操作 - 知乎](https://zhuanlan.zhihu.com/p/681506469)

单请求/批处理版本的预填充，解码和追加的自注意力算子，涵盖了不同格式的KV-Cache（变长数组，页表等等）。

# sglang

专为复杂多轮交互设计的LLM推理引擎

RadixAttention通过前缀树结构复用多轮对话的共享KV缓存

# mooncake

https://zhuanlan.zhihu.com/p/706317088

prefill和decode分离，分布式多级kv pool，分块预填充流水线并行

当请求的预填充可以在不分块的情况下转发且不会损害 TBT SLO 时，才会内联到解码批处理中

根据系统负载确定是否接受或拒绝传入请求

按层传输

```
2∗N∗(B∗L∗H∗D∗P)
```

在 P 和 D 之间做负载均衡，要么从整个节点层面直接切换 P 和 D 的角色，要么 P 和 D 节点能够承担一些混杂的请求

# AE分离

MegaScale-Infer：对MOE来说，专家很多，分配给每个专家的token数量很少，导致FFN需要更大的 batch size才能充分GPU

batch size不能无限增大：时延要求，显存限制 (KV)

分离+流水线：

增大FFN的batch size

各自选并行，Attnention采用DP+TP，FFN采用EP+TP

各自选择硬件，Attention选择IO更好的硬件，FFN选择计算更好的硬件

通信库去掉proxy buffer，直接注册user memory为RDMA可访问区域

# deepseek

```
https://zhuanlan.zhihu.com/p/27777601573
DeepEP provides high-throughput and low-latency all-to-all GPU kernels
通信库，训练和推理都能用
让专家最多分布在M个设备中，在每次选择Topk之前，先对筛选出M个有着最高亲和性的设备
deepseek则通过给门控激活值引入bias item来实现负载均衡与模型性能之间的平衡
EPLB采用了 “冗余专家” 策略，即复制高负载的专家，然后将这些复制后的专家合理分配到各个 GPU 上，以实现负载均衡。

MLA(KV->C)
降低Nhead
逐步升高Nhead
https://zhuanlan.zhihu.com/p/27976368445
C_q=XW_dq[d,4H]
Q_c=C_qW_uq[4H,DH]
Q_r=RoPE(C_qW_qr)[4H,0.5DH]
从C
Q_i=[Q_c_i,Q_r_i][1.5H]

C_kv=XW_dkv[d,4H]
[K_c,V_c]=C_kv[W_uk,W_uv][4H,DH]
K_r=RoPE(XW_kr)[d,0.5H]
从X
K_i=[K_c_i,K_r][1.5H]

Q_iK_i^T=[Q_c_i,Q_r_i][K_c_i^T;K_r^T]
        =[C_qW_uq_iW_uk_i^T,Q_r_i][C_kv^T;K_r^T]
        =[L,4H][4H,L]+...
S_i=act(Q_iK_i^T)V_i=act(Q_i[C_kv,K_r]^T)C_kv[L,4H]
O=[S_iW_uv_i,][W_o_i;]=[S_i,][W_uv_iW_o_i;]
计算量增大4倍
FlashMLA is an efficient MLA decoding kernel for Hopper GPUs
只存储C_kv=[B,L,4*H],K_r=[B,L,H/2]
MHA=2*[B,L,D/p,H],MQA=2*[B,L,1,H],GQA

MTP
增加head，多头同时生成token
提升训练和推理的性能
{推理时一个head生成一个token，并行验证}

https://zhuanlan.zhihu.com/p/22681871459
优化器前后都有空泡(等腰梯形)
ZB-H1:将后向拆分为B、W，优先B，优化器前没有空泡了(直角梯形)，激活存储不变最大p
ZB-H2:前向提前，优化器提前，优化器后面没有空泡(平行四边形)，激活最大2p-1
ZB-V:交错成V
forward变小，平行四边形缩短
计算：MLP(B),MLP(W),MLP(F),ATTN(B),ATTN(W),ATTN(F)
通信：DISPATCH(F),DISPATCH(B),COMBINE(F),COMBINE(B)
增加一条反向流水线（参数翻倍），同时计算一个前向和反向，计算通信重叠
X

fp8 train
DeepGEMM 是一个专为干净、高效的 FP8 通用矩阵乘法 (GEMM) 而设计的库，具有细粒度缩放功能，如DeepSeek-V3中所述。它支持普通和混合专家 (MoE) 分组 GEMM

https://zhuanlan.zhihu.com/p/20844750193
v3-base+grpo=r1-zero
v3-base+sft+grpo=r1(数据集需要zero产生)

https://zhuanlan.zhihu.com/p/27001519472
Fire-Flyer File System是一种高性能分布式文件系统
Smallpond是基于3FS构建的轻量级数据处理框架。

https://zhuanlan.zhihu.com/p/24841366485
Native Sparse Attention做预训练的原生模型就是稀疏的注意力结构，这里的稀疏化是指KV序列的稀疏化
降低L
压缩倍数7.88
prefill+decode(cot+answer)
分块压缩KV（MLP层），选择topk（当前token和压缩token的attn score）块对应原序列，最近窗口，加权求和
压缩的目的是把一段长度为8的序列压成c个序列
NSA的注意力算子类似Flash-Attention2
NSA精度优于MHA
训练速度：在64k上下文下，NSA的前向传播速度提升9倍，反向传播速度提升6倍，得益于硬件对齐设计。
解码速度：NSA在64k上下文下的解码速度提升11.6倍，显著降低了解码延迟，尤其随着序列长度增加。
稀疏序列，减少计算和加载，不能降低存储
mla稀疏head，增加计算和加载，能够降低存储
```

# flexgen

[FlexGen论文笔记 - 知乎](https://zhuanlan.zhihu.com/p/664164593)

LLM推理的内存占用主要来自模型权重和KV缓存。

z字调度:一层权重，n个batch

把各类tensor即weights, activations, KV cache分别按比例切分，保存在不同存储介质上

disk空间映射成virtual memory来加快访存速度，直接读写page cache

读写文件需要：用户内存-page cache-disk

可搭配pp

WD-4bit，token稀疏

# InfiniGen

核心是利用 CPU 内存，生成新令牌时，把大部分 KV 缓存放CPU内存，借助前一层注意力输入X，预判当前层关键令牌，针对性加载其键值到 GPU 计算，舍弃不重要令牌

把 KV 缓存当动态资源池在 CPU 内存维护，当 CPU 内存达用户定义上限，启动管理机制，选淘汰策略剔除不常用 KV 条目。

# powerinfer

relu后有很多0，说明W1有很多列的计算是没有用的，W2很多行不用参与计算

它根据离线数据对神经元进行分类，将热激活的神经元（例如，索引 3、5、7）分配给 GPU 内存，将其他神经元分配给 CPU 内存。在接收到输入后，预测器会识别当前层中的哪些神经元可能被激活。例如，它预测神经元 3、4 和 5 的激活。然后，CPU 和 GPU 都会处理预测的活跃神经元，忽略不活跃的神经元。GPU 计算神经元 3 和 5，而 CPU 处理神经元 4。一旦神经元 4 的计算完成，其输出就会发送到 GPU 进行结果集成。

PowerInfer 是基于 llama.cpp 框架做的

# quantize

降低存储，权重、激活、KV Cache的sizeof(dtype)

量化激活和权重仅限于8位，在非线性op前激活需要反量化16位，在非线性op后要重新量化

根据量化过程中是否引入零点偏移（zero point），可将大语言模型量化分为对称量化与非对称量化两大类。对称量化与非对称量化的核心区别在于浮点数零点与量化后整数空间零点是否重合，这直接决定了算子在执行计算时的逻辑复杂度与精度表现。对称量化采用关于零对称的映射范围，仅通过一个缩放因子$s$进行线性变换，使得浮点数的零点在量化后依然对应整数零点，这种设计的最大优势在于矩阵乘法等核心算子在执行时无需处理零点偏移带来的额外项，极大地简化了GPU的指令流，提升了推理效率，但在处理如ReLU激活\cite{glorot2011deep}后仅分布在正数域的数据时，由于必须在负数区间预留一半的编码位，会导致严重的比特利用率浪费。相比之下，非对称量化通过引入零点偏移量$z$，将浮点域的实际动态范围完整映射到整数目标区间，这种方式能够更紧凑地拟合原始数据分布，从而在相同位宽下获得更高的表示精度，特别是在处理非对称分布的激活值时优势明显，但在算子实现层面，由于零点偏移量的存在，计算过程中必须实时处理由于零点不为零而产生的额外偏移校准项，这在一定程度上增加了硬件的算力开销与逻辑控制复杂度。
$$
s &= \frac{\symup{max}(|x_{float}|)}{x^{max}_{int}}\\
        x_{int} &= \symup{round}\left( \frac{x_{float}}{s} \right)\\
        x_{float} &= x_{int} \times s
$$

$$
s &= \frac{x^{max}_{float} - x^{min}_{float}}{x^{max}_{int} - x^{min}_{int}}\\
        z &= x^{min}_{int} - \symup{round}\left( \frac{x^{min}_{float}}{s} \right)\\
        x_{int} &= \symup{round}\left( \frac{x_{float}}{s} + z \right)\\
        x_{float} &= (x_{int} - z) \times s
$$

对称量化与非对称量化仅定义了浮点数与低精度整数之间的基础数值映射规则，决定量化过程中是否引入零点偏移、量化区间是否关于原点对称，属于量化的底层数学实现方式。而 ZeroQuant\cite{yao2022zeroquant}、LLM.int8()\cite{dettmers2022gpt3}、SmoothQuant\cite{xiao2023smoothquant}、GPTQ\cite{frantar2022gptq}、AWQ\cite{lin2024awq}、LUT-GEMM\cite{park2022lut}等量化算法，则是面向大语言模型特性设计的整套量化优化策略，进一步解决激活值异常波动、权重分布不均、量化误差逐层累积、硬件计算效率低下等大语言模型特有的量化难题。



AQ+AD

zeroquant

模型参数达到6.7B时，hidden state中存在一些绝对值很大的离群值，分布在每个token的特定channel，保留fp16

```
[diag(X16)X8,X16][W8diag(W16);W16]=diag(X16)X8W8diag(W16)+X16W16
```

[bitsandbytes/bitsandbytes/nn/modules.py at main · bitsandbytes-foundation/bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/bitsandbytes/nn/modules.py#L846)

相比于 FP16，LLM.int8() 慢了大约 15% 到 23%

SmoothQuant：将激活值一部分量化难度转移到权重上，这种技术可以使用INT8算术单元（即W8A8）进行高效计算

```
per-token activation quantization适合gemm，但离群值出现在特定channel
diag(X16)X8W8diag(W16)
diag(W16)=[out,out]

smooth将特定channel的难度传递给W的特定行(input channel)
X16diag(s)^{-1}diag(s)W16=X16'W16'

smooth:
获取激活per channel最大值，获取参数per input channel最大值，计算scale
缩放layernorm和线性层参数

s可以是任意值，但要降低量化难度
sj=max(|Xj|)^{alpha}/max(|Wj|)^{1-alpha}
norm刚好作用于hidden，将激活的缩小融合至pre-ln
只在第一个W smooth

参数per tensor，激活per tensor或per token
获取激活per tensor最大值 ，计算激活静态scale
获取参数per tensor最大值 ，计算参数静态scale，(保存激活静态scale，)计算反量化scale(乘以激活静态scale)
(激活per tensor，缩放layernorm)

激活per-tensor:
pre-ln:
    w/=is
linear:
    quant_scale=is
    dequant_scale=is*ws
    forward(i=8/16,o=16):
        out=dequant_scale*(XW) if has pre-ln#i=8
        out=dequant_scale*(X/quant_scale*W) else#i=16

激活per-token:
quant_scale=x.abs().max(dim=-1, keepdim=True)[0].div(127.0)
#[seqlen,1]
dequant_scale=quant_scale*ws
out=dequant_scale*(X/quant_scale*W)
```

int8参数还是太大了，带宽开销大

GPTQ：只4位量化权重，需要反量化到FP16再计算

gptq可以用于其他量化格式

[QLoRA、GPTQ：模型量化概述 - 知乎](https://zhuanlan.zhihu.com/p/646210009)

计算每个参数对目标的影响，确定了参数剪枝的次序，每次剪枝一个参数，其他的参数更新一次

目标：参数量化前后，给同样的输入，输出结果的差异要尽可能小。

```
[out,in]
1假设参数矩阵的同一行参数互相之间是相关的，而不同行之间的参数互不相关。
2不需要找影响最小的参数，按列顺序量化对精度影响也不大
1+2:参数矩阵所有行的量化可以并行
Lazy Batch-Updates:可以先记录第 i 列的更新量，在量化到第 i 列时，再一次性更新参数，这样就可以减少 IO 的次数。将参数矩阵按每 128 列划分为一个个 block，量化某一列时，block 内的参数立即更新，而 block 后面的列只记录更新量，延迟更新。当一个 block 的参数全部量化完成，再统一对后面的所有参数做一次更新。
```

```
从前往后逐个linear：计算H,scale,zero，更新参数
替换线性层，量化参数(scale,pack)
scale=[output_channel,num_group]
在AQ+AD的基础上缩小group
qweight=(in/32*bit,out)
一个元素的4bit连续存储
qzeros=(in/group_size,out/32*bit)
bs>=17，反量化+fp16mm
```

awq: 类似smoothquat，缩放W的行后再量化

AWQ在将大型语言模型量化到较低比特数时，通常能够提供优于或至少与GPTQ相当的精度；GPTQ和AWQ之间的推理速度比较并不像精度比较那样清晰明了

LUT-GEMM：激活生成表，权重生成索引，表存放在共享内存
$$
W=\sum B_i*A_i+Z\\
XW=\sum XB_i*A_i+XZ\\
B\in[-1,1]
$$

```
Ai=[(gs*)num_group,o],Z=[(gs*)num_group,o]
Ai=2^i*scale,Z=-scale*zero+...
W=pack(unpack(W_gptq))
一个元素的4bit不连续存储
算出sum_[.<j<.](+/-Xkj)的所有可能，存于表中
表的索引Bi_[.<j<.]l，全1
查出数据后乘以系数Ai，Z，再求和
bs>=17，反量化+fp16mm
```

W8A8-INT和W4A16-INT精度差不多1-3% accuracy degradation，速度各有优势，比fp16快

W8A8-FP无损，需要H卡

vllm框架集成了用于W8A8（FP和INT）的优化CUTLASS内核，以及用于W4A16-INT的混合精度Marlin内核。其他方法失去了显著的准确性，并且在高性能推理框架中缺乏有效的支持

```
class ColumnParallelLinear(LinearBase):
    self.quant_method.create_weights(

class GPTQMarlinLinearMethod(LinearMethodBase):
    kernel_type = choose_mp_linear_kernel(mp_linear_kernel_config)
    qweight = PackedvLLMParameter(
    layer.register_parameter("qweight", qweight)
    layer.register_parameter("g_idx", g_idx)
    layer.register_parameter("scales", scales)
    layer.register_parameter("qzeros", qzeros)
    self.kernel = kernel_type

class MarlinLinearKernel(MPLinearKernel):
    output = ops.gptq_marlin_gemm(reshaped_x,

torch.ops._C.gptq_marlin_gemm(
在PyTorch中，自定义算子通常通过torch.ops命名空间注册，其中_C模块通常代表C++扩展。
```

QServe:

W4A8结合W4A16和W8A8的优点

分块矩阵乘法，最后需要加法

W8A8先矩阵乘法再累加再反量化fp16。per token+per channel，分块矩阵的scale是相同的

W4A16先反量化再矩阵乘法再累加

W4A4先矩阵乘法再反量化fp16再累加。group size小于input channel size，分块矩阵的scale不同

反量化开销大，W4A4寄存器开销大：int32->fp16，矩阵乘法结果是int32

W4A8先反量化INT8再矩阵乘法再累加再反量化fp16，并没有解决问题？

KV4除了降低存储，还降低了attn的带宽需求

动态量化是指在模型的量化过程中，根据模型不同层或不同参数的重要性以及对精度的影响程度，动态地选择不同的量化位宽

