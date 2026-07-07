```
docker run -itd --ipc host --net host --gpus all --name cyy_fa -v ~/flashattn:/flashattn pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel bash
```
# v1

2-4X speedup. FlashAttention has memory linear in sequence length. Our implementation uses Apex's FMHA code as a starting point.

v1.0.9

```
O = softmax(QK^T*scale)V
= exp(QK^T*scale)/sum(exp(QK^T*scale))V
= exp(QK^T*scale)V/sum(exp(QK^T*scale))
= exp((QK^T-m)*scale)V/sum(exp((QK^T-m)*scale))

lse=log(sum(exp(QK^T*scale)))>=QK^T*scale

sum(exp(QK^T-m)*scale)
=exp(log(sum(exp(QK^T*scale)))-m*scale)
=exp(lse-m*scale)

m=max(QK^T,lse_old/scale)
```
```
QK^T可分块计算，分块结果独立，所有历史块的exp((QK^T-m)*scale)每次都要更新
exp((QK^T-m)*scale)V可分块计算，分块结果求和
可逐块完成exp((QK^T-m)*scale)V/sum(exp((QK^T-m)*scale))：算完一块，更新m，denominator_new，lse，numerator_old，累加，除以denominator_new
```

## cuda
gmem-reg-smem
```
LDG	Load Global	    global memory -> register
LDS	Load Shared	    shared memory -> register
STS	Store Shared	register -> shared memory
STG	Store Global	register -> global memory
```
forward
```
grid=[b, h, q_num_splits]
//q head is split, blockIdx.z does q_tile_id%q_num_splits==blockIdx.z
for loop_step_idx in range(k_tile_num):
    qkv.ldg
    lse.ldg
    qkv.sts.lds
    
    for (int l = blockIdx.z; l < q_tile_num-begin; l += q_num_splits) 
        QK^T
        
        o_tmp.ldg
        q.ldg
        
        mask
        
        lse.sts
        lse/scale
        lse.ldg

        m_new
        exp((x-m_new)*scale)
        denominator_new=sum(exp((x-m_new)*scale))
        
        s.stg
        q.sts

        p_numerator V

        m_new_scale
        lse.lds
        o.sts

        out_numerator_old_scale=exp(lse_old-m_new_scale)
        denominator_new+=out_numerator_old_scale

        lse_new=m_new_scale+log(denominator_new)
        lse.stg

        out_numerator_old*=out_numerator_old_scale
        o.lds

        o*=1.f/denominator_new
        o.stg
        o_tmp.stg
        
        q.lds
```
flash-attention/flash_attn/flash_attn_interface.py
```
def flash_attn_unpadded_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                             dropout_p, softmax_scale=None, causal=False, return_attn_probs=False,
                             deterministic=False):
        q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
        cu_seqlens_q: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into q.
        max_seqlen_q: int. Maximum query sequence length in the batch.
        dropout_p: float. Dropout probability.
        deterministic: bool. Whether or not to ensure deterministic execution.
    return FlashAttnFunc.apply(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,

class FlashAttnFunc(torch.autograd.Function):
    def forward(ctx, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p,
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        out, softmax_lse, rng_state, S_dmask = _flash_attn_forward(
            q, k, v, torch.empty_like(q), cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
            dropout_p, softmax_scale, causal=causal, return_softmax=return_softmax
        )
        ctx.save_for_backward(q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state)
        ctx.dropout_p = dropout_p
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.deterministic = deterministic
        return out if not return_softmax else (out, softmax_lse, S_dmask)
    def backward(ctx, dout, *args):
        _flash_attn_backward(
            rng_state=rng_state, num_splits=1 if ctx.deterministic else 0,

def _flash_attn_forward(q, k, v, out, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                        dropout_p, softmax_scale, causal, return_softmax, num_splits=0,
                        generator=None):
    num_splits: how much to parallelize over the seqlen_q dimension. num_splits=0 means
    it will be set by an internal heuristic. We're exposing num_splits mostly for benchmarking.
    softmax_lse, rng_state, *rest = flash_attn_cuda.fwd(
        q, k, v, out, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p,
        softmax_scale, False, causal, return_softmax, num_splits, generator
    )
    S_dmask = rest[0] if return_softmax else None
    return out, softmax_lse, rng_state, S_dmask

def _flash_attn_backward(dout, q, k, v, out, softmax_lse, dq, dk, dv, cu_seqlens_q, cu_seqlens_k,
    num_splits: whether to parallelize over the seqlen_k dimension (num_splits > 1) or
    not (num_splits = 1). num_splits=0 means it will be set by an internal heuristic.
    _, _, _, softmax_d = flash_attn_cuda.bwd(
        dout, q, k, v, out, softmax_lse, dq, dk, dv, cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k, dropout_p, softmax_scale, False, causal,
        num_splits, generator, rng_state)
    return dq, dk, dv, softmax_d
```
flash-attention/csrc/flash_attn/fmha_api.cpp
```
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Fused Multi-head Self-attention";
    m.def("fwd", &mha_fwd, "Forward pass");
    m.def("bwd", &mha_bwd, "Backward pass");
    m.def("fwd_block", &mha_fwd_block, "Forward pass (blocksparse)");
    m.def("bwd_block", &mha_bwd_block, "Backward pass (blocksparse)");
}

//max_seqlen_k,max_seqlen_q需要padding
std::vector<at::Tensor>
mha_fwd(const at::Tensor &q,         
        const bool zero_tensors,//false

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    bool is_dropout = p_dropout > 0.0;
    Launch_params<FMHA_fprop_params> launch_params(dprops, stream, is_dropout, return_softmax);

    TORCH_CHECK(out.dtype() == q_dtype);

    int blocksize_c = head_size > 64 ? 128 : 256;
    // k tile size
    int max_seqlen_k = ((max_seqlen_k_ + blocksize_c - 1) / blocksize_c) * blocksize_c;
    if( max_seqlen_k_ <= 128 ) {
        max_seqlen_k = 128;
    } else if( max_seqlen_k_ <= 256 ) {
        max_seqlen_k = 256;
    }
    //大于256，blocksize_c的整倍
    int max_seqlen_q = ((max_seqlen_q_ + 16 - 1) / 16) * 16;

    bool loop = max_seqlen_k > blocksize_c;
    // k tile num >1, need old out
    auto opts = q.options();
    if (loop) { o_tmp = torch::empty({total_q, num_heads, head_size}, opts.dtype(at::kFloat)); }
    //old out dtype==fp32
    auto softmax_lse = torch::empty({batch_size, num_heads, max_seqlen_q}, opts.dtype(at::kFloat));
    //log(sum(exp(x*scale)))
    at::Tensor s;
    if (return_softmax) { s = torch::empty({ batch_size, num_heads, max_seqlen_q, max_seqlen_k }, opts); }

    set_params_fprop(launch_params.params,
                     batch_size,
                     max_seqlen_q,
                     max_seqlen_k,
                     num_heads,
                     head_size,
    run_fmha_fwd(launch_params);

void set_params_fprop(FMHA_fprop_params &params,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t h,
                      const size_t d,
    params.q_row_stride_in_elts = q.stride(0);
    //token_stride
    params.k_row_stride_in_elts = k.stride(0);
    params.v_row_stride_in_elts = v.stride(0);
    params.q_head_stride_in_elts = q.stride(1);
    params.k_head_stride_in_elts = k.stride(1);
    params.v_head_stride_in_elts = v.stride(1);

//根据headdim范围选kernel，设置headdim tilesize，headdim tilenum==1
void run_fmha_fwd(Launch_params<FMHA_fprop_params> &launch_params) {
    if (launch_params.params.d <= 32) {
        run_fmha_fwd_hdim32(launch_params);
    } else if (launch_params.params.d <= 64) {
        run_fmha_fwd_hdim64(launch_params);
    } else if (launch_params.params.d <= 128) {
        run_fmha_fwd_hdim128(launch_params);
    }
}
```
flash-attention/csrc/flash_attn/src/fmha_fwd_hdim32.cu
```
void run_fmha_fwd_hdim32(Launch_params<FMHA_fprop_params> &launch_params) {
    FP16_SWITCH(launch_params.params.is_bf16, ([&] {
        if (launch_params.params.seqlen_k == 128) {
            using Kernel_traits = FMHA_kernel_traits<128, 32, 16, 1, 4, 0x08u, elem_type>;
            run_fmha_fwd_loop<Kernel_traits>(launch_params);
        } else if (launch_params.params.seqlen_k >= 256) {
            using Kernel_traits = FMHA_kernel_traits<256, 32, 16, 1, 4, 0x08u, elem_type>;
            run_fmha_fwd_loop<Kernel_traits>(launch_params);
        }
    }));
}
```
flash-attention/csrc/flash_attn/src/fmha/kernel_traits.h
```
template<int S, int D, int STEP, int WARPS_M, int WARPS_N, uint32_t FLAGS = 0x08u, typename elem_type_=__half>
//k tilesize,headdim tilesize,q tilesize,warp in q,warp in k
struct FMHA_kernel_traits {
    // The CTA description for the 1st GEMM.
    using Cta_tile_p = fmha::Cta_tile_extd<STEP, S, D, WARPS_M, WARPS_N, 1>;
    //[q tilesize,headdim tilesize][headdim tilesize,k tilesize] 
    //warp in q,warp in k,warp in headdim
    // The CTA description for the 2nd GEMM.
    using Cta_tile_o = fmha::Cta_tile_extd<STEP, D, S, WARPS_M, 1, WARPS_N>;
    //[qt,kt][kt,ht] 
    //warp in q,warp in headdim,warp in k
    //128 thread 都分配在kt

    // Do we use one buffer for K and V.
    //1
    static constexpr bool SHARE_SMEM_FOR_K_AND_V = (FLAGS & 0x08u) != 0u;
    // Do we keep K in registers.
    //1
    static constexpr bool K_IN_REGS = (FLAGS & 0x10u) == 0u;
    // Do we keep V in registers.
    //1
    static constexpr bool V_IN_REGS = (FLAGS & 0x100u) == 0u;

    // The global memory tile to load Q.
    using Gmem_tile_q = fmha::Gmem_tile_qkv<Cta_tile_p, fmha::BITS_PER_ELEMENT_A, STEP, D>;
    //[qt,ht]
    // The shared memory tile to swizzle Q.
    //缓冲
    using Smem_tile_q = fmha::Smem_tile_a<Cta_tile_p, fmha::Row, Gmem_tile_q::BYTES_PER_LDG, 2>;

    // The global memory tile to load K.
    using Gmem_tile_k = fmha::Gmem_tile_qkv<Cta_tile_p, fmha::BITS_PER_ELEMENT_B, S, D>;
    //[kt,ht]
    // The shared memory tile to swizzle K.
    using Smem_tile_k = fmha::Smem_tile_b<Cta_tile_p, fmha::Col>;

    // The global memory tile to load V.
    using Gmem_tile_v = fmha::Gmem_tile_qkv<Cta_tile_o, fmha::BITS_PER_ELEMENT_B, S, D>;
    // The shared memory tile to swizzle V.
    using Smem_tile_v = fmha::Smem_tile_v<Cta_tile_o>;

    // The global memory tile to store O.
    //o_tmp: store new out, load old
    //o: store final out
    using Gmem_tile_o = fmha::Gmem_tile_o<Cta_tile_o>;

    // The shared memory tile for O.
    //ldg out_old, sts out_numerator_new, update out_numerator_old, lds add, out_numerator/denominator
    using Smem_tile_o = fmha::Smem_tile_o<Cta_tile_o>;;

    // The global memory tile to load/store S.
    //score: store exp((x-m)*scale)，m=max(max(x),log(sum(exp(x))))
    using Gmem_tile_s = fmha::Gmem_tile_mma_s<Cta_tile_p>;

    // The global memory tile to store the softmax sum.
    //softmax_lse: store new log(sum(exp(x*scale))), load old
    using Gmem_softmax_sum = fmha::Gmem_summary_stats<Cta_tile_p>;

    // The shared memory tile to store dp sum.
    //softmax_lse: sts lse_old, lds lse_old，用于线程之间交换数据
    using Smem_dp_sum = fmha::Smem_tile_dp_sum<Gmem_tile_q, 2>;
```
### gemm
```
Cta_tile: qkvh tilesize,warp num
Hmma_tile: tensor core计算量，计算次数
Fragment: tensor core寄存器
```
flash-attention/csrc/flash_attn/src/fmha/gemm.h
```
template<int M, int N, int K, int WARPS_M, int WARPS_N, int WARPS_K>
using Cta_tile_extd = Cta_tile_<M, N, K, WARPS_M, WARPS_N, WARPS_K>;

template<
    // The number of rows in the CTA tile.
    int M_,
    // The number of cols in the CTA tile.
    int N_,
    // The number of elements in the the K dimension of the GEMM loop.
    int K_,
    // The number of rows of warps.
    int WARPS_M_,
    // The number of cols of warps.
    int WARPS_N_,
    // The number of warps in the K dimension of the GEMM loop.
    int WARPS_K_>
struct Cta_tile_ {
    // The number of warps per CTA.
    static constexpr int WARPS_PER_CTA = WARPS_M * WARPS_N * WARPS_K;

//Mma_tile 描述 Tensor Core 如何切分 CTA tile 的类型。
template<typename Cta_tile>
struct Hmma_tile {
    // The number of elements computed with a single warp-MMA.
    static constexpr int M_PER_MMA = 16, N_PER_MMA = 16, K_PER_MMA = 16;
    //tensor core计算量

    // The number of elements computed with a single CTA-MMA.
    static constexpr int M_PER_MMA_PER_CTA = M_PER_MMA * Cta_tile::WARPS_M,
        N_PER_MMA_PER_CTA = N_PER_MMA * Cta_tile::WARPS_N,
        K_PER_MMA_PER_CTA = K_PER_MMA * Cta_tile::WARPS_K;
    //cta计算量

    // The number of MMAs needed to compute the GEMM.
    static constexpr int MMAS_M = DivUpConstexpr(Cta_tile::M, M_PER_MMA_PER_CTA),
        MMAS_N = DivUpConstexpr(Cta_tile::N, N_PER_MMA_PER_CTA),
        MMAS_K = DivUpConstexpr(Cta_tile::K, K_PER_MMA_PER_CTA);
    //计算次数


template< typename Layout >
struct Fragment_a : public Fragment<uint16_t, 8> {
};

template< typename Layout >
struct Fragment_b : public Fragment<uint16_t, 8> {
};

template<
    // The type of the elements.
    typename Data_type_,
    // The number of elements.
    int NUM_ELTS_,
    // The alignment if you want to force a value -- use 0 otherwise.
    int ALIGNMENT_ = 0,
    // The base class.
    typename Base_ = Fragment_base_<Data_type_, NUM_ELTS_, 8 * sizeof(Data_type_), ALIGNMENT_>
>
struct alignas(static_cast<int>(Base_::ALIGNMENT)) Fragment : public Base_ {
    static constexpr int BYTES_PER_LOAD_STORE = Base_::NUM_REGS * sizeof(uint32_t);
    uint32_t regs_[Base_::NUM_REGS];
    inline __device__ uint32_t& reg(int ii) {
        return this->regs_[ii];
    }
    inline __device__ void clear() {
        #pragma unroll
        for( int ii = 0; ii < Base_::NUM_REGS; ++ii ) {
            asm volatile("mov.u32 %0, 0; \n" : "=r"(this->reg(ii)) : );
        }
    }

template< typename Data_type_, int NUM_ELTS_, int BITS_PER_ELT_, int ALIGNMENT_ >
struct Fragment_base_ {
    // The size of byte of a single register.
    static constexpr int BYTES_PER_REG = 4;
    // The number of registers needed to store the fragment.
    static constexpr int NUM_REGS = DivUpConstexpr(NUM_ELTS * BITS_PER_ELT, BITS_PER_REG);
    static constexpr int ALIGNMENT = ALIGNMENT_ > 0 ? ALIGNMENT_ : MinConstexpr(NUM_REGS * BYTES_PER_REG, 16);

struct Fragment_accumulator : public Fragment<float, 8> {

template<typename elem_type, typename Acc, typename A, typename B, int M, int N>
inline __device__ void gemm_cl(Acc (&acc)[M][N], const A (&a)[M], const B (&b)[N]) {
    //[Mma_tile_p::MMAS_M][Mma_tile_p::MMAS_N]次数
    using Shape = cutlass::gemm::GemmShape<16 * M, 16 * N, 16>;
#if defined(__CUDA_ARCH__) &&  __CUDA_ARCH__ >= 800
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

    using WarpMma = typename cutlass::gemm::warp::DefaultMmaTensorOp<
    constexpr int kIters = Shape::kK / InstructionShape::kK;
    //按tensor core大小切分，向量内积

    using FragmentA = typename WarpMma::ArchMmaOperator::FragmentA;
    using FragmentB = typename WarpMma::ArchMmaOperator::FragmentB;
    using FragmentC = typename WarpMma::FragmentC;
    FragmentA a_cl[kIters][M];
    FragmentA b_cl[kIters][N];

    constexpr int kRegs = InstructionShape::kK == 16 ? 4 : 2;
    #pragma unroll
    for (int iter = 0; iter < kIters; iter++) {
        #pragma unroll
        for (int mi = 0; mi < M; mi++) {
            uint32_t *a_ptr = a_cl[iter][mi].raw_data();
            #pragma unroll
            for (int ki = 0; ki < kRegs; ki++) {
                a_ptr[ki] = a[mi].regs_[iter * kRegs + ki];
            }
        }
    }
    #pragma unroll
    for (int iter = 0; iter < kIters; iter++) {
        #pragma unroll
        for (int ni = 0; ni < N; ni++) {
            uint32_t *b_ptr = b_cl[iter][ni].raw_data();
            #pragma unroll
            for (int ki = 0; ki < kRegs; ki++) {
                // b_ptr[ki] = b[ni].regs_[iter * kRegs + ki];
                // TD [2022-06-02] For some reason the order for frag_b is different.
                b_ptr[ki] = b[ni].regs_[InstructionShape::kK == 16 ? iter * kRegs + ki : ki * kRegs + iter];
            }
        }
    }

    WarpMma mma_op;
    // mma_op(c_cl, a_cl, b_cl, c_cl);
    #pragma unroll
    for (int iter = 0; iter < kIters; iter++) {
        mma_op(c_cl, reinterpret_cast<const typename WarpMma::FragmentA (&)>(a_cl[iter]),
               reinterpret_cast<const typename WarpMma::FragmentB (&)>(b_cl[iter]), c_cl);
        /*
        for mi in M:
            for ni in N:
                acc[mi][ni] += a[mi] @ b[ni]
                //tensor core
        */
    }

    // The modified c_cl is not copied back into acc, idk why
    #pragma unroll
    for (int mi = 0; mi < M; mi++) {
        #pragma unroll
        for (int ni = 0; ni < N; ni++) {
            #pragma unroll
            for (int i =0; i < 8; i++) {
                acc[mi][ni].elt(i) = c_cl[mi * N * 8 + ni * 8 + i];
            }
        }
    }
```
### gmem

qkv: load

softmax_lse: store new log(sum(exp(x*scale))), load old

o_tmp: store new out, load old

o: store final out

score: store exp((x-m)*scale)，m=max(max(x),log(sum(exp(x))))

```
Gmem_tile_qkv
    uint4 fetch_[LDGS];//存储一个tile的数据
Gmem_summary_stats//softmax_lse
Gmem_tile_o
Gmem_tile_mma_s//score
```
flash-attention/csrc/flash_attn/src/fmha/gmem_tile.h
```
template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile_,
    // The number of bits per element.
    int BITS_PER_ELEMENT,
    // The number of rows of Q, K or V loaded by this tile.
    int ROWS_,
    // The number of columns.
    int COLS,
    int BYTES_PER_LDGS_ = 16
    //一个线程一次读取
>
struct Gmem_tile_qkv {
    static constexpr int BYTES_PER_LDG = BYTES_PER_LDGS_;
    //16;uint4
    // The size of a row in bytes.
    static constexpr int BYTES_PER_ROW = COLS * BITS_PER_ELEMENT / 8;
    //headdim*2
    //64
    // The number of threads to load a "row" of the matrix.
    static constexpr int THREADS_PER_ROW = BYTES_PER_ROW / BYTES_PER_LDG;
    //4
    // The number of "rows" loaded per LDG. 
    //用上所有thread能够加载的行数
    //32
    static constexpr int ROWS_PER_LDG = Cta_tile::THREADS_PER_CTA / THREADS_PER_ROW;
    // The number of LDGs needed to load a chunk of the Q matrix. 需要加载的次数
    static constexpr int LDGS = DivUpConstexpr(ROWS, ROWS_PER_LDG);

    uint4 fetch_[LDGS];
    //regs

    inline __device__ Gmem_tile_qkv(void *ptr_, const uint32_t row_stride_in_elts,
                                    const uint32_t head_stride_in_elts, const int headdim,
                                    const BInfo &binfo, const int tidx, bool use_seqlen_q)
        : row_stride_in_bytes(row_stride_in_elts * BYTES_PER_ELEMENT)
        , actual_seqlen(use_seqlen_q ? binfo.actual_seqlen_q : binfo.actual_seqlen_k)
        , ptr(reinterpret_cast<char *>(ptr_))
        , tidx_(tidx)
        , col_predicate((tidx % THREADS_PER_ROW) * (BYTES_PER_LDG / BYTES_PER_ELEMENT) < headdim) {

        int row = tidx / THREADS_PER_ROW;
        int col = tidx % THREADS_PER_ROW;

        uint32_t row_offset = (uint32_t)(((use_seqlen_q ? binfo.sum_s_q : binfo.sum_s_k) + row) * row_stride_in_bytes);
        row_offset += (uint32_t)(binfo.bidh * head_stride_in_elts * BYTES_PER_ELEMENT);
        ptr += row_offset + col * BYTES_PER_LDG;
        //thread load/store 的起点
    }
    inline __device__ void move(const int steps = 1) {
        //step单位是tile
        ptr += (uint32_t)ROWS * row_stride_in_bytes * steps;
        //跳过不需要处理、已经处理过的token
        actual_seqlen -= ROWS * steps;
    }
    inline __device__ void load() {
        int row_ = tidx_ / THREADS_PER_ROW;
        const void *ptrs[LDGS];
        uint32_t preds[LDGS];
        #pragma unroll
        for( int ii = 0; ii < LDGS; ++ii ) {
            ptrs[ii] = ptr + (uint32_t)ii * ROWS_PER_LDG * row_stride_in_bytes;
            preds[ii] = col_predicate && ((row_ + ii * ROWS_PER_LDG) < min(ROWS, actual_seqlen));
            fetch_[ii] = make_uint4(0, 0, 0, 0);
        }

        Ldg_functor<uint4, LDGS> fct(fetch_, ptrs);
        #pragma unroll
        for( int ii = 0; ii < LDGS; ++ii ) {
            fct.load(ii, preds[ii]);
        }
    }

    // Store data to shared memory.
    template< typename Smem_tile >
    inline __device__ void commit(Smem_tile &smem_tile) {
        smem_tile.store(fetch_);
    }

template<
    typename Cta_tile,
    int BYTES_PER_ELEMENT = 2
>
struct Gmem_tile_o {
    static constexpr int BYTES_PER_STG = BYTES_PER_ELEMENT * 4;
    //一次加载4个元素
    //o=8;o_tmp=16
    static constexpr int COLS = Cta_tile::N;
    // The size of a row in bytes.
    //headdim
    static constexpr int BYTES_PER_ROW = COLS * BYTES_PER_ELEMENT;
    static constexpr int THREADS_PER_ROW = BYTES_PER_ROW / BYTES_PER_STG;
    static constexpr int ROWS = Cta_tile::M;
    // The number of "rows" stored per iteration of the loop. The output of 1 MMA.
    static constexpr int ROWS_PER_LOOP = ROWS <= 64 ? ROWS : (int)Mma_tile::M_PER_MMA_PER_CTA;
    //16
    // The number of outter loop for the stores.
    static constexpr int LOOPS = ROWS / ROWS_PER_LOOP;
    //1
    static constexpr int ROWS_PER_STG = Cta_tile::THREADS_PER_CTA / THREADS_PER_ROW;
    // The number of STGs needed to store a chunk of the Q matrix.
    static constexpr int STGS_PER_LOOP = DivUpConstexpr(ROWS_PER_LOOP, ROWS_PER_STG);
    //列数据量大，分多次加载
    // The number of STGs needed to store a chunk of the Q matrix in total.
    static constexpr int STGS = STGS_PER_LOOP * LOOPS;
    
    inline __device__ Gmem_tile_o(void *ptr, const uint32_t row_stride_in_elts,
                                  const uint32_t head_stride_in_elts, const int headdim,
                                  const BInfo &binfo, const int tidx)
        : row_stride_in_bytes(row_stride_in_elts * BYTES_PER_ELEMENT)
        , actual_seqlen_q(binfo.actual_seqlen_q)
        , ptr_(reinterpret_cast<char *>(ptr))
        , tidx_(tidx)
        , col_predicate((tidx % THREADS_PER_ROW) * (BYTES_PER_STG / BYTES_PER_ELEMENT) < headdim) {

        // Compute the position in the sequence (within the CTA for the moment).
        int row = tidx / THREADS_PER_ROW;
        // Compute the position of the thread in the row.
        int col = tidx % THREADS_PER_ROW;

        uint32_t row_offset = (uint32_t)((binfo.sum_s_q + row) * row_stride_in_bytes);
        row_offset += (uint32_t)(binfo.bidh * head_stride_in_elts * BYTES_PER_ELEMENT);
        // Assemble the final pointer.
        ptr_ += row_offset + col * BYTES_PER_STG;
    }
    
//score
template< typename Cta_tile, typename Base = Gmem_tile_mma_sd<Cta_tile, sizeof(uint16_t)> >
struct Gmem_tile_mma_s : public Base {
    // The number of mmas in the vertical dimension.
    static constexpr int M = Base::MMAS_M;
    // The number of mmas in the horizontal dimension.
    static constexpr int N = Base::MMAS_N;
    inline __device__ Gmem_tile_mma_s(const Params &params, const Block_info& binfo, const int tidx) 
        : Base(params.s_ptr, params, binfo.bidb, binfo.bidh, tidx) {
    }
    template<typename Mask, typename Fragment>
    inline __device__ void store(const Fragment (&frag)[N][M], const Mask& mask){
        #pragma unroll
        for( int mi = 0; mi < M; mi++ ) {
            #pragma unroll
            for( int ni = 0; ni < N; ni++ ) {
                uint4 dst;
                dst.x = frag[ni][mi].reg(0);
                dst.y = frag[ni][mi].reg(2);
                dst.z = frag[ni][mi].reg(1);
                dst.w = frag[ni][mi].reg(3);
                if( mask.any_valid(mi, ni) ) {
                    Base::store(dst, mi, ni);
                }
            }
        }
    }

template< typename Cta_tile, int BYTES_PER_ELEMENT >
struct Gmem_tile_mma_sd {
    static constexpr int BYTES_PER_STG = BYTES_PER_ELEMENT * 8;
    // The number of MMAs in the M dimension.
    static constexpr int MMAS_M = Mma_tile::MMAS_M;
    // The number of MMAs in the N dimension.
    static constexpr int MMAS_N = Mma_tile::MMAS_N;
    static constexpr int BYTES_PER_ROW = THREADS_PER_CTA * BYTES_PER_STG;
    //所有线程完成一次mma结果的写
    // The distance between elements stored per loop (in bytes).
    static constexpr int LOOP_STRIDE_BYTES = MMAS_M * MMAS_N * BYTES_PER_ROW;
    //mma并行生成一块，一个tile有MMAS_M * MMAS_N块
    inline __device__ Gmem_tile_mma_sd(void *ptr, const Params &params, const int bidb, const int bidh, const int tidx) 
        : ptr_(static_cast<char *>(ptr)) {

        uint32_t bidx = bidb * params.h + bidh;

        // The distance between two blocks (in bytes).
        const uint32_t block_stride_bytes = params.seqlen_q * params.seqlen_k * BYTES_PER_ELEMENT;
        // Set store location for each thread at the beginning of the loop
        ptr_ += bidx * block_stride_bytes + tidx * BYTES_PER_STG;
    }
    inline __device__ void move(const int steps = 1) {
        ptr_ += LOOP_STRIDE_BYTES * steps;
    }
    inline __device__ void store(const Type &data, const int mi, const int ni) {
        // size_t offset = (mi * MMAS_N + ni) * BYTES_PER_ROW;
        uint32_t offset = (mi * MMAS_N + ni) * BYTES_PER_ROW;
        fmha::stg(ptr_ + offset, data);
    }

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile
>
struct Gmem_summary_stats {
    static constexpr int ROWS = Cta_tile::M;
    inline __device__ Gmem_summary_stats(void *ptr, const Params &params, const int tidx)
        : ptr_(reinterpret_cast<char *>(ptr)), tidx_(tidx) {
        uint32_t bidx = bidb * params.h + bidh;

        // Extract the position in the warp.
        int warp = tidx / Cta_tile::THREADS_PER_WARP;
        int lane = tidx % Cta_tile::THREADS_PER_WARP;

        uint32_t block_stride_bytes = params.seqlen_q * BYTES_PER_ELEMENT;

        ptr_row_ = ptr_ + bidx * block_stride_bytes;
        ptr_ += bidx * block_stride_bytes + (lane / 4) * BYTES_PER_ELEMENT;
    }
    inline __device__ void move(const int steps) {
        ptr_ += ROWS * BYTES_PER_ELEMENT * steps;
        ptr_row_ += ROWS * BYTES_PER_ELEMENT * steps;
    }
    inline __device__ void store_row(const uint32_t (&data)[MMAS_M], const int row) {
        #pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi) {
            // TODO: Not sure if it's right for MMAS_M > 1
            fmha::stg(ptr_row_ + mi * BYTES_PER_MMA + row * BYTES_PER_ELEMENT, data[mi]);
        }
    }
```
### smem

q,k,v: 缓冲

o: ldg out_old, sts out_numerator_new, update out_numerator_old, lds add, out_numerator/denominator

softmax_lse: sts lse_old, lds lse_old，用于线程之间交换数据

```
Smem_tile
    const uint32_t smem_;
    int smem_read_offset_;
    int smem_write_offset_;
```

flash-attention/csrc/flash_attn/src/fmha/smem_tile.h
```
#include <fmha/gemm.h>
template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    //Gmem_tile_q::BYTES_PER_LDG//16
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE
    //2
>
struct Smem_tile_a<Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE>
    : public Smem_tile_row_a<Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE> {
    // The base class.
    using Base = Smem_tile_row_a<Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

//load
template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    //16
    // The number of buffers per tile.
    int BUFFERS_PER_TILE,
    //2
    // How many rows to use for the XOR pattern to avoid bank conflicts?
    int ROWS_PER_XOR_PATTERN_ = Rows_per_xor_pattern_row_a<Cta_tile::K>::VALUE
>
struct Smem_tile_row_a : public Smem_tile_without_skews<Cta_tile,
                                                               Cta_tile::M,
                                                               Cta_tile::K,
                                                               fmha::BITS_PER_ELEMENT_A,
                                                               BYTES_PER_STS,
                                                               BUFFERS_PER_TILE,
                                                               0,
                                                               ROWS_PER_XOR_PATTERN_,
                                                               1> {
    using Mma_tile = fmha::Hmma_tile<Cta_tile>;    
    // The fragment.
    using Fragment = Fragment_a<Row>;
    inline __device__ Smem_tile_row_a(void *smem, int tidx) : Base(smem, tidx) {
        // The row and column read by the thread.
        int smem_read_row  = (tidx & 0x0f);
        //[0,15]->[0,15] [16,31]->[0,15]
        constexpr int ROWS_PER_PACKING = Base::BYTES_PER_ROW / Base::BYTES_PER_ROW_BEFORE_PACKING;
        int smem_read_col = ((smem_read_row / ROWS_PER_PACKING) % Base::ROWS_PER_XOR_PATTERN) * Base::COLS_PER_XOR_PATTERN;
        smem_read_col ^= (tidx & 0x10) / 16;
        //[0,15]->0 [16,31]->1

        // The shared memory offset.
        this->smem_read_offset_ = smem_read_row*Base::BYTES_PER_ROW_BEFORE_PACKING + smem_read_col*BYTES_PER_LDS;
    }
    inline __device__ void load(Fragment (&a)[Mma_tile::MMAS_M], int ki) {
        #pragma unroll
        for( int mi = 0; mi < Mma_tile::MMAS_M; ++mi ) {
            // Jump by as many matrix rows as needed (a row in smem may pack multiple matrix rows).
            int offset = mi * Mma_tile::M_PER_MMA_PER_CTA * Base::BYTES_PER_ROW_BEFORE_PACKING;

            // Load using LDSM.M88.4.
            uint4 tmp;
            // ldsm(tmp, this->smem_ + this->smem_read_offset_ + this->smem_read_buffer_ + offset);
            ldsm(tmp, this->smem_ + this->smem_read_offset_ + offset);

            // Store the value into the fragment.
            a[mi].reg(0) = tmp.x;
            a[mi].reg(1) = tmp.y;
            a[mi].reg(2) = tmp.z;
            a[mi].reg(3) = tmp.w;
        }

        // Move the offset to the next possition. See doc/mma_smem_layout.xlsx.
        static_assert(Mma_tile_with_padding::MMAS_K < 64, "Not implemented");
        if(        Mma_tile_with_padding::MMAS_K >= 32 && ki % 16 == 15 ) {
            this->smem_read_offset_ ^= 31 * BYTES_PER_LDS * 2;
        } else if( Mma_tile_with_padding::MMAS_K >= 16 && ki %  8 ==  7 ) {
            this->smem_read_offset_ ^= 15 * BYTES_PER_LDS * 2;
        } else if( Mma_tile_with_padding::MMAS_K >=  8 && ki %  4 ==  3 ) {
            this->smem_read_offset_ ^=  7 * BYTES_PER_LDS * 2;
        } else if( Mma_tile_with_padding::MMAS_K >=  4 && ki %  2 ==  1 ) {
            this->smem_read_offset_ ^=  3 * BYTES_PER_LDS * 2;
        } else if( Mma_tile_with_padding::MMAS_K >=  2 ) {
            this->smem_read_offset_ ^=  1 * BYTES_PER_LDS * 2;
        }
    }


template< int N >
struct Rows_per_xor_pattern_row_a : public Rows_per_xor_pattern_a<N> {
};

template< int N >
struct Rows_per_xor_pattern_a {
    // The size in bits.
    enum { N_IN_BITS = N * fmha::BITS_PER_ELEMENT_A };
    // The number of rows.
    enum { VALUE = N_IN_BITS <= 256 ? 2 : (N_IN_BITS <= 512 ? 4 : 8) };
    //行优先存储，一个warp访问一列，可能访问到同一个bank，行宽越大，越容易访问到
    //假设nwarp=2,nbank=4
    //bank0,1
    //2,3
    //0,1
    //2,3
    //0,1,2,3
    //0,1,2,3
    //需要将每一行的col地址进行重新分配:原写到bank0的数据写到bank3，读也从bank3，多了一层地址映射
    //0,1,2,3
    //1,2,3,0
    //smem_write_col  ^ smem_write_xor;
    //a^b==c^b:a==c，两个不同地址不会映射到同一个
    //a和b的位数相同，高位恒为0，集合的范围不变
    //不同行采用不同xor，不可能所有行都采用不同xor，分组，每value行内的xor不同，行宽越大，分组要越大
};

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The layout of the tile.
    typename Layout,
    // The size of the STS.
    int BYTES_PER_STS = 16,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE = 1,
    // Use or not predicates
    bool USE_PREDICATES = true
>
struct Smem_tile_b {
};

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    //16
    // The number of buffers per tile.
    int BUFFERS_PER_TILE
    //1
>
struct Smem_tile_b< Cta_tile, Col, BYTES_PER_STS, BUFFERS_PER_TILE >
    : public Smem_tile_col_b<Cta_tile,
                             BYTES_PER_STS,
                             BUFFERS_PER_TILE> {

    // The base class.
    using Base = Smem_tile_col_b< Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    //16
    // The number of buffers per tile.
    int BUFFERS_PER_TILE,
    //1
    // How many rows to use for the XOR pattern to avoid bank conflicts?
    int ROWS_PER_XOR_PATTERN_ = Rows_per_xor_pattern_col_b<Cta_tile::K>::VALUE
>
struct Smem_tile_col_b : public Smem_tile_without_skews<Cta_tile,
                                                           Cta_tile::N,
                                                           Cta_tile::K,
    using Fragment = Fragment_b< Col>;
    inline __device__ Smem_tile_col_b(void *smem, int tidx) : Base(smem, tidx) {
        static_assert(WARPS_M == 1);
        static_assert(WARPS_N == 4 || WARPS_N == 8);
        static_assert(WARPS_K == 1);
        //1,4,1

        // The masks to select the warps.
        const int WARP_MASK_N = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::N;
        //struct Warp_masks<2, 4, 1> { enum { M = 0x20, N = 0xc0, K = 0x00 }; };
        //8个warp，需要3个bit，M用低1位，N用高2位，左移5位
        //tidx & WARP_MASK_N==warp_id in N
        //warp的分配从M开始

        // The divisor for the warps.
        const int WARP_DIV_N = WARPS_M *       1 * Cta_tile::THREADS_PER_WARP;
        //步幅

        // The row and column read by the thread.
        int smem_read_row  = (tidx & WARP_MASK_N) / WARP_DIV_N * Mma_tile::N_PER_MMA +
        //warp id*element per warp(16)
                             (tidx & 0x07) +
                             (tidx & 0x10) / 2;
        //32->16
        //[0,7]->[0,7] [8,15]->[0,7] [16-23]->[8,15] [24,31]->[8,15]

        constexpr int ROWS_PER_PACKING = Base::BYTES_PER_ROW / Base::BYTES_PER_ROW_BEFORE_PACKING;
        //reshape
        int smem_read_col = ((smem_read_row / ROWS_PER_PACKING) % Base::ROWS_PER_XOR_PATTERN) * Base::COLS_PER_XOR_PATTERN;
        //xor pattern
        smem_read_col ^= (tidx & 0x08) / 8;
        //[0,7]->0 [16,23]->0 [8,15]->1 [24,31]->1
        // The shared memory offset.
        this->smem_read_offset_ = smem_read_row*Base::BYTES_PER_ROW_BEFORE_PACKING + smem_read_col*BYTES_PER_LDS;
    }

template< int N >
struct Rows_per_xor_pattern_col_b : public Rows_per_xor_pattern_b<N> {
};

template< int N >
struct Rows_per_xor_pattern_b {
    // The size in bits.
    enum { N_IN_BITS = N * fmha::BITS_PER_ELEMENT_B };
    // The number of rows.
    enum { VALUE = N_IN_BITS <= 256 ? 2 : (N_IN_BITS <= 512 ? 4 : 8) };
};

template<typename Cta_tile>
struct Smem_tile_v : public fmha::Smem_tile_without_skews<Cta_tile, Cta_tile::K, Cta_tile::N, 16, 16, 1, 0, Rows_per_xor_pattern_col_b<Cta_tile::N>::VALUE, 1> {
    using Fragment = Fragment_b< fmha::Col>;
    inline __device__ Smem_tile_v(void *smem, int tidx) : Base(smem, tidx) {

        // The row/col read by the thread.
        int read_row, read_col;

        read_row = (tidx & 0xe0) / 2 + (tidx & 0x0f);
        //warpi->16i
        //[0,15]->[0,15] [16-31]->[0,15]
        constexpr int ROWS_PER_PACKING = Base::BYTES_PER_ROW / Base::BYTES_PER_ROW_BEFORE_PACKING;
        read_col = ((read_row / ROWS_PER_PACKING) % Base::ROWS_PER_XOR_PATTERN) * Base::COLS_PER_XOR_PATTERN;
        read_col ^= (tidx & 0x10) / 16;
        //[0,15]->0 [16,32]->1

        // The shared memory offset.
        this->smem_read_offset_ = read_row * Base::BYTES_PER_ROW_BEFORE_PACKING + read_col * BYTES_PER_LDS;
    }
    inline __device__ void load(Fragment (&b)[Mma_tile::MMAS_N], int ki) {
#pragma unroll
        for( int ni = 0; ni < Mma_tile::MMAS_N; ++ni ) {
            // Jump by 16 * #warps row.
            int row = ki * 16 * Cta_tile::WARPS_K;

            // Load the data using LDSM.MT88.2.
            uint4 tmp;
            fmha::ldsmt(tmp, this->smem_ + this->smem_read_offset_ + row * Base::BYTES_PER_ROW_BEFORE_PACKING);
            b[ni].reg(0) = tmp.x;
            b[ni].reg(1) = tmp.y;
            b[ni].reg(2) = tmp.z;
            b[ni].reg(3) = tmp.w;

            // Move the pointer for the next ni. I expect the compiler to not recompute those.
            if( Mma_tile::MMAS_N == 1 ) {
                // noop
            } else if( Mma_tile::MMAS_N == 2 ) {
                this->smem_read_offset_ ^= BYTES_PER_LDS * 2;
            } else if( Mma_tile::MMAS_N == 4 ) {
                this->smem_read_offset_ ^= BYTES_PER_LDS * (ni % 2 == 0 ? 2 : 6);
            } else if (Mma_tile::MMAS_N == 8) {
                this->smem_read_offset_ ^= BYTES_PER_LDS * (ni % 4 == 3 ? 14 : (ni % 2 == 1 ? 6 : 2));
            } else {
                assert(false);  // Not implemented!
            }
        }
    }

//store
template<
    // The description of the tile computed by this CTA.
    typename Cta_tile,
    // The number of rows in the 2D shared memory buffer.
    int M_,
    // The number of cols.
    int N_,
    //headdim>=32
    // The size in bits of each element.
    int BITS_PER_ELEMENT_,
    // The number of bytes per STS.
    int BYTES_PER_STS_ = 16,
    //16
    // The number of buffers. (Used in multistage and double buffer cases.)
    int BUFFERS_PER_TILE_ = 1,
    //q2;kv1
    // Do we enable the fast path for LDS.128 and friends.
    int ENABLE_LDS_FAST_PATH_ = 0,
    // The number of rows that are used for the XOR swizzling to allow fast STS/LDS.
    //0
    int ROWS_PER_XOR_PATTERN_ = 8,
    // The number of cols that are used for the XOR swizzling to allow fast STS/LDS.
    int COLS_PER_XOR_PATTERN_ = 1,
    // Use or not predicates
    bool USE_PREDICATES_ = true
>
struct Smem_tile_without_skews {
    // To support arbitrary N, we pad some values to a power-of-2.
    enum { N_WITH_PADDING = Next_power_of_two<N_>::VALUE };
    // The number of bytes per row without packing of rows.
    enum { BYTES_PER_ROW_BEFORE_PACKING = N_WITH_PADDING * BITS_PER_ELEMENT / 8 };

    // The number of bytes per row -- we want at least 128B per row.
    enum { BYTES_PER_ROW = Max<BYTES_PER_ROW_BEFORE_PACKING, 128>::VALUE };
    //headdim==32需要两行
    // The number of rows in shared memory (two rows may be packed into a single one).
    enum { ROWS = M_ * BYTES_PER_ROW_BEFORE_PACKING / BYTES_PER_ROW };
    //reshape，把原来多行看成一行

    // The number of threads per row.
    enum { THREADS_PER_ROW_UNBOUNDED = BYTES_PER_ROW / BYTES_PER_STS };
    //128/16=8
    // The number of threads per row.
    enum { THREADS_PER_ROW = Min<Cta_tile::THREADS_PER_CTA, THREADS_PER_ROW_UNBOUNDED>::VALUE };
    //8
    // The number of STS per row.
    enum { STS_PER_ROW = BYTES_PER_ROW / THREADS_PER_ROW / BYTES_PER_STS };
    //1
    // The number of rows written with a single STS.
    enum { ROWS_PER_STS = Cta_tile::THREADS_PER_CTA / THREADS_PER_ROW };
    //16
    // The number of STS needed to store all rows.
    enum { STS_PER_COL = Div_up<ROWS, ROWS_PER_STS>::VALUE };
    // The number of STS in total.
    enum { STS = STS_PER_COL * STS_PER_ROW };
    //一次写多行

    static constexpr bool PARTIAL_STORE = ROWS_PER_STS > ROWS;
    static constexpr int STORING_THREADS = PARTIAL_STORE ? ROWS * THREADS_PER_ROW : Cta_tile::THREADS_PER_CTA;

    enum { BYTES_PER_BUFFER = STS * BYTES_PER_STS * STORING_THREADS };
    // The number of buffers.
    enum { BUFFERS_PER_TILE = BUFFERS_PER_TILE_ };
    // The size in bytes of total buffers.
    enum { BYTES_PER_TILE = BYTES_PER_BUFFER * BUFFERS_PER_TILE };
    enum { BYTES_PER_TILE_INC_BOUNDARY = BYTES_PER_TILE - BYTES_PER_BUFFER };

    // The shared memory pointer.
    const uint32_t smem_;
    // The read offset. Reserve 4 offsets if needed.
    int smem_read_offset_;
    // The write offset.
    int smem_write_offset_;
    // The buffer base offset for read.
    // int smem_read_buffer_;
    // The buffer base offset for write.
    // int smem_write_buffer_;
    const int tidx_;

    inline __device__ Smem_tile_without_skews(void *smem, int tidx)
        : smem_(__nvvm_get_smem_pointer(smem)), tidx_(tidx) {
        //把 C++ 里的 shared memory 指针转换成 PTX shared-memory address

        // The row written by a thread. See doc/mma_smem_layout.xlsx.
        int smem_write_row = tidx / THREADS_PER_ROW;

        // The XOR pattern.
        int smem_write_xor = smem_write_row % ROWS_PER_XOR_PATTERN * COLS_PER_XOR_PATTERN;
        //同一行的thread的pattern相同
        //col_xor=2^k，把pattern左移k位，连续2^k列的地址的高于k的位相同，pattern相同，变换相同，以k列为一组进行变换
        // Compute the column and apply the XOR pattern.
        int smem_write_col = (tidx % THREADS_PER_ROW) ^ smem_write_xor;
        // The offset.
        this->smem_write_offset_ = smem_write_row*BYTES_PER_ROW + smem_write_col*BYTES_PER_STS;

    template< int N >
    inline __device__ void store(const Store_type (&data)[N], uint64_t = 0) {
        uint32_t smem_ptrs[N];
        this->compute_store_pointers(smem_ptrs);//store地址
        // Trying to reduce the shared mem for Q from 4KB per buffer to 2KB per buffer.
        if (!PARTIAL_STORE || (tidx_ / THREADS_PER_ROW < ROWS)) {
            sts(smem_ptrs, data);
        }
    }

    inline __device__ void compute_store_pointers(uint32_t (&ptrs)[N]) {
        #pragma unroll
        for( int ii = 0; ii < N; ++ii ) {
            //N 次 ldg
            // Decompose the STS into row/col.
            int row = ii / STS_PER_ROW;
            int col = ii % STS_PER_ROW;

            // Assemble the offset.
            int offset = smem_write_offset_ + row*ROWS_PER_STS*BYTES_PER_ROW;

            // Take the column into account.
            if( STS_PER_ROW > 1 ) {
                offset += col*THREADS_PER_ROW*BYTES_PER_STS;
            }

            // Apply the XOR pattern if needed.
            if( ROWS_PER_STS < ROWS_PER_XOR_PATTERN ) {
                const int m = row * ROWS_PER_STS % ROWS_PER_XOR_PATTERN;
                offset ^= m * COLS_PER_XOR_PATTERN * BYTES_PER_STS;
            }

            ptrs[ii] = smem_ + offset;
        }
    }

    inline __device__ void move_to_next_read_buffer() {
        if( BUFFERS_PER_TILE > 1 && smem_read_offset_ >= BYTES_PER_TILE_INC_BOUNDARY ) {
            this->smem_read_offset_ -= BYTES_PER_TILE_INC_BOUNDARY;
        } else if( BUFFERS_PER_TILE > 1 ) {
            this->smem_read_offset_ += BYTES_PER_BUFFER;
        }
    }

template <typename Cta_tile>
struct Smem_tile_o {
    static constexpr int THREADS_PER_ROW = Cta_tile::N * BYTES_PER_ELEMENT / BYTES_PER_LDS;
    uint32_t smem_write_, smem_read_;
    inline __device__ Smem_tile_o(void *smem, int tidx) {
        int write_row = (tidx & 0x1c) / 4;
        //4个lane一组，[0-7]
        const int lane = tidx % 32;
        const int warp = tidx / 32;

        constexpr int ELEMENTS_PER_STS = BYTES_PER_STS / BYTES_PER_ELEMENT;
        constexpr int STS_PER_WARP = 16 * Mma_tile::MMAS_N / ELEMENTS_PER_STS;
        //N_PER_MMA = 16,The number of MMAs per warp in N 
        int write_col = warp * STS_PER_WARP + lane % STS_PER_WARP;

        // Assemble the write pointer.
        smem_write_ = smem_ + write_row * BYTES_PER_ROW + write_col * BYTES_PER_STS;

        // The element read by each thread.
        int read_row = tidx / THREADS_PER_ROW;
        int read_col = tidx % THREADS_PER_ROW;

        // Take the XOR pattern into account for the column.
        read_col ^= 2 * (read_row % (Cta_tile::N == 16 ? 2 : (Cta_tile::N == 32 ? 4 : 8)));

        this->smem_read_ = smem_ + read_row * BYTES_PER_ROW + read_col * BYTES_PER_LDS;
    }
    template <int M, int N>
    inline __device__ void store(const Accumulator (&acc)[M][N], int mi) {
    inline __device__ void load(uint4 (&out)[LDS_PER_LOOP]) const {
        for( int ii = 0; ii < LDS_PER_LOOP; ++ii ) {
            for( int jj = 0; jj < Cta_tile::WARPS_K; ++jj ) {
            out[ii] = zero_init ? tmp[0] : fmha::fadd4(out[ii], tmp[0]);
            for( int jj = 1; jj < Cta_tile::WARPS_K; ++jj ) {
                out[ii] = fmha::fadd4(out[ii], tmp[jj]);

//softmax_lse
template<
    typename Gmem_tile,
    // The number of buffers. (Used in multistage and double buffer cases.)
    int BUFFERS_PER_TILE_ = 1
>
struct Smem_tile_dp_sum {

    inline __device__ Smem_tile_dp_sum(float *smem, const int tidx)
        : smem_(smem), smem_read_buffer_(smem), smem_write_buffer_(smem), tidx_(tidx) {
    }


```
### Gemm_Q_K
```
#q[0:Mma_tile_p::MMAS_M,0].lds
#kv一次性加载到寄存器
#向量内积
for( int ki = 1; ki < Mma_tile_p::MMAS_K; ++ki ) {
    q[0:Mma_tile_p::MMAS_M,ki].lds
    #q两份小smem
    gemm_cl(acc_p, q[ki - 1], k[ki - 1]);
        #按tensor core k大小切分，向量内积
        for (int iter = 0; iter < kIters; iter++) {
            #按tensor core m,n大小切分，向量外积
            for mi in M:
                for ni in N:
                    acc[mi][ni] += a[mi] @ b[ni]
                    #tensor core
}
{
    int ki = Mma_tile_p::MMAS_K;
    gemm_cl(acc_p, q[ki - 1], k[ki - 1]);
}
```
```
Gemm_Q_K
    Fragment_k frag_k[Mma_tile_p::MMAS_K][Mma_tile_p::MMAS_N];//计算次数
    Fragment_q frag_q[2][Mma_tile_p::MMAS_M];//计算次数
    Smem_tile_q smem_q;
    Smem_tile_k smem_k;
```
flash-attention/csrc/flash_attn/src/fmha_fprop_kernel_1xN.h
```
template<typename Kernel_traits, bool K_in_regs, typename elem_type_=__half>
struct Gemm_Q_K : public Gemm_Q_K_base<Kernel_traits> {
    static constexpr int SMEM_OFFSET_O = Smem_tile_q::BYTES_PER_TILE;
    static constexpr int SMEM_OFFSET_SOFTMAX = SMEM_OFFSET_O + Smem_tile_o::BYTES_PER_TILE;
    static constexpr int SMEM_OFFSET_V = Smem_tile_q::BYTES_PER_TILE + (SHARE_SMEM_FOR_K_AND_V ? 0 : Smem_tile_k::BYTES_PER_TILE);
    // Q | K / V
    //   | O | SOFTMAX

    Fragment_k frag_k[Mma_tile_p::MMAS_K][Mma_tile_p::MMAS_N];//计算次数

    __device__ inline Gemm_Q_K(char * smem_, const int tidx) 
        : Base(smem_, smem_ + Smem_tile_q::BYTES_PER_TILE, tidx) {
    }

    __device__ inline void load_k(){
        #pragma unroll
        for( int ki = 0; ki < Mma_tile_p::MMAS_K; ++ki ) {
            Base::smem_k.load(frag_k[ki], ki);
        }
    }

    template<typename Acc, int M, int N>
    __device__ inline void operator()(Acc (&acc_p)[M][N]){
        // Do this part of P^T = (Q * K^T)^T.
        #pragma unroll
        for( int ki = 1; ki < Mma_tile_p::MMAS_K; ++ki ) {
            // Trigger the load from shared memory for the next series of Q values.
            Base::smem_q.load(Base::frag_q[ki & 1], ki);
            // Do the math for the values already in registers.
            fmha::gemm_cl<elem_type>(acc_p, Base::frag_q[(ki - 1) & 1], frag_k[(ki - 1)]);
        }
        // Do the final stage of math.
        {
            int ki = Mma_tile_p::MMAS_K;
            fmha::gemm_cl<elem_type>(acc_p, Base::frag_q[(ki - 1) & 1], frag_k[(ki - 1)]);
        }
    }
    
    __device__ inline void reload_k(){
        // Noop.
    }

template<typename Kernel_traits>
struct Gemm_Q_K_base {
    using Fragment_q = typename Smem_tile_q::Fragment;
    __device__ inline Gemm_Q_K_base(char * smem_ptr_q, char * smem_ptr_k, const int tidx) 
        : smem_q(smem_ptr_q, tidx)
        , smem_k(smem_ptr_k, tidx) {

    }
    Fragment_q frag_q[2][Mma_tile_p::MMAS_M];//计算次数
    Smem_tile_q smem_q;
    Smem_tile_k smem_k;
    __device__ inline void load_q() {
        smem_q.load(frag_q[0], 0);
    }
    __device__ inline void reload_q() {
        smem_q.load(frag_q[0], 0);
    }
```
### softmax
```
Softmax
    float elt_[MMAS_M * 2][MMAS_N * 4];
    Smem_tile_red smem_max_;
    //利用 shared memory 完成 跨 warp 通信和数据重排。
    Smem_tile_red smem_sum_; 
    //sum(exp((x-m)*scale))
```
flash-attention/csrc/flash_attn/src/fmha/softmax.h
```
template<typename Cta_tile, typename Kernel_traits>
struct Softmax : public Softmax_base<Cta_tile, Kernel_traits> {
    using Base = Softmax_base<Cta_tile, Kernel_traits>;
    using Smem_tile_red = Smem_tile_reduce<Cta_tile, Kernel_traits>;

    const uint32_t params_scale_bmm1_;
    Smem_tile_red smem_max_;
    Smem_tile_red smem_sum_;
    inline __device__ Softmax(const Params &params, void *smem, int tidx)
        : Base(params, smem, tidx)
        , params_scale_bmm1_(params.scale_bmm1) 
        , smem_sum_(static_cast<float*>(smem), tidx)
        , smem_max_(static_cast<float*>(smem) + Smem_tile_red::ELTS_PER_TILE, tidx) {
    }
    inline __device__ void unpack_noscale(const Accumulator (&acc)[MMAS_M][MMAS_N]) {
        #pragma unroll
        for( int mi = 0; mi < MMAS_M; ++mi ) {
            #pragma unroll
            for( int ni = 0; ni < MMAS_N; ++ni ) {
                // 1st row - 4 elements per row.
                this->elt_[2 * mi + 0][4 * ni + 0] = acc[mi][ni].elt(0);
    template<bool zero_init=true>
    __device__ inline void reduce_max(float (&frag)[2 * MMAS_M]){ 
        MaxOp<float> max;
        reduce_<zero_init>(frag, max, smem_max_);
    }
    template<bool zero_init=true>
    __device__ inline void reduce_sum_before_sync_(float (&frag)[2 * MMAS_M]){
        SumOp<float> sum;
        thread_reduce_<zero_init>(frag, sum);
        quad_reduce(frag, frag, sum);
        smem_sum_.store(frag);
    }
    template<bool zero_init=true, typename Operator>
    __device__ inline void reduce_(float (&frag)[2 * MMAS_M], Operator &op, Smem_tile_red & smem_red) {
        thread_reduce_<zero_init>(frag, op);
        quad_reduce(frag, frag, op);
        //连续 4 个 lane 负责同一行的不同列，因此把这 4 个 lane 的结果合并。
        smem_red.store(frag);
        //利用 shared memory 完成跨 warp 通信和数据重排。
        __syncthreads();
        typename Smem_tile_red::read_t tmp[2 * MMAS_M];
        smem_red.load(tmp);
        quad_allreduce(frag, tmp, op);
    }
    template<bool zero_init=true, typename Operator>
    __device__ inline void thread_reduce_(float (&frag)[2 * MMAS_M], Operator &op) {
        #pragma unroll
        for( int mi = 0; mi < 2 * MMAS_M; mi++ ) {
            frag[mi] = zero_init ? this->elt_[mi][0] : op(frag[mi], this->elt_[mi][0]);
            #pragma unroll
            for( int ni = 1; ni < 4 * MMAS_N; ni++ ) {
                frag[mi] = op(frag[mi], this->elt_[mi][ni]);
            }
        }
    }
    template<typename elem_type=__half, int K, int M>
    inline __device__ void pack(Fragment_a (&dst)[K][M]) const {
    template<int NROWS>
    __device__ inline void reduce_max_after_sync_(float (&frag)[NROWS][MMAS_M],
                                                  const int (&rows)[NROWS]){
        MaxOp<float> max;
        reduce_after_sync_(frag, rows, max, smem_max_);
    }
    template<int NROWS, typename Operator>
    __device__ inline void reduce_after_sync_(float (&frag)[NROWS][MMAS_M],
                                              const int (&rows)[NROWS],
                                              Operator &op, Smem_tile_red & smem_red) {
        #pragma unroll
        for (int ii = 0; ii < NROWS; ii++) {
            typename Smem_tile_red::read_t tmp[MMAS_M];
            smem_red.load_row(tmp, rows[ii]);
            quad_allreduce(frag[ii], tmp, op);
        }
    }

template<typename Cta_tile, typename Kernel_traits>
struct Softmax_base {
    static constexpr int GROUPS = fmha::DivUpConstexpr(Cta_tile::WARPS_N, 4);
    //4 warp组成一组
    // The number of elements that we are going to store per row.
    static constexpr int ELEMENTS_PER_ROW = Cta_tile::WARPS_N / GROUPS;
    //一个warp负责一个元素
    static constexpr int ROWS = Cta_tile::M * GROUPS;
    //4个元素一行
    // The total number of elements.
    static constexpr int ELEMENTS = ROWS * ELEMENTS_PER_ROW;

    // The elements.
    float elt_[MMAS_M * 2][MMAS_N * 4];
    
    inline __device__ Softmax_base(const Params &params, void *smem, int tidx)
        :  // packed_mask_ptr_(reinterpret_cast<const char*>(params.packed_mask_ptr)),
          smem_(reinterpret_cast<float *>(smem)), tidx_(tidx) {
        int warp = tidx / Cta_tile::THREADS_PER_WARP;
        int lane = tidx % Cta_tile::THREADS_PER_WARP;

        // Decompose the warp index into M and N.
        int warp_m = warp % Cta_tile::WARPS_M;
        int warp_n = warp / Cta_tile::WARPS_M;

        // Decompose the warp-n index into group/position-inside-the-group.
        int warp_g = warp_n / ELEMENTS_PER_ROW;
        int warp_i = warp_n % ELEMENTS_PER_ROW;

        // The location written by the threads.
        int write_row = warp_g * (ROWS / GROUPS) + warp_m * Mma_tile::M_PER_MMA + lane / 4;
        //切列，拼行
        int write_col = warp_i;

        // Assemble the write pointer.
        smem_write_ = &smem_[write_row * ELEMENTS_PER_ROW + write_col];

        // Assemble the read pointer.
        smem_read_ = &smem_[warp_m * Mma_tile::M_PER_MMA + lane / 4];

    template <bool scale_max=true>
    inline __device__ void scale_apply_exp(const float (&max)[MMAS_M * 2], const float scale_) {
        //e^x == 2^(x * log2(e))
        const float max_scale = scale_max ? scale_ * M_LOG2E : M_LOG2E;
        const float scale = scale_ * M_LOG2E;
        #pragma unroll
        for( int mi = 0; mi < MMAS_M * 2; ++mi ) {
            // Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
            // max * log_2(e)) This allows the compiler to use the ffma
            // instruction instead of fadd and fmul separately.
            const float max_scaled = max[mi] * max_scale;
            #pragma unroll
            for( int ni = 0; ni < MMAS_N * 4; ++ni ) {
                elt_[mi][ni] = apply_exp2_(elt_[mi][ni] * scale, max_scaled);
            }
        }
    }

inline __device__ float apply_exp2_(float x, float max) {
    return exp2f(x - max);

template <typename Cta_tile, typename Kernel_traits>
struct Smem_tile_reduce {
    static constexpr int ROWS = WARPS_M * MMAS_M * 16;
    static constexpr int COLS = WARPS_N;
    static_assert(COLS == 4 || COLS == 8);
    static constexpr int ROWS_PER_XOR_PATTERN = (COLS == 8) ? 4 : 8;
    static constexpr int BYTES_PER_TILE = ROWS * COLS * sizeof(float);
    static constexpr int ELTS_PER_TILE = ROWS * COLS;

    using read_t = typename ReadType<COLS>::T;

    __device__ inline Smem_tile_reduce(float *smem_, const int tidx) {

        int lane = tidx % 32;
        int warp = tidx / 32;

        int warp_m = warp % WARPS_M;
        int warp_n = warp / WARPS_M;

        qid_ = lane % 4;
        int qp = lane / 4;

        // Swizzle the column to avoid 2-fold bank conflicts when we have 8 warps.
        // This won't affect reading as we assume commutative reduction ops.
        const int col = warp_n ^ (qp / ROWS_PER_XOR_PATTERN);
        smem_write_ = &smem_[warp_m * 16 * MMAS_M * WARPS_N + qp * WARPS_N + col];
        smem_read_ = &reinterpret_cast<read_t *>(smem_)[warp_m * 16 * MMAS_M * 4 + qp * 4 + qid_];
        smem_read_row_ = &reinterpret_cast<read_t *>(smem_)[warp_m * 16 * MMAS_M * 4 + qid_];
    }
    __device__ inline void store(float (&frag)[2 * MMAS_M]) {
        if( qid_ == 0 ) {
            #pragma unroll
            for( int mi = 0; mi < MMAS_M; mi++ ) {
                int offset = mi * 16 * WARPS_N;
                smem_write_[offset + 0 * 8 * WARPS_N] = frag[mi * 2 + 0];
                smem_write_[offset + 1 * 8 * WARPS_N] = frag[mi * 2 + 1];
            }
        }
    }
    __device__ inline void load(read_t (&frag)[2 * MMAS_M]) {
        #pragma unroll
        for( int mi = 0; mi < MMAS_M; mi++ ) {
            int offset = mi * 16 * 4;
            frag[mi * 2 + 0] = smem_read_[offset + 0 * 8 * 4];
            frag[mi * 2 + 1] = smem_read_[offset + 1 * 8 * 4];
        }
    }
    __device__ inline void load_row(read_t (&frag)[MMAS_M], int row) {
        #pragma unroll
        for( int mi = 0; mi < MMAS_M; mi++ ) {
            int offset = mi * 16 * 4;
            frag[mi] = smem_read_row_[offset + 0 * 8 * 4 + row * 4];
        }
    }
```
### mask
flash-attention/csrc/flash_attn/src/fmha/mask.h
```
template<typename Cta_tile, bool Is_causal=false>
struct Mask {
    using Mma_tile = fmha::Hmma_tile<Cta_tile>;

    template<typename BInfo>
    __device__ Mask(const BInfo &binfo, int tidx, const int loop_step_idx_ = 0)
        : actual_seqlen_k(binfo.actual_seqlen_k - loop_step_idx_ * Cta_tile::N)
        , loop_step_idx(loop_step_idx_) {

        const int warp = tidx / Cta_tile::THREADS_PER_WARP;
        const int lane = tidx % Cta_tile::THREADS_PER_WARP;

        const int warp_n = (warp / Cta_tile::WARPS_M);
        const int warp_m = (warp % Cta_tile::WARPS_M);
        //一个warp负责二维矩阵的一块

        const int quad = lane / 4;
        //warp 分成8组，每组1行
        const int tid = (lane % 4) * 2;
        //线程组内每个线程负载两个元素，一个线程组负责8个元素
        row = warp_m * 16 + quad;
        col = warp_n * 16 + tid;
        //元素为单位
    }
    inline __device__ void load(const int it) {
        row_offset = it * Cta_tile::M + row;
    }
    inline __device__ bool is_valid(const int mi, const int ni, const int ii, const int jj) const {
```
### utils
flash-attention/csrc/flash_attn/src/fmha/utils.h
```
template< typename Data_type, int N >
struct Ldg_functor {
    // Ctor.
    inline __device__ Ldg_functor(Data_type (&fetch)[N], const void* (&ptrs)[N])
        : fetch_(fetch), ptrs_(ptrs) {
    }

    // Clear the element.
    inline __device__ void clear(int ii) {
        fmha::clear(fetch_[ii]);
    }

    // Trigger the loads.
    inline __device__ void load(int ii, bool p) {
        if( p ) {
            ldg(fetch_[ii], ptrs_[ii]);
        }
    }

    // The fetch registers.
    Data_type (&fetch_)[N];
    // The pointers.
    const void* (&ptrs_)[N];
};

inline __device__ void ldg(uint4 &dst, const void *ptr) {
    dst = *reinterpret_cast<const uint4*>(ptr);
}

template< int N >
inline __device__ void sts(uint32_t (&ptrs)[N], const uint4 (&data)[N]) {
    sts_<uint4, N>(ptrs, data);
}

template< typename Data_type, int N >
inline __device__ void sts_(uint32_t (&ptrs)[N], const Data_type (&data)[N]) {
    #pragma unroll
    for( int ii = 0; ii < N; ++ii ) {
        sts(ptrs[ii], data[ii]);
    }
}

inline __device__ void sts(uint32_t ptr, uint4 val) {
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};\n"
        :
        : "r"(ptr)
        , "r"(val.x)
        , "r"(val.y)
        , "r"(val.z)
        , "r"(val.w));
    //向 shared memory 写 4 个 32-bit word
}

inline __device__ void ldsm(uint4 &dst, uint32_t ptr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 730
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(dst.x), "=r"(dst.y), "=r"(dst.z), "=r"(dst.w) : "r"(ptr));
#endif
//从 shared memory 里加载一个矩阵片段到寄存器
}

template<typename Operator, int M>
__device__ inline void  quad_reduce(float (&dst)[M], float (&src)[M], Operator &op) {
    #pragma unroll
    for(int mi=0; mi < M; mi++){
        dst[mi] = src[mi];
        dst[mi] = op(dst[mi], __shfl_down_sync(uint32_t(-1), dst[mi], 2));
        dst[mi] = op(dst[mi], __shfl_down_sync(uint32_t(-1), dst[mi], 1));
    }
}

template<typename Operator, int M>
__device__ inline void quad_allreduce(float (&dst)[M], float (&src)[M], Operator &op) {
    #pragma unroll
    for(int mi=0; mi < M; mi++){
        dst[mi] = src[mi];
        dst[mi] = Allreduce<4>::run(dst[mi], op);
    }
}

template<int THREADS>
struct Allreduce {
    static_assert(THREADS == 32 || THREADS == 16 || THREADS == 8 || THREADS == 4);
    template<typename T, typename Operator>
    static __device__ inline T run(T x, Operator &op) {
        constexpr int OFFSET = THREADS / 2;
        x = op(x, __shfl_xor_sync(uint32_t(-1), x, OFFSET));
        return Allreduce<OFFSET>::run(x, op);
    }
};

inline __device__ void stg(void *ptr, uint4 val) {
    *reinterpret_cast<uint4*>(ptr) = val;
}

static inline __device__ uint4 fadd4(uint4 a, uint4 b) {
    float4 c;
    c.x = reinterpret_cast<const float&>(a.x) + reinterpret_cast<const float&>(b.x);
    c.y = reinterpret_cast<const float&>(a.y) + reinterpret_cast<const float&>(b.y);
    c.z = reinterpret_cast<const float&>(a.z) + reinterpret_cast<const float&>(b.z);
    c.w = reinterpret_cast<const float&>(a.w) + reinterpret_cast<const float&>(b.w);
    return reinterpret_cast<const uint4&>(c);
}
```
### loop
smem usage: 一次加载一个qkv_tile到smem，kv一次性移动到regs，q是在计算QK^T过程中逐步移动到regs
```
Q | K / V       | softmax_lse
  | O | SOFTMAX | softmax_lse
```
伪代码
```
grid=[b, h, q_num_splits]
//q head is split, blockIdx.z does q_tile_id%q_num_splits==blockIdx.z
for loop_step_idx in range(k_tile_num):
    int begin = Is_causal ? loop_step_idx * k_tile_size / q_tile_size : 0;
    const int begin_mod_z = begin % gridDim.z;
    begin = begin_mod_z <= blockIdx.z ? begin - begin_mod_z : begin + gridDim.z - begin_mod_z;
    if no actual_seqlen_k left or no actual_seqlen_q left:
        return 

    qkv.ldg
    if (loop_step_idx!=0) { __syncthreads(); }
    if (loop_step_idx!=0) {
        softmax_lse.ldg(p_prev_lse)
    }
    qv.sts
    __syncthreads
    qv.lds
    __syncthreads
    k.sts
    __syncthreads
    k.lds

    for (int l = blockIdx.z; l < q_tile_num-begin; l += q_num_splits) 
        if no actual_seqlen_q left:
            break

        QK^T

        if (loop_step_idx!=0) { o_tmp.ldg(out) }
        if (l + q_num_splits < q_tile_num-begin) {
            q.ldg
        }

        apply_mask

        if(l < q_num_splits) {
            __syncthreads
        }
        if (loop_step_idx!=0) {
            softmax_lse.sts(p_prev_lse)
            p_max = p_prev_lse / scale
        }
        if (l + q_num_splits < q_tile_num-begin) {
            if (loop_step_idx!=0) 
                softmax_lse.ldg(p_prev_lse)
        }

        reduce_max(p_max)

        scale_apply_exp(p_max, scale)
        
        reduce_sum(p_sum)
        
        s.stg

        if (l + step_stride < q_tile_num-begin) {
            q.sts
        }

        p_numerator V

        reduce_max(p_max_o, rows)
        p_max_o *= scale
        if (loop_step_idx!=0) {
            softmax_lse.lds(p_prev_scale_o, rows)
        }

        o.sts

        __syncthreads
        reduce_sum(p_sum_o, rows)
        if (loop_step_idx!=0) {
            p_prev_scale_o = expf(p_prev_scale_o - p_max_o)
            p_sum_o += p_prev_scale_o
        }

        p_sum_log = p_max_o + __logf(p_sum_o);
        softmax_lse.stg(p_sum_log)
            
        if (loop_step_idx!=0) {
            out = fmul4(out, p_prev_scale_o)
        }
        o.lds(out)

        out = fmul4(out, 1.f / p_sum_o)
        if o is finish:
            o.stg(out)
        else:
            o_tmp.stg(out)

        if (l + q_num_splits < q_tile_num-begin) 
            q.lds
```

flash-attention/csrc/flash_attn/src/flash_fwd_launch_template.h
```
template<typename Kernel_traits>
void run_fmha_fwd_loop(Launch_params<FMHA_fprop_params> &launch_params) {
    constexpr int blocksize_c = Kernel_traits::Cta_tile_p::N;
    const int loop_steps = (launch_params.params.seqlen_k + blocksize_c - 1) / blocksize_c;

    constexpr int smem_size_softmax_lse = Kernel_traits::Smem_dp_sum::BYTES_PER_TILE;
    // Don't need smem_size_softmax_lse if we're not looping
    const int smem_size = fmha::get_dynamic_smem_size<Kernel_traits>()
        + (loop_steps > 1 ? smem_size_softmax_lse : 0);

        auto kernel = launch_params.params.is_causal
            ? (launch_params.return_softmax
               ? &fmha_fwd_loop_kernel<Kernel_traits, IsDropoutConst, true, true>
               : &fmha_fwd_loop_kernel<Kernel_traits, IsDropoutConst, true, false>)
            : (launch_params.return_softmax
               ? &fmha_fwd_loop_kernel<Kernel_traits, IsDropoutConst, false, true>
               : &fmha_fwd_loop_kernel<Kernel_traits, IsDropoutConst, false, false>);

        if( smem_size >= 48 * 1024 ) {
            FMHA_CHECK_CUDA(cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        }
                
        if (launch_params.params.num_splits <= 0) {
            int ctas_per_sm;
            cudaError status_ = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &ctas_per_sm, kernel, Kernel_traits::THREADS, smem_size);
            constexpr int M = Kernel_traits::Cta_tile_p::M;
            //q tile
            launch_params.params.num_splits = num_splits_heuristic_fwd(
                launch_params.params.b * launch_params.params.h, dprops->multiProcessorCount,
                ctas_per_sm,
                /*max_splits=*/std::min(30, (launch_params.params.seqlen_q + M - 1 / M))
                //one q tile <=cta的任务量<=one head
            );
        dim3 grid(launch_params.params.b, launch_params.params.h, launch_params.params.num_splits);
        kernel<<<grid, Kernel_traits::THREADS, smem_size, launch_params.stream>>>(
            launch_params.params);

//ctas的数量决定效率，求解效率最高的数量
inline int num_splits_heuristic_fwd(int batch_nheads, int num_SMs, int ctas_per_sm, int max_splits) {
    float max_efficiency = 0.f;
    std::vector<float> efficiency;
    efficiency.reserve(max_splits);
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        float n_waves = float(batch_nheads * num_splits) / (num_SMs * ctas_per_sm);
        //串行次数
        float eff = n_waves / ceil(n_waves);
        //浪费
        // printf("num_splits = %d, eff = %f\n", num_splits, eff);
        if (eff > max_efficiency) { max_efficiency = eff; }
        efficiency.push_back(eff);
    }
    //倾向于少split
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        if (efficiency[num_splits - 1] > 0.95 * max_efficiency) {
            // printf("num_splits chosen = %d\n", num_splits);
            return num_splits;
        }
    }
    return 1;
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Return_softmax>
__global__ void fmha_fwd_loop_kernel(FMHA_fprop_params params) {
    fmha::device_1xN_loop<Kernel_traits, Is_dropout, Is_causal, Return_softmax>(params);
}
```
flash-attention/csrc/flash_attn/src/fmha_kernel.h
```
template<int THREADS_PER_CTA>
struct BlockInfoPadded {

    template<typename Params>
    __device__ BlockInfoPadded(const Params &params,
                               const int bidb,
                               const int bidh,
                               const int tidx)
        : bidb(bidb), bidh(bidh), h(params.h) {

        // The block index.
        sum_s_k = params.cu_seqlens_k[bidb];
        actual_seqlen_k = params.cu_seqlens_k[bidb + 1] - sum_s_k;
        sum_s_q = params.cu_seqlens_q[bidb];
        actual_seqlen_q = params.cu_seqlens_q[bidb + 1] - sum_s_q;

    __device__ bool stop_early(const int start_col = 0) const {
        return actual_seqlen_k <= start_col;
    }
```
flash-attention/csrc/flash_attn/src/fmha_fprop_kernel_1xN.h
```
//for k,v tile in head
template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Return_softmax, typename Params>
inline __device__ void device_1xN_loop(const Params &params) {
    // The block index for the batch.
    const int bidb = blockIdx.x;
    // The block index for the head.
    const int bidh = blockIdx.y;
    constexpr int M = Kernel_traits::Cta_tile_p::M;
    //q tile size
    const int STEPS = (params.seqlen_q + M - 1) / M;
    //q tile num
    constexpr int blocksize_c = Kernel_traits::Cta_tile_p::N;
    //k tile size
    if (params.seqlen_k == blocksize_c) {
        //k tile num==1
        fmha::device_1xN_<Kernel_traits, Is_dropout, Is_causal, Return_softmax, true, true>(params, bidb, bidh, STEPS, ph, 0);
    } else {
        const int max_loop_steps = (params.seqlen_k + blocksize_c - 1) / blocksize_c;
        // k tile num
        fmha::device_1xN_<Kernel_traits, Is_dropout, Is_causal, Return_softmax, true, false>(params, bidb, bidh, STEPS, ph, 0);
        for (int loop_step_idx = 1; loop_step_idx < max_loop_steps - 1; loop_step_idx++) {
            fmha::device_1xN_<Kernel_traits, Is_dropout, Is_causal, Return_softmax, false, false>(params, bidb, bidh, STEPS, ph, loop_step_idx);
        }
        fmha::device_1xN_<Kernel_traits, Is_dropout, Is_causal, Return_softmax, false, true>(params, bidb, bidh, STEPS, ph, max_loop_steps - 1);

//for q tile in head
template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Return_softmax, bool Is_first, bool Is_last, typename Params, typename Prng>
inline __device__ void device_1xN_(const Params &params, const int bidb, const int bidh, int steps, Prng &ph, const int loop_step_idx) {
    extern __shared__ char smem_[];
    // How many steps to jump per iteration, which is the same as params.num_splits.
    const int step_stride = gridDim.z;
    const BlockInfoPadded<Kernel_traits::THREADS> binfo(params, bidb, bidh, tidx);
    if( binfo.stop_early(loop_step_idx * Cta_tile_p::N) ) return;
    //no actual_seqlen_k left

    int begin = Is_causal ? loop_step_idx * Cta_tile_p::N / Cta_tile_p::M : 0;
    //当前 kv tile 看到的 q tile 的起点
    const int begin_mod_z = begin % gridDim.z;
    begin = begin_mod_z <= blockIdx.z ? begin - begin_mod_z : begin + gridDim.z - begin_mod_z;
    //以split为周期，找到第一个周期
    if ((begin + blockIdx.z) * Cta_tile_p::M >= binfo.actual_seqlen_q) return;
    //no actual_seqlen_q left
    const int steps_og = steps;
    //q tile num
    steps -= begin;
    //去掉kv不需要处理的q tile

    gmem_q.move(begin + blockIdx.z);
    //移动到需要处理的q tile
    gmem_o.move(begin + blockIdx.z);
    gmem_o_tmp.move(begin + blockIdx.z);
    if (Return_softmax) {
        gmem_s.move(begin + blockIdx.z);
    }
    gmem_softmax_lse.move(begin + blockIdx.z);

    if (!Is_first) {
        gmem_k.move(loop_step_idx);
        gmem_v.move(loop_step_idx);
        if (Return_softmax) { gmem_s.move(loop_step_idx * steps_og); }???
    }

    //qkv.ldg
    gmem_k.load();
    gmem_q.load();
    gmem_v.load();
    if (!Is_first) { __syncthreads(); }
    //wait softmax_lse.stg???
    float p_prev_lse[Mma_tile_p::MMAS_M * 2];
    if (!Is_first) {
        gmem_softmax_lse.load(reinterpret_cast<uint32_t(&)[Mma_tile_p::MMAS_M * 2]>(p_prev_lse));
        //lse_old==log(sum(exp(x*scale)))
    }

    //qv.sts
    gmem_q.commit(gemm_q_k.smem_q);
    gmem_v.commit(smem_v);
    if( !Kernel_traits::SHARE_SMEM_FOR_K_AND_V ) {
        gmem_k.commit(gemm_q_k.smem_k);
    }

    //qv.lds
    __syncthreads();
    //wait qv.sts???
    gemm_q_k.load_q();
    typename Smem_tile_v::Fragment frag_v[Mma_tile_o::MMAS_K][Mma_tile_o::MMAS_N];
    #pragma unroll
    for( int ki = 0; ki < Mma_tile_o::MMAS_K; ++ki ) {
        smem_v.load(frag_v[ki], ki);

    //k.sts
    if( Kernel_traits::SHARE_SMEM_FOR_K_AND_V ) {
        __syncthreads();
        //wait v.lds
        gmem_k.commit(gemm_q_k.smem_k);
        __syncthreads();
        //wait k.sts
    }

    //k.lds
    gemm_q_k.load_k();

    for (int l = blockIdx.z; l < steps; l += step_stride) {
        if ((begin + l) * Cta_tile_p::M >= binfo.actual_seqlen_q) break;
        //no actual_seqlen_q left

        //QK^T
        fmha::Fragment_accumulator acc_p[Mma_tile_p::MMAS_M][Mma_tile_p::MMAS_N];
        gemm_q_k(acc_p);

        //qo_tmp.ldg
        uint4 out[Gmem_tile_o::STGS_PER_LOOP];
        if (!Is_first) { gmem_o_tmp.load(out, 0); }
        //out_old
        if (l + step_stride < steps) {
            gemm_q_k.smem_q.move_to_next_write_buffer();
            gmem_q.move(step_stride);
            gmem_q.load();
        }
        // Load the mask for that iteration.
        mask.load(begin + l);

        //mask
        // Convert from the accumulator type to FP32 for Softmax.
        // frament->regs
        softmax.unpack_noscale(acc_p);
        // Apply the mask.
        // set -INFINITY
        softmax.apply_mask(mask);

        //lse_old/scale
        if( Kernel_traits::SHARE_SMEM_FOR_K_AND_V && l < step_stride ) {
            __syncthreads();
            //kv和softmax共享smem，等所有线程用完kv_smem
        }
        float p_max[Mma_tile_p::MMAS_M * 2];
        if (!Is_first) {
            smem_softmax_lse.store_pair(p_prev_lse);
            //lse_old==log(sum(exp(x*scale)))
            for (int mi = 0; mi < Mma_tile_p::MMAS_M * 2; mi++) { p_max[mi] = p_prev_lse[mi] / params.scale_bmm1f; }
            //log(sum(exp(x*scale)))/scale==log(sum(exp(x)))
        }
        if (l + step_stride < steps) {
            if (!Is_first) {
                gmem_softmax_lse.load_next(reinterpret_cast<uint32_t(&)[Mma_tile_p::MMAS_M * 2]>(p_prev_lse),
                                           step_stride);
            }
            //lse_old==log(sum(exp(x*scale)))
        }

        //m_new
        softmax.template reduce_max</*zero_init=*/Is_first>(p_max);
        //use regs,smem to compute m=max(max(x),log(sum(exp(x))))

        //p_numerator=exp((x-m)*scale)
        softmax.scale_apply_exp(p_max, params.scale_bmm1f);
        
        //denominator_new
        float p_sum[Mma_tile_p::MMAS_M * 2];
        softmax.reduce_sum_before_sync_(p_sum);
        //use regs,smem to compute sum(exp((x-m)*scale))

        //s.stg
        using Frag_p = fmha::Fragment_a<fmha::Row>;
        Frag_p frag_p[Mma_tile_o::MMAS_K][Mma_tile_o::MMAS_M];
        softmax.template pack<elem_type>(frag_p);
        //fragment=exp((x-m)*scale)
        if (Return_softmax) {
            gmem_s.store(frag_p, mask);
            gmem_s.move(step_stride);
        }

        //q.sts
        if (l + step_stride < steps) {
            gmem_q.commit(gemm_q_k.smem_q);
        }

        //p_numerator V
        fmha::Fragment_accumulator acc_o[Mma_tile_o::MMAS_M][Mma_tile_o::MMAS_N];
        #pragma unroll
        for( int ki = 0; ki < Mma_tile_o::MMAS_K; ++ki ) 
            fmha::gemm_cl<elem_type>(acc_o, frag_p[ki], frag_v[ki]);

        //m_new*scale
        float p_max_o[Gmem_tile_o::STGS_PER_LOOP][Mma_tile_o::MMAS_M];
        int rows[Gmem_tile_o::STGS_PER_LOOP];
        for (int jj = 0; jj < Gmem_tile_o::STGS_PER_LOOP; jj++) {
            rows[jj] = tidx / Gmem_tile_o::THREADS_PER_ROW + jj * Gmem_tile_o::ROWS_PER_STG;
        }
        softmax.reduce_max_after_sync_(p_max_o, rows);
        //按 O 输出线程的新行映射，重新读取、归约并广播之前的 m=max(max(x),log(sum(exp(x)))) 数据。
        for (int jj = 0; jj < Gmem_tile_o::STGS_PER_LOOP; jj++) {
            p_max_o[jj][0] *= params.scale_bmm1f;
        }
        float p_prev_scale_o[Gmem_tile_o::STGS_PER_LOOP];
        if (!Is_first) {
            smem_softmax_lse.load(p_prev_scale_o, rows);
            //lse_old==log(sum(exp(x*scale)))
        }

        //o.sts
        smem_o.store(acc_o, 0);
        //out_numerator_new

        //out_numerator_old_scale, denominator_new
        __syncthreads();//???
        float p_sum_o[Gmem_tile_o::STGS_PER_LOOP][Mma_tile_o::MMAS_M];
        softmax.reduce_sum_after_sync_(p_sum_o, rows);
        //sum(exp((x-m)*scale))
        if (!Is_first) {
            for (int jj = 0; jj < Gmem_tile_o::STGS_PER_LOOP; jj++) {
                p_prev_scale_o[jj] = expf(p_prev_scale_o[jj] - p_max_o[jj][0]);
                //log(sum(exp(x*scale)))-m_new*scale==log(sum(exp((x-m_new)*scale)))
                //sum(exp((x-m_new)*scale))
                p_sum_o[jj][0] += p_prev_scale_o[jj];
            }
        }

        //lse_new
        float p_sum_log[Gmem_tile_o::STGS_PER_LOOP][Mma_tile_o::MMAS_M];
        #pragma unroll
        for (int jj = 0; jj < Gmem_tile_o::STGS_PER_LOOP; jj++) {
            float sum = p_sum_o[jj][0];
            //sum(exp((x-m)*scale))
            p_sum_log[jj][0] = (sum == 0.f || sum != sum) ? -INFINITY : p_max_o[jj][0] + __logf(sum);
            //m*scale+log(sum(exp((x-m)*scale)))=log(sum(exp(x*scale)))
            if (tidx % Gmem_tile_o::THREADS_PER_ROW == 0) {
                gmem_softmax_lse.store_row(
                    reinterpret_cast<uint32_t(&)[Mma_tile_p::MMAS_M]>(p_sum_log[jj]), rows[jj]);
            }
            //store lse_new
        }
        gmem_softmax_lse.move(step_stride);

        //update out_numerator_old
        if (!Is_first) {
            for (int jj = 0; jj < Gmem_tile_o::STGS_PER_LOOP; jj++) {
                out[jj] = fmha::fmul4(out[jj], p_prev_scale_o[jj]);
                //out_numerator_old/sum(exp((x-m_old')*scale))*sum(exp((x-m_new)*scale))
                //out_numerator_old/exp((m_new-m_old')*scale)
                //exp((x-m_new)*scale)V
            }
        }

        //out_numerator_new
        smem_o.template load</*zero_init=*/Is_first>(out);

        //out
        const bool is_final_write =
            Is_last
            //max_seqlen_k end
            || ((loop_step_idx + 1) * Cta_tile_p::N >= binfo.actual_seqlen_k)
            //no actual_seqlen_k left
            || ((Is_causal) && ((begin + l) * Cta_tile_p::M < (loop_step_idx + 1) * Cta_tile_p::N));
            //q has no k left
        //o is finish
        #pragma unroll
        for (int jj = 0; jj < Gmem_tile_o::STGS_PER_LOOP; jj++) {
            float sum = p_sum_o[jj][0];
            //sum(exp((x-m_new)*scale))
            float inv_sum = (sum == 0.f || sum != sum) ? 1.f : 1.f / sum;
            if (Is_dropout && is_final_write) {
                inv_sum *= params.rp_dropout;
            }
            out[jj] = fmha::fmul4(out[jj], inv_sum);
            //out_numerator_new/sum(exp((x-m_new)*scale))==out
        }
        if (is_final_write) {
            gmem_o.template store<elem_type>(out, 0);
            gmem_o.move(step_stride);
        } else {
            gmem_o_tmp.store(out, 0);
        }
        // Move to the next part of the output.
        if (!(Is_first && Is_last)) { gmem_o_tmp.move(step_stride); }

        // Make sure we are reading from the correct buffer.
        gemm_q_k.smem_q.move_to_next_read_buffer();
        // Trigger the load from shared memory for the next series of Q values.
        if (l + step_stride < steps) {
            gemm_q_k.reload_q();
            //lds
        }
```

## triton

### forward
seqlen_q//q_num_splits==q_tile_size，内层q_loop没有了

先只考虑分子，最后再除以分母
```
grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch * nheads)

load Q block
for each K/V block:
    load K block
    scores = q @ k.T
    m_new=max(max(scores*scale,1),lse)
    p = exp(scores*scale - m_new)
    sum_p=sum(p,1)
    out_scale=exp(m-m_new)
    out *= out_scale
    load V block
    out += p @ v
    m=m_new
    sum_p_new=exp(lse-m_new)+sum_p
    lse=m_new+log(sum_p_new)
out *= exp(m - lse)
store lse
store out block
```

flash-attention/flash_attn/flash_attn_triton.py
```
flash_attn_func = FlashAttnFunc.apply

class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, bias=None, causal=False, softmax_scale=None):
        o, lse, ctx.softmax_scale = _flash_attn_forward(
        ctx.save_for_backward(q, k, v, o, lse, bias)
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse, bias = ctx.saved_tensors
        with torch.inference_mode():
            dq = torch.empty_like(q)
            dk = torch.empty_like(k)
            dv = torch.empty_like(v)
            _flash_attn_backward(

def _flash_attn_forward(q, k, v, bias=None, causal=False, softmax_scale=None):
    batch, seqlen_q, nheads, d = q.shape

    seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128
    lse = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)
    o = torch.empty_like(q)

    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    # padding
    BLOCK = 128
    num_warps = 4 if d <= 64 else 8
    grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch * nheads)

    _fwd_kernel[grid](
        BLOCK_M=BLOCK,
        BLOCK_N=BLOCK,
        num_warps=num_warps,
        num_stages=1,

@triton.jit
def _fwd_kernel(
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    q_ptrs = (
        Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + offs_d[None, :])
    )
    #[BLOCK_M,BLOCK_HEADDIM]
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)

    q = tl.load(q_ptrs)
    end_n = seqlen_k if not IS_CAUSAL else tl.minimum((start_m + 1) * BLOCK_M, seqlen_k)
    for start_n in range(0, end_n, BLOCK_N):
        k = tl.load(k_ptrs + start_n * stride_kn)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k, trans_b=True)
        if IS_CAUSAL:
            qk += tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], 0, float("-inf"))
            #condition 为 True  -> 取 0
            #condition 为 False -> 取 -inf
            #softmax(-inf) = 0
        m_ij = tl.maximum(tl.max(qk, 1) * softmax_scale, lse_i)         
        p = tl.exp(qk * softmax_scale - m_ij[:, None])     
        l_ij = tl.sum(p, 1)
        acc_o_scale = tl.exp(m_i - m_ij)
        tl.store(t_ptrs, acc_o_scale)
        acc_o_scale = tl.load(t_ptrs)
        acc_o = acc_o * acc_o_scale[:, None]
        v = tl.load(v_ptrs + start_n * stride_vn)
        p = p.to(v.dtype)
        acc_o += tl.dot(p, v)
        m_i = m_ij
        l_i_new = tl.exp(lse_i - m_ij) + l_ij
        lse_i = m_ij + tl.log(l_i_new)
    o_scale = tl.exp(m_i - lse_i)
    tl.store(t_ptrs, o_scale)
    o_scale = tl.load(t_ptrs)
    acc_o = acc_o * o_scale[:, None]
    lse_ptrs = Lse + off_hb * seqlen_q_rounded + offs_m
    tl.store(lse_ptrs, lse_i)
    out_ptrs = (
        Out
        + off_b * stride_ob
        + off_h * stride_oh
        + (offs_m[:, None] * stride_om + offs_d[None, :])
    )
    if EVEN_M:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o)
```

### backward

```
P = softmax(S)
dS_ij = P_ij * (dP_ij - sum_k P_ik * dP_ik)
= P_ij * (dP_ij - P_i dP_i^T)
= P_ij * (dP_ij - delta_i)

delta_i = P_i dP_i^T 
= P_i (dO_i V^T)^T 
= P_i V dO_i^T 
= O_i dO_i^T

O = P V
dP = dO V^T
dP_ij = dO_i V_j^T
dV = P^T dO
dV_j = P_*j^T dO
dV_j = sum_i(P_ij^T dO_i)

S = Q K^T
dK^T = Q^T dS
dK = dS^T Q
dK_j = sum_i(dS_ij^T Q_i)
dQ = dS K
dQ_i = sum_j(dS_ij K_j)
```

计算delta_i

计算P_ij，更新dV_j，计算dP_ij，dS_ij，更新dK_j，dQ_i

伪代码：
```
grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch * nheads)
for O block:
    load O, dO block
    delta = tl.sum(o * do, axis=1)
    tl.store(delta)

grid = lambda META: (
    triton.cdiv(seqlen_k, META["BLOCK_N"]) if META["SEQUENCE_PARALLEL"] else 1,
    batch * nheads,
)
#for K,V:
#可并行
k = tl.load(k_ptrs)
v = tl.load(v_ptrs)
for Q block:
    q = tl.load(q_ptrs)
    qk = tl.dot(q, k, trans_b=True)
    lse_i = tl.load(LSE + offs_m_curr)
    p = tl.exp(qk * softmax_scale - lse_i[:, None])
    do = tl.load(do_ptrs)
    dv += tl.dot(p.to(do.dtype), do, trans_a=True)
    dp = tl.dot(do, v, trans_b=True)
    Di = tl.load(D + offs_m_curr)
    ds = (p * (dp - Di[:, None]) * softmax_scale).to(q.dtype)
    dk += tl.dot(ds, q, trans_a=True)
    #dq = tl.load(dq_ptrs, eviction_policy="evict_last")
    #dq += tl.dot(ds, k)
    #tl.store(dq_ptrs, dq, eviction_policy="evict_last")
    dq = tl.dot(ds, k)
    tl.atomic_add(dq_ptrs, dq)
    #K,V block并行的话，每个block都会更新dQ，需要原子操作
tl.store(dv_ptrs, dv)
tl.store(dk_ptrs, dk)
```

```
def _flash_attn_backward(
    do, q, k, v, o, lse, dq, dk, dv, bias=None, causal=False, softmax_scale=None
):
    grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch * nheads)
    _bwd_preprocess_do_o_dot[grid](
    #计算delta_i
    
    grid = lambda META: (
        triton.cdiv(seqlen_k, META["BLOCK_N"]) if META["SEQUENCE_PARALLEL"] else 1,
        batch * nheads,
    )
    #[1,batch * nheads]
    #[k_num_splits,batch * nheads]
    _bwd_kernel[grid](
        seqlen_q // 32,  seqlen_k // 32, # key for triton cache (limit number of compilations)

@triton.jit
def _bwd_preprocess_do_o_dot(
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    o = tl.load(Out + off_b * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :],
    do = tl.load(
    delta = tl.sum(o * do, axis=1)
    tl.store(Delta + off_hb * seqlen_q_rounded + offs_m, delta)

@triton.jit
def _bwd_kernel(
    if not SEQUENCE_PARALLEL:
        num_block_n = tl.cdiv(seqlen_k, BLOCK_N)
        for start_n in range(0, num_block_n):
            _bwd_kernel_one_col_block(
                ATOMIC_ADD=False,
    else:
        start_n = tl.program_id(0)
        _bwd_kernel_one_col_block(
            ATOMIC_ADD=True,

@triton.jit
def _bwd_kernel_one_col_block(
    begin_m = 0 if not IS_CAUSAL else ((start_n * BLOCK_N) // BLOCK_M) * BLOCK_M
    offs_qm = begin_m + tl.arange(0, BLOCK_M)
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
  
    k = tl.load(k_ptrs)
    v = tl.load(v_ptrs)
    num_block_m = tl.cdiv(seqlen_q, BLOCK_M)
    for start_m in range(begin_m, num_block_m * BLOCK_M, BLOCK_M):
        offs_m_curr = start_m + offs_m
        q = tl.load(q_ptrs)
        qk = tl.dot(q, k, trans_b=True)
        if IS_CAUSAL:
            qk = tl.where(offs_m_curr[:, None] >= (offs_n[None, :]), qk, float("-inf"))
        lse_i = tl.load(LSE + offs_m_curr)
        p = tl.exp(qk * softmax_scale - lse_i[:, None])
        do = tl.load(do_ptrs)
        dv += tl.dot(p.to(do.dtype), do, trans_a=True)
        dp = tl.dot(do, v, trans_b=True)
        Di = tl.load(D + offs_m_curr)
        ds = (p * (dp - Di[:, None]) * softmax_scale).to(q.dtype)
        dk += tl.dot(ds, q, trans_a=True)
        if not ATOMIC_ADD:
            dq = tl.load(dq_ptrs, eviction_policy="evict_last")
            dq += tl.dot(ds, k)
            tl.store(dq_ptrs, dq, eviction_policy="evict_last")
        else:  
            dq = tl.dot(ds, k)
            tl.atomic_add(dq_ptrs, dq)
        dq_ptrs += BLOCK_M * stride_dqm
        q_ptrs += BLOCK_M * stride_qm
        do_ptrs += BLOCK_M * stride_dom
    dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_d[None, :])
    dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_d[None, :])
    _bwd_store_dk_dv(

@triton.jit
def _bwd_store_dk_dv(
    tl.store(dv_ptrs, dv)
    tl.store(dk_ptrs, dk)
```

# v2

2 speedup compared to FlashAttention

csrc
```
grid=[b, h, q_num_splits]
for q
    for k,v
```

先只考虑分子，最后再除以分母，减少了非矩阵乘法运算。调整loop顺序，不需要load o，IO次数减少

# decoding

Q小，利用率不足，将KV长序列切成多个子序列并行，子序列再做reduce

flash_attn_with_kvcache 

we split the loading across different thread blocks, with a separate kernel to combine results.

Flash-Decoding 和 FlashDecoding++ 属于完全不同的两个独立研究团队

# v3

FlashAttention-3 is optimized for Hopper GPUs. Compare it to a standard implementation in PyTorch, FlashAttention-2, FlashAttention-2 in Triton (which uses H100-specific instructions), as well as a vendor’s implementation of FlashAttention-2 optimized for H100 GPUs from cuDNN. FlashAttention-3 is up to 2.0× faster than FlashAttention-2

flash-attention/hopper

一部分 warp 负责 TMA 数据搬运，一部分 warp 负责 WGMMA 计算，一部分 warp 负责处理 softmax / rescale

数据搬运和计算重叠，GEMM 和 softmax 交错执行

FP8

# v4

FlashAttention-4 is written in CuTeDSL and optimized for Hopper and Blackwell GPUs, achieves up to 1.3 speedup over cuDNN 9.13 and 2.7 over Triton on B200 GPUs with BF16

flash-attention/flash_attn/cute

leveraging tensor memory and the 2-CTA MMA mode to reduce shared memory traffic and atomic adds in the backward pass. 