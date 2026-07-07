#include<cuda.h>
#include<stdio.h>
#include<cub/cub.cuh>
#include<sys/time.h>
#define BLOCK_SIZE 256

double getTime(){
    struct timeval tv;
    gettimeofday(&tv,NULL);
    return (double)(tv.tv_sec*1e3+tv.tv_usec/1e3);
}

void init(float*hA,int M){
    for(int i=0;i<M;i++){
        hA[i]=rand()%1000;    
    }
}

void check(float*hC1,float*hC2,int M){
    for(int m=0;m<M;m++){
        if(abs(hC1[m]-hC2[m])>1e-6){
            printf("error:%f %f\n",hC1[m],hC2[m]);
        }
    }
}

void hsoftmax(float*hA,float*hB,int M,int B){
    for(int b=0;b<B;b++){
        float*ha=hA+b*M;
        float*hb=hB+b*M;
        float m_total = -FLT_MAX;
        for(int m=0;m<M;m++){
            m_total=fmaxf(m_total, ha[m]);
        }
        float d = 0.0F;
        for(int m=0;m<M;m++){
            d += expf(ha[m] - m_total);
        }
        float d_total_inverse = 1.0F/d;
        for(int m=0;m<M;m++){
            hb[m] = expf(ha[m] - m_total) * d_total_inverse;
        }
    }
}

template<int THREADBLOCK_SIZE>
__global__ void naive_softmax(
    const float * x,
    float * y,
    int V)
{
    int thread_id = threadIdx.x;
    int vector_id = blockIdx.x;

    x += vector_id * V;
    y += vector_id * V;

    typedef cub::BlockReduce<float, THREADBLOCK_SIZE> BlockReduce;

    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ float d_total_inverse;

    float d_partial = 0.0F;
    for(int elem_id = thread_id; elem_id < V; elem_id += THREADBLOCK_SIZE)
        d_partial += __expf(x[elem_id]);

    float d = BlockReduce(temp_storage).Sum(d_partial);
    if (thread_id == 0)
        d_total_inverse = __fdividef(1.0F, d);
    __syncthreads();

    for(int elem_id = thread_id; elem_id < V; elem_id += THREADBLOCK_SIZE)
        y[elem_id] = __expf(x[elem_id]) * d_total_inverse;
}

__device__ __forceinline__ float max_op(float a, float b)
{
    return fmaxf(a, b);
}

template<int THREADBLOCK_SIZE>
__global__ void safe_softmax(
    const float * x,
    float * y,
    int V)
{
    int thread_id = threadIdx.x;
    int vector_id = blockIdx.x;

    x += vector_id * V;
    y += vector_id * V;

    typedef cub::BlockReduce<float, THREADBLOCK_SIZE> BlockReduce;

    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ float m_total;
    __shared__ float d_total_inverse;

    float m_partial = -FLT_MAX;
    for(int elem_id = thread_id; elem_id < V; elem_id += THREADBLOCK_SIZE)
        m_partial = max_op(m_partial, x[elem_id]);

    float m = BlockReduce(temp_storage).Reduce(m_partial, max_op);
    if (thread_id == 0)
        m_total = m;
    __syncthreads();

    float d_partial = 0.0F;
    for(int elem_id = thread_id; elem_id < V; elem_id += THREADBLOCK_SIZE)
        d_partial += __expf(x[elem_id] - m_total);

    float d = BlockReduce(temp_storage).Sum(d_partial);
    if (thread_id == 0)
        d_total_inverse = __fdividef(1.0F, d);
    __syncthreads();

    for(int elem_id = thread_id; elem_id < V; elem_id += THREADBLOCK_SIZE)
        y[elem_id] = __expf(x[elem_id] - m_total) * d_total_inverse;
}

struct __align__(8) MD
{
    float m;
    float d;
};

__device__ __forceinline__ MD reduce_md_op(MD a, MD b)
{
    bool a_bigger = (a.m > b.m);
    MD bigger_m = a_bigger ? a : b;
    MD smaller_m = a_bigger ? b : a;
    MD res;
    res.d = bigger_m.d + smaller_m.d * __expf(smaller_m.m - bigger_m.m);
    res.m = bigger_m.m;
    return res;
}

template<int THREADBLOCK_SIZE>
__global__ void online_softmax(
    const float * x,
    float * y,
    int V)
{
    int thread_id = threadIdx.x;
    int vector_id = blockIdx.x;

    x += vector_id * V;
    y += vector_id * V;

    typedef cub::BlockReduce<MD, THREADBLOCK_SIZE> BlockReduce;

    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ MD md_total;

    MD md_partial;
    md_partial.m = -FLT_MAX;
    md_partial.d = 0.0F;
    for(int elem_id = thread_id; elem_id < V; elem_id += THREADBLOCK_SIZE)
    {
        MD new_elem;
        new_elem.m = x[elem_id];
        new_elem.d = 1.0F;
        md_partial = reduce_md_op(md_partial, new_elem);
    }

    MD md = BlockReduce(temp_storage).Reduce(md_partial, reduce_md_op);
    if (thread_id == 0)
        md_total = md;
    __syncthreads();

    float d_total_inverse = __fdividef(1.0F, md_total.d);
    for(int elem_id = thread_id; elem_id < V; elem_id += THREADBLOCK_SIZE)
        y[elem_id] = __expf(x[elem_id] - md_total.m) * d_total_inverse;
}

void dnaive(float*hA,float*hB,unsigned int M,unsigned int B){
    double cstart=getTime();
    float*dA,*dB;
    cudaMalloc((void**)&dA,sizeof(float)*M*B);
    cudaMalloc((void**)&dB,sizeof(float)*M*B);

    cudaMemcpy(dA,hA,sizeof(float)*M*B,cudaMemcpyHostToDevice);

    dim3 grid_d={B,1,1};
    dim3 block_d={BLOCK_SIZE,1,1};

    cudaEvent_t start,end;
    float dur;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    naive_softmax<BLOCK_SIZE><<<grid_d,block_d>>>(dA,dB,M);
    cudaEventRecord(end);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&dur,start,end);

    printf("kernel:%f\n",dur);

    float*hB1=(float*)malloc(sizeof(float)*M*B);
    cudaMemcpy(hB1,dB,sizeof(float)*M*B,cudaMemcpyDeviceToHost);

    double cend=getTime();
    printf("device:%lf\n",cend-cstart);

    check(hB,hB1,M);

    cudaFree(dA);
    cudaFree(dB);
    free(hB1);
}

void dsafe(float*hA,float*hB,unsigned int M,unsigned int B){
    double cstart=getTime();
    float*dA,*dB;
    cudaMalloc((void**)&dA,sizeof(float)*M*B);
    cudaMalloc((void**)&dB,sizeof(float)*M*B);

    cudaMemcpy(dA,hA,sizeof(float)*M*B,cudaMemcpyHostToDevice);

    dim3 grid_d={B,1,1};
    dim3 block_d={BLOCK_SIZE,1,1};

    cudaEvent_t start,end;
    float dur;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    safe_softmax<BLOCK_SIZE><<<grid_d,block_d>>>(dA,dB,M);
    cudaEventRecord(end);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&dur,start,end);

    printf("kernel:%f\n",dur);

    float*hB1=(float*)malloc(sizeof(float)*M*B);
    cudaMemcpy(hB1,dB,sizeof(float)*M*B,cudaMemcpyDeviceToHost);

    double cend=getTime();
    printf("device:%lf\n",cend-cstart);

    check(hB,hB1,M);

    cudaFree(dA);
    cudaFree(dB);
    free(hB1);
}

void donline(float*hA,float*hB,unsigned int M,unsigned int B){
    double cstart=getTime();
    float*dA,*dB;
    cudaMalloc((void**)&dA,sizeof(float)*M*B);
    cudaMalloc((void**)&dB,sizeof(float)*M*B);

    cudaMemcpy(dA,hA,sizeof(float)*M*B,cudaMemcpyHostToDevice);

    dim3 grid_d={B,1,1};
    dim3 block_d={BLOCK_SIZE,1,1};

    cudaEvent_t start,end;
    float dur;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    online_softmax<BLOCK_SIZE><<<grid_d,block_d>>>(dA,dB,M);
    cudaEventRecord(end);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&dur,start,end);

    printf("kernel:%f\n",dur);

    float*hB1=(float*)malloc(sizeof(float)*M*B);
    cudaMemcpy(hB1,dB,sizeof(float)*M*B,cudaMemcpyDeviceToHost);

    double cend=getTime();
    printf("device:%lf\n",cend-cstart);

    check(hB,hB1,M);

    cudaFree(dA);
    cudaFree(dB);
    free(hB1);
}

int main(int argc, char*argv[]){
    int M=1000,B=100;
    if(argc>1){
        M=atoi(argv[1]);
        B=atoi(argv[2]);
    }
    printf("%d %d\n",M,B);

    float*hA=(float*)malloc(sizeof(float)*M*B);
    float*hB=(float*)malloc(sizeof(float)*M*B);

    init(hA,M*B);

    double start,end;
    start=getTime();
    hsoftmax(hA,hB,M,B);
    end=getTime();
    printf("host:%lf\n",end-start);
    free(hA);
    free(hB);

    dnaive(hA,hB,M,B);
    dsafe(hA,hB,M,B);
    donline(hA,hB,M,B);
    return 0;
}