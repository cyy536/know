#include<cuda.h>
#include<stdio.h>
#include<stdlib.h>
#include<string>
#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}
template<const unsigned int blockSize>
__global__ void kReduce(float*A,float*B,float*out,int len){
    // 每次缩小一半数据量
    int blockBegin=blockIdx.x*blockSize*2;
    A+=blockBegin;
    __shared__ float sA[blockSize];
    int step=blockSize;
    if(blockBegin+threadIdx.x<len){
        sA[threadIdx.x]=A[threadIdx.x];
        if(blockBegin+threadIdx.x+step<len){
            sA[threadIdx.x]+=A[threadIdx.x+step];
        }
    }
    step/=2;
    while(step){
        __syncthreads();
        if(threadIdx.x<step&&blockBegin+threadIdx.x+step<len){
            sA[threadIdx.x]+=sA[threadIdx.x+step];
        }
        step/=2;
    }
    if(threadIdx.x==0){
        B[blockIdx.x]=sA[0];
        atomicAdd(out,sA[0]);
    }
}
template<const unsigned int blockSize>
__global__ void kReduce1(float*A,float*B,float*out,int len){
    // 分组reduce
    int blockBegin=blockIdx.x*blockSize*2;
    A+=blockBegin;
    float sum=0;
    if(blockBegin+threadIdx.x<len){
        sum=A[threadIdx.x];
        if(blockBegin+threadIdx.x+blockSize<len){
            sum+=A[threadIdx.x+blockSize];
        }
    }

    sum+=__shfl_down_sync(0xffffffff,sum,16);
    sum+=__shfl_down_sync(0xffffffff,sum,8);
    sum+=__shfl_down_sync(0xffffffff,sum,4);
    sum+=__shfl_down_sync(0xffffffff,sum,2);
    sum+=__shfl_down_sync(0xffffffff,sum,1);
    int warpIdx=threadIdx.x>>5;
    int laneIdx=threadIdx.x&31;
    __shared__ float ss[32];
    if(laneIdx==0){
        ss[warpIdx]=sum;
    }
    __syncthreads();
    sum=laneIdx*32<blockSize?ss[laneIdx]:0;
    sum+=__shfl_down_sync(0xffffffff,sum,16);
    sum+=__shfl_down_sync(0xffffffff,sum,8);
    sum+=__shfl_down_sync(0xffffffff,sum,4);
    sum+=__shfl_down_sync(0xffffffff,sum,2);
    sum+=__shfl_down_sync(0xffffffff,sum,1);

    if(threadIdx.x==0){
        B[blockIdx.x]=sum;
        atomicAdd(out,sum);
    }
}
template<const unsigned int BLOCKSIZE>
void dReduce(float*A,float*B,float*out,const unsigned int len){
    dim3 gridDim={(len+2*BLOCKSIZE-1)/(2*BLOCKSIZE),1,1};
    dim3 blockDim={BLOCKSIZE,1,1};
    kReduce1<BLOCKSIZE><<<gridDim,blockDim>>>(A,B,out,len);
    cudaDeviceSynchronize();
}
template<const unsigned int BLOCKSIZE>
void hReduce(float*A,float*B,float*out,const unsigned int len){
    int datasize=2*BLOCKSIZE;
    for(int i=0;i<len;i+=datasize){
        float sum=0;
        for(int j=i;j<i+datasize&&j<len;j++){
            sum+=A[j];
        }
        B[i/datasize]=sum;
        *out+=sum;
    }
}
void check(float*A,float*B,int len){
    bool right=true;
    for(int i=0;i<len;i++){
        if(std::abs(A[i]-B[i])>1e-6){
            right=false;
            printf("%d %f %f\n",i,A[i],B[i]);
        }
    }
    if(right){
        printf("right\n");
    }
}
void init(float*A,int len){
    for(int i=0;i<len;i++){
        A[i]=rand()%1000;
    }
}
int main(int argc,char*argv[]){
    int len=1000000;
    if(argc>1){
        len=std::stoi(argv[1]);
    }
    const unsigned int bs=512;
    int gs=(len+2*bs-1)/(2*bs);
    float*hA,*hB,*hout;
    hA=(float*)malloc(sizeof(float)*len);
    hB=(float*)malloc(sizeof(float)*gs);
    hout=(float*)malloc(sizeof(float));
    init(hA,len);
    *hout=0;
    hReduce<bs>(hA,hB,hout,len);
    
    float*dA,*dB,*dout,*hB1,*hout1;
    cudaMalloc((void**)&dA,sizeof(float)*len);
    cudaMemcpy(dA,hA,sizeof(float)*len,cudaMemcpyHostToDevice);
    cudaMalloc((void**)&dB,sizeof(float)*gs);
    cudaMalloc((void**)&dout,sizeof(float));
    CHECK_CUDA_ERROR(cudaMemset(dout,0,sizeof(float)));
    hB1=(float*)malloc(sizeof(float)*gs);
    hout1=(float*)malloc(sizeof(float));
    dReduce<bs>(dA,dB,dout,len);
    cudaMemcpy(hB1,dB,sizeof(float)*gs,cudaMemcpyDeviceToHost);
    cudaMemcpy(hout1,dout,sizeof(float),cudaMemcpyDeviceToHost);
    
    check(hB,hB1,gs);
    check(hout,hout1,1);
    free(hA);
    free(hB);
    free(hB1);
    free(hout);
    free(hout1);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dout);
    return 0;
}