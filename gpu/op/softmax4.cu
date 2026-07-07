#include<cuda.h>
#include<float.h>
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
__global__ void kSoftmax(float*A,float*B,const unsigned int len){
    A+=blockIdx.x*len;
    B+=blockIdx.x*len;
    int dataPerThread=(len+blockSize-1)/blockSize;

    float threadMax=FLT_MIN;
    for(int i=0;i<dataPerThread;i++){
        if(threadIdx.x+i*blockSize<len){
            threadMax=fmaxf(threadMax,A[threadIdx.x+i*blockSize]);
        }
    }
    __shared__ float sm[blockSize];
    int step=blockSize/2;
    sm[threadIdx.x]=threadMax;
    while(step){
        __syncthreads();
        if(threadIdx.x<step){
            sm[threadIdx.x]=fmaxf(sm[threadIdx.x],sm[threadIdx.x+step]);
        }
        step/=2;
    }
    __syncthreads();
    float blockMax=sm[0];

    float threadSum=0;
    for(int i=0;i<dataPerThread;i++){
        if(threadIdx.x+i*blockSize<len){
            B[threadIdx.x+i*blockSize]=__expf(A[threadIdx.x+i*blockSize]-blockMax);
            threadSum+=B[threadIdx.x+i*blockSize];
        }
    }
    step=blockSize/2;
    sm[threadIdx.x]=threadSum;
    while(step){
        __syncthreads();
        if(threadIdx.x<step){
            sm[threadIdx.x]+=sm[threadIdx.x+step];
        }
        step/=2;
    }
    __syncthreads();
    float blockSum=sm[0];

    for(int i=0;i<dataPerThread;i++){
        if(threadIdx.x+i*blockSize<len){
            B[threadIdx.x+i*blockSize]/=blockSum;
        }
    }
}
template<const unsigned int BLOCKSIZE>
void dSoftmax(float*A,float*B,const unsigned int len,const unsigned int numBatch){
    dim3 gd={numBatch,1,1};
    dim3 bd={BLOCKSIZE,1,1};
    kSoftmax<BLOCKSIZE><<<gd,bd>>>(A,B,len);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}
void hSoftmax(float*A,float*B,const unsigned int len,const unsigned int numBatch){
    for(int i=0;i<numBatch;i++){
        float*hA=A+i*len;
        float*hB=B+i*len;
        float blockMax=FLT_MIN;
        for(int j=0;j<len;j++){
            blockMax=fmaxf(blockMax,hA[j]);
        }
        float blockSum=0;
        for(int j=0;j<len;j++){
            blockSum+=expf(hA[j]-blockMax);
        }
        for(int j=0;j<len;j++){
            hB[j]=exp(hA[j]-blockMax)/blockSum;
        }
    }
}
void init(float*hA,int len){
    for(int i=0;i<len;i++){
        hA[i]=rand()%1000;
    }
}
void check(float*A,float*B,int len){
    bool right=true;
    for(int i=0;i<len;i++){
        if(std::abs(A[i]-B[i])>1e-6){
            printf("%d %f %f\n",i,A[i],B[i]);
            right=false;
        }
    }
    if(right){
        printf("right\n");
    }
}
int main(int argc,char*argv[]){
    int len=10000,numBatch=1000;
    if(argc>1){
        len=std::stoi(argv[1]);
        if(argc>2){
            numBatch=std::stoi(argv[2]);
        }
    }
    float*hA,*hB;
    hA=(float*)malloc(sizeof(float)*len*numBatch);
    hB=(float*)malloc(sizeof(float)*len*numBatch);
    init(hA,len*numBatch);
    hSoftmax(hA,hB,len,numBatch);

    float*dA,*dB,*hB1;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&dA,sizeof(float)*len*numBatch));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&dB,sizeof(float)*len*numBatch));
    hB1=(float*)malloc(sizeof(float)*len*numBatch);
    CHECK_CUDA_ERROR(cudaMemcpy(dA,hA,sizeof(float)*len*numBatch,cudaMemcpyHostToDevice));
    dSoftmax<512>(dA,dB,len,numBatch);
    CHECK_CUDA_ERROR(cudaMemcpy(hB1,dB,sizeof(float)*len*numBatch,cudaMemcpyDeviceToHost));

    check(hB,hB1,len*numBatch);
    free(hA);
    free(hB);
    free(hB1);
    cudaFree(dA);
    cudaFree(dB);
    return 0;
}