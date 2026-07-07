#include<cuda.h>
#include<stdlib.h>
#include<stdio.h>
#include<string>
#include<curand_kernel.h>
#include<sys/time.h>
#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}
template<const unsigned int blockSize>
__global__ void intraBlockSum(int *A,int *buffer,int *B,int len){
    unsigned int blockNum=(len+blockSize-1)/blockSize;
    __shared__ int bb[blockSize],ss[32];
    for(int i=blockIdx.x;i<blockNum;i+=gridDim.x){
        int *bA=A+i*blockSize;
        int *bB=B+i*blockSize;
        // read to share
        bb[threadIdx.x]=i*blockSize+threadIdx.x<len?bA[threadIdx.x]:0;
        __syncthreads();

        int warpIdx=threadIdx.x>>5;
        int laneIdx=threadIdx.x&31;
        int *wb=bb+warpIdx*32;
        // warp sum
        if(laneIdx==0){
            int ans=0;
            for(int i=0;i<32;i++){
                ans+=wb[i];
                wb[i]=ans;
            }
            ss[warpIdx]=wb[31];
        }
        __syncthreads();
        // addPre in warp buffer
        if(warpIdx==0&&laneIdx==0){
            int ans=0;
            for(int i=0;i<32;i++){
                ans+=ss[i];
                ss[i]=ans;
            }
        }
        __syncthreads();
        // addPre
        if(warpIdx>=1){
            wb[laneIdx]+=ss[warpIdx-1];
        }

        // write buffer
        if(threadIdx.x==blockSize-1){
            buffer[i]=bb[blockSize-1];
        }
        
        // write output
        if(i*blockSize+threadIdx.x<len){
            bB[threadIdx.x]=bb[threadIdx.x];
        }
        __syncthreads();
    }
}
template<const unsigned int blockSize>
__global__ void interBlockSum(int *A,int len){
    int ans=0;
    for(int i=0;i<len;i++){
        ans+=A[i];
        A[i]=ans;
    }
}
template<const unsigned int blockSize>
__global__ void addPre(int *A,int *buffer,int len){
    unsigned int blockNum=(len+blockSize-1)/blockSize;
    for(int i=blockIdx.x;i<blockNum;i+=gridDim.x){
        int *bA=A+i*blockSize;
        if(i>=1&&i*blockSize+threadIdx.x<len){
           bA[threadIdx.x]+=buffer[i-1]; 
        }
    }
}
template<const unsigned int blockSize>
void dPreSum(int *A,int *buffer,int *B,int len){
    int blockNum=(len+blockSize-1)/blockSize;
    dim3 blockDim={blockSize,1,1};
    dim3 gridDim={(unsigned int)std::min(blockNum,128),1,1};
    intraBlockSum<blockSize><<<gridDim,blockDim>>>(A,buffer,B,len);
    interBlockSum<blockSize><<<1,1>>>(buffer,blockNum);
    addPre<blockSize><<<gridDim,blockDim>>>(B,buffer,len);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}
void hPreSum(const int* input, size_t n, int* output) {
    int sum = 0;
    for (size_t i = 0; i < n; ++i) {
      sum += input[i];
      output[i] = sum;
    }
}
void init(int*hA,int len){
    for(int i=0;i<len;i++){
        hA[i]=rand()%1000;
    }
}
void check(int*A,int*B,int len){
    bool right=true;
    for(int i=0;i<len;i++){
        if(A[i]!=B[i]){
            right=false;
            printf("%d %d %d\n",i,A[i],B[i]);
        }
    }
    if(right){
        printf("right\n");
    }
}
int main(int argc,char*argv[]){
    int len=100000;
    if(argc>1){
        len=std::stoi(argv[1]);
    }
    int *hA,*hO;
    hA=(int*)malloc(sizeof(int)*len);
    hO=(int*)malloc(sizeof(int)*len);
    init(hA,len);
    hPreSum(hA,len,hO);
    int *dA,*dB,*dO,*hO1;
    cudaMalloc((void**)&dA,sizeof(int)*len);
    cudaMalloc((void**)&dB,sizeof(int)*len);
    cudaMalloc((void**)&dO,sizeof(int)*len);
    hO1=(int*)malloc(sizeof(int)*len);
    cudaMemcpy(dA,hA,sizeof(int)*len,cudaMemcpyHostToDevice);
    dPreSum<512>(dA,dB,dO,len);
    cudaMemcpy(hO1,dO,sizeof(int)*len,cudaMemcpyDeviceToHost);
    check(hO,hO1,len);

    free(hA);
    free(hO);
    free(hO1);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dO);
}