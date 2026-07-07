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
#define toFloat(x) (*((float4*)(&(x))))
template<const unsigned int blockSize,const unsigned int dpt>
__global__ void kmm(float*A,float*B,float*C,int M,int K,int N){
    constexpr int dataBlock=blockSize*dpt;

    A+=blockSize*blockIdx.y*dpt*K;
    B+=blockSize*blockIdx.x*dpt;
    C+=blockSize*blockIdx.y*dpt*N+blockSize*blockIdx.x*dpt;
    __shared__ float sA[dataBlock][dataBlock],sB[dataBlock][dataBlock];
    float ans[dpt][dpt];
    for(int i=0;i<dpt;i++){
        for(int j=0;j<dpt;j++){
            ans[i][j]=0;
        }
    }
    for(int k=0;k<K;k+=dataBlock){
        for(int i=0;i<dpt;i++){
            for(int j=0;j<dpt;j+=4){
                toFloat(sA[threadIdx.y*dpt+i][threadIdx.x*dpt+j])=toFloat(A[(threadIdx.y*dpt+i)*K+threadIdx.x*dpt+j+k]);
                toFloat(sB[threadIdx.y*dpt+i][threadIdx.x*dpt+j])=toFloat(B[(threadIdx.y*dpt+i+k)*N+threadIdx.x*dpt+j]);
            }
        }
        __syncthreads();
        for(int i=0;i<dpt;i++){
            for(int j=0;j<dpt;j++){
                for(int kk=0;kk<dataBlock;kk++){
                    ans[i][j]+=sA[threadIdx.y*dpt+i][kk]*sB[kk][threadIdx.x*dpt+j];
                }
            }
        }
        __syncthreads();
    }
    for(int i=0;i<dpt;i++){
        for(int j=0;j<dpt;j+=4){
            toFloat(C[(threadIdx.y*dpt+i)*N+threadIdx.x*dpt+j])=toFloat(ans[i][j]);
        }
    }
}
template<const unsigned int BLOCKSIZE,const unsigned int DPT>
void dmm(float*A,float*B,float*C,const unsigned int M,const unsigned int K,const unsigned int N){
    const unsigned int DATASIZE=BLOCKSIZE*DPT;
    dim3 gd={(N+DATASIZE-1)/DATASIZE,(M+DATASIZE-1)/DATASIZE,1};
    dim3 bd={BLOCKSIZE,BLOCKSIZE,1};
    kmm<BLOCKSIZE,DPT><<<gd,bd>>>(A,B,C,M,K,N);
    cudaDeviceSynchronize();
}
void hmm(float*A,float*B,float*C,const unsigned int M,const unsigned int K,const unsigned int N){
    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
            C[i*N+j]=0;
            for(int k=0;k<K;k++){
                C[i*N+j]+=A[i*K+k]*B[k*N+j];
            }
        }
    }
}
void init(float*A,int len){
    for(int i=0;i<len;i++){
        A[i]=rand()%1000;
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
    const unsigned int bs=8,dpt=4;
    int M=256,K=256,N=256;
    if(argc>1){
        M=std::stoi(argv[1]);
        if(argc>2){
            K=std::stoi(argv[2]);
            if(argc>3){
                N=std::stoi(argv[3]);
            }
        }
    }
    float*hA,*hB,*hC;
    hA=(float*)malloc(sizeof(float)*M*K);
    hB=(float*)malloc(sizeof(float)*N*K);
    hC=(float*)malloc(sizeof(float)*M*N);
    init(hA,M*K);
    init(hB,N*K);
    hmm(hA,hB,hC,M,K,N);

    float*dA,*dB,*dC,*hC1;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&dA,sizeof(float)*M*K));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&dB,sizeof(float)*N*K));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&dC,sizeof(float)*M*N));
    hC1=(float*)malloc(sizeof(float)*M*N);
    cudaMemcpy(dA,hA,sizeof(float)*M*K,cudaMemcpyHostToDevice);
    cudaMemcpy(dB,hB,sizeof(float)*K*N,cudaMemcpyHostToDevice);
    dmm<bs,dpt>(dA,dB,dC,M,K,N);
    cudaMemcpy(hC1,dC,sizeof(float)*M*N,cudaMemcpyDeviceToHost);
    check(hC,hC1,M*N);
    free(hA);
    free(hB);
    free(hC);
    free(hC1);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return 0;
}
   