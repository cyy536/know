#include<math.h>
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
void hLayernorm(const float* inp, float* out, const float* weight, const float* bias, int C, int B) {
    float eps = 1e-5f;
    for (int b = 0; b < B; b++) {
        const float* x = inp + b * C ;
        // calculate the mean
        float m = 0.0f;
        for (int i = 0; i < C; i++) {
            m += x[i];
        }
        m = m/C;
        // calculate the variance (without any bias correction)
        float v = 0.0f;
        for (int i = 0; i < C; i++) {
            float xshift = x[i] - m;
            v += xshift * xshift;
        }
        v = v/C;
        // calculate the rstd
        float s = 1.0f / sqrtf(v + eps);
        float* out_bt = out + b * C ;
        for (int i = 0; i < C; i++) {
            float n = (s * (x[i] - m)); 
            float o = n * weight[i] + bias[i];
            out_bt[i] = o; 
        }
    }
}
template<const unsigned int blockSize>
__global__ void kLayernorm(const float* inp, float* out, const float* weight, const float* bias, int len) {
    float eps = 1e-5f;
    const float* A = inp + blockIdx.x * len;
    int dataPerThread=(len+blockSize-1)/blockSize;
    float threadSum=0;
    for(int i=0;i<dataPerThread;i++){
        if(threadIdx.x+blockSize*i<len){
            threadSum+=A[threadIdx.x+blockSize*i];
        }
    }
    __shared__ float sm[blockSize];
    sm[threadIdx.x]=threadSum;
    int step=blockSize/2;
    while(step){
        __syncthreads();
        if(threadIdx.x<step&&threadIdx.x+step<len){
            sm[threadIdx.x]+=sm[threadIdx.x+step];
        }
        step/=2;
    }
    __syncthreads();
    float mean=sm[0]/len;

    threadSum=0;
    for(int i=0;i<dataPerThread;i++){
        if(threadIdx.x+blockSize*i<len){
            threadSum+=(A[threadIdx.x+blockSize*i]-mean)*(A[threadIdx.x+blockSize*i]-mean);
        }
    }
    sm[threadIdx.x]=threadSum;
    step=blockSize/2;
    while(step){
        __syncthreads();
        if(threadIdx.x<step&&threadIdx.x+step<len){
            sm[threadIdx.x]+=sm[threadIdx.x+step];
        }
        step/=2;
    }
    __syncthreads();
    float scale=1/sqrtf(sm[0]/len+eps);

    float* out_bt = out + blockIdx.x * len;
    for(int i=0;i<dataPerThread;i++){
        int tmpIdx=threadIdx.x+blockSize*i;
        if(tmpIdx<len){
            out_bt[tmpIdx]=(scale*(A[tmpIdx]-mean))*weight[tmpIdx]+bias[tmpIdx];
        }
    }
}
template<const unsigned int BLOCKSIZE>
void dLayernorm(float*A,float*B,float*weight,float*bias,const unsigned int len,const unsigned int numBatch){
    dim3 gd={numBatch,1,1};
    dim3 bd={BLOCKSIZE,1,1};
    kLayernorm<BLOCKSIZE><<<gd,bd>>>(A,B,weight,bias,len);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}
void init(float*hA,int len){
    for(int i=0;i<len;i++){
        hA[i]=rand()%1000;
    }
}
void check(float*A,float*B,int len){
    bool right=true;
    for(int i=0;i<len;i++){
        // if(std::abs(A[i]-B[i])>1e-6){
        if(std::abs(A[i]-B[i])>1){
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
    float*hA,*hB,*hweight,*hbias;
    hA=(float*)malloc(sizeof(float)*len*numBatch);
    hB=(float*)malloc(sizeof(float)*len*numBatch);
    hweight=(float*)malloc(sizeof(float)*len);
    hbias=(float*)malloc(sizeof(float)*len);
    init(hA,len*numBatch);
    init(hweight,len);
    init(hbias,len);
    hLayernorm(hA,hB,hweight,hbias,len,numBatch);

    float*dA,*dB,*hB1,*dweight,*dbias;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&dA,sizeof(float)*len*numBatch));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&dB,sizeof(float)*len*numBatch));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&dweight,sizeof(float)*len));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&dbias,sizeof(float)*len));
    hB1=(float*)malloc(sizeof(float)*len*numBatch);
    CHECK_CUDA_ERROR(cudaMemcpy(dA,hA,sizeof(float)*len*numBatch,cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(dweight,hweight,sizeof(float)*len,cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(dbias,hbias,sizeof(float)*len,cudaMemcpyHostToDevice));
    dLayernorm<512>(dA,dB,dweight,dbias,len,numBatch);
    CHECK_CUDA_ERROR(cudaMemcpy(hB1,dB,sizeof(float)*len*numBatch,cudaMemcpyDeviceToHost));

    check(hB,hB1,len*numBatch);
    free(hA);
    free(hB);
    free(hB1);
    free(hweight);
    free(hbias);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dweight);
    cudaFree(dbias);
    return 0;
}