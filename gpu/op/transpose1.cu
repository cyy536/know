#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <string>
#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}
template<const unsigned int blockSize>
__global__ void kTranspose(float* input, float* output, int M, int N){
    int x=threadIdx.x+blockDim.x*blockIdx.x;
    int y=threadIdx.y+blockDim.y*blockIdx.y;
    if(y<N&&x<M){
        output[y*M+x]=input[x*N+y];
    }
}
template<const unsigned int blockSize>
__global__ void kTranspose1(float* input, float* output, int M, int N){
    int x=threadIdx.x+blockDim.x*blockIdx.x;
    int y=threadIdx.y+blockDim.y*blockIdx.y;
    __shared__ float sm[blockSize][blockSize+1];
    if(y<M&&x<N){
        sm[threadIdx.y][threadIdx.x]=input[y*N+x];
    }
    __syncthreads();
    y=threadIdx.y+blockDim.x*blockIdx.x;
    x=threadIdx.x+blockDim.y*blockIdx.y;
    if(y<N&&x<M){
        output[y*M+x]=sm[threadIdx.x][threadIdx.y];
    }
}
void host_transpose(float* input, int M, int N, float* output) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            output[i * M + j] = input[j * N + i];
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
int main(int argc,char*argv[]) {
    unsigned int M = 128;
    unsigned int N = 256;
    if(argc>1){
        M=std::stoi(argv[1]);
        if(argc>2){
            N=std::stoi(argv[2]);
        }
    }
    constexpr unsigned int BLOCK_SIZE = 32;

    float *h_matrix = (float *)malloc(sizeof(float) * M * N);
    float *h_matrix_tr_ref = (float *)malloc(sizeof(float) * N * M);
    init(h_matrix, M * N);
    host_transpose(h_matrix, M, N, h_matrix_tr_ref);


    float *d_matrix;
    cudaMalloc((void **) &d_matrix, sizeof(float) * M * N);
    cudaMemcpy(d_matrix, h_matrix, sizeof(float) * M * N, cudaMemcpyHostToDevice);

    float *d_output0;
    cudaMalloc((void **) &d_output0, sizeof(float) * N * M);                           
    float *h_output0 = (float *)malloc(sizeof(float) * N * M);                    

    dim3 block_size0={BLOCK_SIZE,BLOCK_SIZE,1};
    // dim3 grid_size0={(M+BLOCK_SIZE-1)/BLOCK_SIZE,(N+BLOCK_SIZE-1)/BLOCK_SIZE,1};                           
    // kTranspose<BLOCK_SIZE><<<grid_size0, block_size0>>>(d_matrix, d_output0, M, N);
    dim3 grid_size0={(N+BLOCK_SIZE-1)/BLOCK_SIZE,(M+BLOCK_SIZE-1)/BLOCK_SIZE,1};                           
    kTranspose1<BLOCK_SIZE><<<grid_size0, block_size0>>>(d_matrix, d_output0, M, N);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    cudaMemcpy(h_output0, d_output0, sizeof(float) * N * M, cudaMemcpyDeviceToHost);

    check(h_output0, h_matrix_tr_ref, M * N);                                    

    free(h_matrix);
    free(h_matrix_tr_ref);
    cudaFree(d_output0);
    free(h_output0);

    return 0;
}