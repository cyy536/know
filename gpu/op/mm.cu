#include<cuda.h>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<sys/time.h>
#define DBD 64
#define STRIDE 4
#define TO_FLOAT4(x) (*((float4*)(&(x))))

double getTime(){
    struct timeval tv;
    gettimeofday(&tv,NULL);
    return (double)(tv.tv_sec*1e3+tv.tv_usec/1e3);
}

void init(float*hA,int M,int K){
    for(int i=0;i<M;i++){
        for(int j=0;j<K;j++){
            hA[i*K+j]=rand()%1000;
        }
    }
}

void hMm(float*hA,float*hB,float*hC,int M,int K,int N){
    for(int m=0;m<M;m++){
        for(int n=0;n<N;n++){
            float tmp=0;
            for(int k=0;k<K;k++){
                tmp+=hA[m*K+k]*hB[k*N+n];
            }
            hC[m*N+n]=tmp;
        }
    }
}

void check(float*hC1,float*hC2,int M,int N){
    for(int m=0;m<M;m++){
        for(int n=0;n<N;n++){
            if(abs(hC1[m*N+n]-hC2[m*N+n])>1e-6){
                printf("error:%f %f\n",hC1[m*N+n],hC2[m*N+n]);
            }
        }
    }
}

__global__ void mmK(float*dA,float*dB,float*dC,unsigned int M,unsigned int K,unsigned int N){
    int m=blockDim.y*blockIdx.y+threadIdx.y;
    int n=blockDim.x*blockIdx.x+threadIdx.x;
    float tmp=0;
    for(int k=0;k<K;k++){
        tmp+=dA[m*K+k]*dB[k*N+n];
    }
    dC[m*N+n]=tmp;
}

__global__ void mmK1(float*dA,float*dB,float*dC,unsigned int M,unsigned int K,unsigned int N){
    int m=blockDim.y*blockIdx.y+threadIdx.y;
    int n=blockDim.x*blockIdx.x+threadIdx.x;
    __shared__ float sA[DBD][DBD];
    __shared__ float sB[DBD][DBD];
    float tmp=0;
    for(int i=0;i<K/DBD;i++){
        sA[threadIdx.y][threadIdx.x]=dA[m*K+i*DBD+threadIdx.x];
        sB[threadIdx.y][threadIdx.x]=dB[(i*DBD+threadIdx.y)*N+n];
        __syncthreads();
        for(int k=0;k<DBD;k++){
            tmp+=sA[threadIdx.y][k]*sB[k][threadIdx.x];
        }
        __syncthreads();
    }
    dC[m*N+n]=tmp;
}

__global__ void mmK2(float*dA,float*dB,float*dC,unsigned int M,unsigned int K,unsigned int N){
    int dmb=(blockDim.y*blockIdx.y+threadIdx.y)*STRIDE;
    int dnb=(blockDim.x*blockIdx.x+threadIdx.x)*STRIDE;
    __shared__ float sA[DBD][DBD];
    __shared__ float sB[DBD][DBD];
    float tmp[STRIDE][STRIDE];
    for(int i=0;i<STRIDE;i++){
        for(int j=0;j<STRIDE;j++){
            tmp[i][j]=0;
        }
    }
    for(int i=0;i<K/DBD;i++){
        for(int o=0;o<STRIDE;o++){
            for(int p=0;p<STRIDE;p++){
                sA[threadIdx.y*STRIDE+o][threadIdx.x*STRIDE+p]=dA[(dmb+o)*K+i*DBD+threadIdx.x*STRIDE+p];
                sB[threadIdx.y*STRIDE+o][threadIdx.x*STRIDE+p]=dB[(i*DBD+threadIdx.y*STRIDE+o)*N+dnb+p];
            }
        }
        __syncthreads();
        for(int o=0;o<STRIDE;o++){
            for(int p=0;p<STRIDE;p++){
                for(int k=0;k<DBD;k++){
                    tmp[o][p]+=sA[threadIdx.y*STRIDE+o][k]*sB[k][threadIdx.x*STRIDE+p];
                }
            }
        }
        __syncthreads();
    }
    for(int o=0;o<STRIDE;o++){
        for(int p=0;p<STRIDE;p++){
            dC[(dmb+o)*N+dnb+p]=tmp[o][p];
        }
    }
}

__global__ void mmK3(float*dA,float*dB,float*dC,unsigned int M,unsigned int K,unsigned int N){
    int dmb=(blockDim.y*blockIdx.y+threadIdx.y)*STRIDE;
    int dnb=(blockDim.x*blockIdx.x+threadIdx.x)*STRIDE;
    __shared__ float sA[DBD][DBD];
    __shared__ float sB[DBD][DBD];
    float tmp[STRIDE][STRIDE];
    for(int i=0;i<STRIDE;i++){
        for(int j=0;j<STRIDE;j++){
            tmp[i][j]=0;
        }
    }
    for(int i=0;i<K/DBD;i++){
        for(int o=0;o<STRIDE;o++){
            for(int p=0;p<STRIDE/4;p+=4){
                TO_FLOAT4(sA[threadIdx.y*STRIDE+o][threadIdx.x*STRIDE+p])=TO_FLOAT4(dA[(dmb+o)*K+i*DBD+threadIdx.x*STRIDE+p]);
                TO_FLOAT4(sB[threadIdx.y*STRIDE+o][threadIdx.x*STRIDE+p])=TO_FLOAT4(dB[(i*DBD+threadIdx.y*STRIDE+o)*N+dnb+p]);
            }
        }
        __syncthreads();
        for(int o=0;o<STRIDE;o++){
            for(int p=0;p<STRIDE;p++){
                for(int k=0;k<DBD;k++){
                    tmp[o][p]+=sA[threadIdx.y*STRIDE+o][k]*sB[k][threadIdx.x*STRIDE+p];
                }
            }
        }
        __syncthreads();
    }
    for(int o=0;o<STRIDE;o++){
        for(int p=0;p<STRIDE/4;p+=4){
            TO_FLOAT4(dC[(dmb+o)*N+dnb+p])=TO_FLOAT4(tmp[o][p]);
        }
    }
}

__global__ void mmK4(float*dA,float*dB,float*dC,unsigned int M,unsigned int K,unsigned int N){
    int dmb=(blockDim.y*blockIdx.y+threadIdx.y)*STRIDE;
    int dnb=(blockDim.x*blockIdx.x+threadIdx.x)*STRIDE;
    __shared__ float sA[DBD][DBD];
    __shared__ float sB[DBD][DBD];
    float tmp[STRIDE][STRIDE];
    float ar[STRIDE],br[STRIDE];//[STRIDE][1],[1][STRIDE]
    for(int i=0;i<STRIDE;i++){
        for(int j=0;j<STRIDE;j++){
            tmp[i][j]=0;
        }
    }
    for(int i=0;i<K/DBD;i++){
        for(int o=0;o<STRIDE;o++){
            for(int p=0;p<STRIDE/4;p+=4){
                TO_FLOAT4(sA[threadIdx.y*STRIDE+o][threadIdx.x*STRIDE+p])=TO_FLOAT4(dA[(dmb+o)*K+i*DBD+threadIdx.x*STRIDE+p]);
                TO_FLOAT4(sB[threadIdx.y*STRIDE+o][threadIdx.x*STRIDE+p])=TO_FLOAT4(dB[(i*DBD+threadIdx.y*STRIDE+o)*N+dnb+p]);
            }
        }
        __syncthreads();
        for(int j=0;j<DBD;j++){
            for(int o=0;o<STRIDE;o++){
                ar[o]=sA[threadIdx.y*STRIDE+o][j];    
            }
            for(int p=0;p<STRIDE/4;p+=4){
                TO_FLOAT4(br[p])=TO_FLOAT4(sB[j][threadIdx.x*STRIDE+p]);
            }
            for(int o=0;o<STRIDE;o++){
                for(int p=0;p<STRIDE;p++){
                    tmp[o][p]+=ar[o]*br[p];    
                }
            }
        }
        __syncthreads();
    }
    for(int o=0;o<STRIDE;o++){
        for(int p=0;p<STRIDE/4;p+=4){
            TO_FLOAT4(dC[(dmb+o)*N+dnb+p])=TO_FLOAT4(tmp[o][p]);
        }
    }
}

void dMm(float*dA,float*dB,float*dC,unsigned int M,unsigned int K,unsigned int N){
    dim3 grid_d={(N+DBD-1)/(DBD),(M+DBD-1)/(DBD),1};
    dim3 block_d={DBD/STRIDE,DBD/STRIDE,1};

    cudaEvent_t start,end;
    float dur;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    mmK4<<<grid_d,block_d>>>(dA,dB,dC,M,K,N);
    cudaEventRecord(end);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&dur,start,end);

    printf("kernel:%f\n",dur);
}

int main(int argc, char*argv[]){
    int M=DBD*2,K=M,N=DBD*2;
    if(argc>1&&M<atoi(argv[1])){
        M=K=N=atoi(argv[1]);
    }
    printf("%d %d %d\n",M,K,N);

    float*hA=(float*)malloc(sizeof(float)*M*K);
    float*hB=(float*)malloc(sizeof(float)*K*N);
    float*hC=(float*)malloc(sizeof(float)*M*N);

    init(hA,M,K);
    init(hB,K,N);

    double start,end;
    start=getTime();
    hMm(hA,hB,hC,M,K,N);
    end=getTime();
    printf("host:%lf\n",end-start);

    start=getTime();
    float*dA,*dB,*dC;
    cudaMalloc((void**)&dA,sizeof(float)*M*K);
    cudaMalloc((void**)&dB,sizeof(float)*K*N);
    cudaMalloc((void**)&dC,sizeof(float)*M*N);

    cudaMemcpy(dA,hA,sizeof(float)*M*K,cudaMemcpyHostToDevice);
    cudaMemcpy(dB,hB,sizeof(float)*K*N,cudaMemcpyHostToDevice);

    dMm(dA,dB,dC,M,K,N);

    float*hC1=(float*)malloc(sizeof(float)*M*N);
    cudaMemcpy(hC1,dC,sizeof(float)*M*N,cudaMemcpyDeviceToHost);

    end=getTime();
    printf("device:%lf\n",end-start);

    check(hC,hC1,M,N);

    free(hA);
    free(hB);
    free(hC);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    free(hC1);
    return 0;
}