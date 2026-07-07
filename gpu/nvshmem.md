## NVSHMEM

```
NVSHMEM并行编程接口，将多个GPU的内存抽象为统一的全局地址空间，允许GPU线程直接通过单边通信原语（如put、get、atomic）访问远程GPU内存。
有源码
DeepEP 把 魔改过的 NVSHMEM 的 GPU 单边通信封装成 dispatch/combine 语义
NCCL的API只能在主机端调用
```