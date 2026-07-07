```
AMD->ROCm, Intel->Intel oneAPI 都被pytorch支持。
ROCm 类似于 NVIDIA 的 CUDA 平台（包含编译器、库、工具等整个生态）。
HIP是ROCm平台中最核心的编程模型和工具，用于将 CUDA 代码转换为可在 AMD GPU 上运行的代码

pytorch支持x86(amd,intel)，arm(apple，高通，联发科)的cpu

http://blog.ezyang.com/2019/05/pytorch-internals/
https://jimmy-shen.medium.com/how-to-understand-pytorch-source-code-1fdbdbbf007e
https://www.bilibili.com/video/BV1L6CgYBEBs/?spm_id_from=333.337.search-card.all.click&vd_source=7c406f161c56ca0055b900d0df22d9bc
Caffe2的GitHub仓库直接并入PyTorch主库
c10(caffe2):tensor and storage struct
c10::TensorImpl{
    c10::Storage{
        data_ptr;
        allocator;
    };
    data_type;
}
aten(a tensor library):tensor operator
at::Tensor{
    c10::TensorImpl;
    autogradMeta{
        at::Tensor grad_;
    };
};
torch._C.TensorBase-pybind11-at::Tensor
torch::Tensor==at::Tensor
PyTorch 的核心是一个 dispatcher，它的功能是根据输入 tensor 的属性把算子 dispatch 到具体的 kernel 上（基于张量的设备类型和布局和dtype）

https://zdevito.github.io/2022/08/04/cuda-caching-allocator.html
struct Block {
    int device; // gpu
    cudaStream_t stream; // allocation stream
    size_t size; // block size in bytes
    BlockPool* pool{nullptr}; // owning memory pool
    void* ptr{nullptr}; // memory address
    bool allocated{false}; // in-use flag
    Block* prev{nullptr}; // prev block if split from a larger allocation
    Block* next{nullptr}; // next block if split from a larger allocation
    // (...)
}
Blocks are allocated per CUDA stream. Each CUDA stream will have its own cache of allocations, with rebalancing done only by emptying all the caches in low-memory conditions.
PyTorch按需实时向GPU申请显存，通过cudaMalloc申请大块显存（Segment），切分为子块（Block）供上层使用，已释放的Block保留在缓存池中，通过合并相邻空闲Block减少碎片
对于 < 1MB 的“小”张量，我们使用单独的池来避免对较大的池进行碎片化。链表、红黑树
Rounding happens twice. First on the size being requested, and again if we are requesting a (possibly bigger) block from CUDA. 
when we discover that CUDA is out of memory (an OOM). In this situation we will try to free up cached but unused memory.
避免频繁调用cudaMalloc
缓存可能占用大量显存，影响其他进程或导致OOM

torch/csrc:
    autograd
    jit
    distributed

The autograd system is moved into C now and is multi-threaded
https://www.bilibili.com/video/BV1Xr4y1r7wa?spm_id_from=333.788.videopod.sections&vd_source=7c406f161c56ca0055b900d0df22d9bc
单个算子的导数==雅可比矩阵dl/dy=[1,y]
链式求导==雅可比矩阵乘法dl/dy*dy/dx
前向求导或反向求导的计算量==多个矩阵相乘的复杂度
[1,m][m,m][m,m][m,m]: m^3*2+m^2, m^2*3
https://www.bilibili.com/video/BV1by4y1B7LD/?spm_id_from=333.337.search-card.all.click&vd_source=7c406f161c56ca0055b900d0df22d9bc
For Tensors that have requires_grad which is True, they will be leaf Tensors if they were created by the user. This means that they are not the result of an operation and so grad_fn is None.
在反向传播过程中，只有叶子节点的梯度值会被自动保留并存储在 .grad 属性中(AccumulateGrad)
next_functions=[创建前向算子的输入的后向算子或AccumulateGrad,...]

dataset:加载，预处理
dataloader:batch预处理

https://github.com/hao-ai-lab/MuxServe/blob/main/muxserve/flexstore/manager.py#L893
tensor.untyped_storage()._share_cuda_()
序列化后写入共享内存
https://github.com/pytorch/pytorch/blob/main/torch/multiprocessing/reductions.py#L155
rebuild_cuda_tensor()
```

```
docker pull nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
docker run -itd --ipc host --net host --gpus all --name cyy_torch -v ~/torch:/torch nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 bash
apt install -y python3.11
apt install -y python3.11-dev
git clone https://github.com/pytorch/pytorch
cd pytorch
git checkout v.2.10.0
git submodule sync
git submodule update --init --recursive
DEBUG=1 MAX_JOBS=16 nohup python -m pip install --no-build-isolation -v -e . > cyyinstall.log 2>&1 &
python setup.py clean
```

# setup.py

```
def build_deps() -> None:
    build_pytorch(
```

```
def build_pytorch(
    # 步骤 4：核心！生成 CMake 编译规则（解析 CMakeLists.txt，生成 Makefile/ninja 脚本）
    cmake.generate(
        version, cmake_python_library, build_python, build_test, my_env, rerun_cmake
    )
    # 步骤 7：核心！执行 CMake 编译，生成 libtorch_python.so 等库
    cmake.build(my_env)
```

```
def configure_extension_build() -> tuple[
    main_libraries: list[str] = ["torch_python"]#_C的依赖库，需要事先产生
    main_sources: list[str] = ["torch/csrc/stub.c"]
    packages = find_packages(include=includes, exclude=excludes)
    C = Extension(
        "torch._C",
        libraries=main_libraries,
        sources=main_sources,
    ext_modules.append(C)
    cmdclass = {
        "bdist_wheel": bdist_wheel,
        "build_ext": build_ext,
        "clean": clean,
        "sdist": sdist,
    }
```

```
def main() -> None:
    if RUN_BUILD_DEPS:
        build_deps()
        mirror_inductor_external_kernels()
    ) = configure_extension_build()
    setup(
        name=TORCH_PACKAGE_NAME,
        version=TORCH_VERSION,
        ext_modules=ext_modules,
        cmdclass=cmdclass,
        packages=packages,
```

# __init__.py

```
    if USE_GLOBAL_DEPS:
        _load_global_deps()
    from torch._C import *
```

```
def _load_global_deps() -> None:
        ctypes.CDLL(global_deps_lib_path, mode=ctypes.RTLD_GLOBAL)
            with open("/proc/self/maps") as f:
                _maps = f.read()

            # libtorch_global_deps.so always depends in cudart, check if its installed and loaded
            if "libcudart.so" not in _maps:
                return
            _preload_cuda_deps()
def _preload_cuda_deps(err: OSError | None = None) -> None:
    cuda_libs: list[tuple[str, str]] = [
        ("cublas", "libcublas.so.*[0-9]"),
    for lib_folder, lib_name in cuda_libs:
        _preload_cuda_lib(lib_folder, lib_name)
def _preload_cuda_lib(lib_folder: str, lib_name: str, required: bool = True) -> None:  # type: ignore[valid-type]
    lib_path = None
    for path in sys.path:
        candidate_lib_paths = _get_cuda_dep_paths(path, lib_folder, lib_name)
        if candidate_lib_paths:
            lib_path = candidate_lib_paths[0]
            break
    if not lib_path and required:
        raise ValueError(f"{lib_name} not found in the system path {sys.path}")
    if lib_path:
        ctypes.CDLL(lib_path)
```

# csrc

## autograd

### python_torch_functions_manual.cpp

```
void initTorchFunctions(PyObject* module) {
  if (PyModule_AddObject(
          module,
          "_VariableFunctionsClass",
  THPVariableFunctionsModule =
      PyType_GenericNew(&THPVariableFunctions, Py_None, Py_None);
  if (PyModule_AddObject(
          module, "_VariableFunctions",
```

### python_variable.cpp

```
static PyTypeObject THPVariableType = {
    THPVariable_pynew, /* tp_new */
};
PyObject* THPVariable_pynew(
  PyObject* obj = THPVariable_WrapWithType(
      torch::utils::base_tensor_ctor(args, kwargs), type);
static PyObject* THPVariable_WrapWithType(
  obj = type->tp_alloc(type, 0);
  auto v = reinterpret_cast<THPVariable*>(obj);
  new (&v->cdata) Tensor(std::forward<T>(var));
```

```
bool THPVariable_initModule(PyObject* module) {
  PyModule_AddObject(module, "_TensorMeta", (PyObject*)&THPVariableMetaType);
  PyModule_AddObject(module, "TensorBase", (PyObject*)&THPVariableType);
  PyModule_AddObject(module, "_TensorBase", (PyObject*)&THPVariableType);
  torch::autograd::initTorchFunctions(module);
```

### python_variable.h

```
struct THPVariable {
  PyObject_HEAD
  // Payload
  at::Tensor cdata;
```

## Module.cpp

```
static std::initializer_list<PyMethodDef> TorchMethods = {
static std::vector<PyMethodDef> methods;
PyObject* initModule() {
  THPUtils_addPyMethodDefs(methods, std::data(TorchMethods));
  //把函数列表里的所有函数加入methods
  static struct PyModuleDef torchmodule = {
      PyModuleDef_HEAD_INIT, "torch._C", nullptr, -1, methods.data()};
  module = PyModule_Create(&torchmodule);
  ASSERT_TRUE(THPVariable_initModule(module));
  ASSERT_TRUE(THPFunction_initModule(module));
```

## stub.c

```
PyMODINIT_FUNC PyInit__C(void)
{
  return initModule();
}
```

# autograd

function.py

```
class _HookMixin:
    @staticmethod
    def _register_hook(backward_hooks, hook):
        if backward_hooks is None:
            backward_hooks = OrderedDict()
        handle = hooks.RemovableHandle(backward_hooks)
        backward_hooks[handle.id] = hook
        return backward_hooks, handle
```

```
flash_attn_qkvpacked_func = FlashAttnQKVPackedFunc.apply

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
```

# distributed

## run

node==机器，worker==进程(GPU)，job==模型实例

rdzv_backend - The backend of the rendezvous (e.g. c10d, etcd). This is typically a strongly consistent key-value store. 存储集群配置信息

C10d Uses a C10d store (by default, TCPStore) as the backend.

TCPStore/etcd：仅支持小数据量 KV（KB 级），比如存 “master_port=6000”“total_workers=8” 这类配置信息；
Redis：支持大容量 KV（MB/GB 级），可存字符串、哈希、甚至二进制数据，适合缓存用户会话、商品信息等业务数据。

```
        store = torch.distributed.TCPStore(
            host_name=os.environ['MASTER_ADDR'],
            port=int(os.environ['MASTER_PORT'])+1,
            world_size=int(os.getenv('WORLD_SIZE', '1')),
            is_master=(int(os.getenv('RANK', '0')) == 0),
            timeout=timedelta(seconds=300),
            wait_for_workers=True,
            use_libuv=True,
        )
```

comm backend：gloo，mpi，nccl，xccl，ucc. 数据传输

```
torchrun will pass the --local-rank=<rank> argument to your script.
```

# library

```
    _batch_invariant_LIB = torch.library.Library("aten", "IMPL")
    _batch_invariant_LIB.impl("aten::mm", mm_batch_invariant, dispatch_key)
为 aten 命名空间下已存在的算子（如 aten::relu）替换 / 添加自定义实现；
```

# nn

## module

```
class Module:
    __call__: Callable[..., Any] = _wrapped_call_impl
    def _wrapped_call_impl(self, *args, **kwargs):
        if self._compiled_call_impl is not None:
            return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
        else:
            return self._call_impl(*args, **kwargs)

    def _call_impl(self, *args, **kwargs):
        forward_call = (self._slow_forward if torch._C._get_tracing_state() else self.forward)
        if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
                or _global_backward_pre_hooks or _global_backward_hooks
                or _global_forward_hooks or _global_forward_pre_hooks):
            return forward_call(*args, **kwargs)

    def __setattr__(self, name: str, value: Union[Tensor, "Module"]) -> None:
        params = self.__dict__.get("_parameters")
        if isinstance(value, Parameter):
            remove_from(
                self.__dict__,
                self._buffers,
                self._modules,
                self._non_persistent_buffers_set,
            )
            self.register_parameter(name, value)
        else:
            modules = self.__dict__.get("_modules")
            if isinstance(value, Module):
                modules[name] = value
            else:
                buffers = self.__dict__.get("_buffers")
            
    def __getattr__(self, name: str) -> Union[Tensor, "Module"]:
        if "_parameters" in self.__dict__:
            _parameters = self.__dict__["_parameters"]
            if name in _parameters:
                return _parameters[name]
        if "_buffers" in self.__dict__:
            _buffers = self.__dict__["_buffers"]
            if name in _buffers:
                return _buffers[name]
        if "_modules" in self.__dict__:
            modules = self.__dict__["_modules"]
            if name in modules:
                return modules[name]

	#深搜
    def named_modules(
            yield prefix, self
            for name, module in self._modules.items():
                if module is None:
                    continue
                submodule_prefix = prefix + ("." if prefix else "") + name
                yield from module.named_modules(
                    memo, submodule_prefix, remove_duplicate
                )

    def modules(self) -> Iterator["Module"]:
        for _, module in self.named_modules():
            yield module
            
    def named_parameters(
        gen = self._named_members(
            lambda module: module._parameters.items(),
        yield from gen

    def _named_members(
        modules = (
            self.named_modules(prefix=prefix, remove_duplicate=remove_duplicate)
            if recurse
            else [(prefix, self)]
        )
        for module_prefix, module in modules:
            members = get_members_fn(module)
            for k, v in members:
                if v is None or v in memo:
                    continue
                if remove_duplicate:
                    memo.add(v)
                name = module_prefix + ("." if module_prefix else "") + k
                yield name, v
     
    def named_children(self) -> Iterator[tuple[str, "Module"]]:
        for name, module in self._modules.items():
                yield name, module
	
	#作用到每个module
    def apply(self, fn: Callable[["Module"], None]) -> Self:
        for module in self.children():
            module.apply(fn)
        fn(self)
	
	#作用到每个tensor
    def _apply(self, fn, recurse=True):
        if recurse:
            for module in self.children():
                module._apply(fn)
        for key, param in self._parameters.items():
            with torch.no_grad():
                param_applied = fn(param)
            param_grad = param.grad
            if param_grad is not None:
                with torch.no_grad():
                    grad_applied = fn(param_grad)
        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)
                
    def register_forward_pre_hook(
        handle = RemovableHandle(
            self._forward_pre_hooks, extra_dict=self._forward_pre_hooks_with_kwargs
        )
        self._forward_pre_hooks[handle.id] = hook
```

## Parameter

```
(Pdb) p type(param)
<class 'torch.nn.parameter.Parameter'>
(Pdb) p type(param.data)
<class 'torch.Tensor'>
```

# Tensor

```
next_functions = [
    (grad_fn_node, input_index),  
    ...
]
第一个元素（grad_fn_node）：真正的「梯度函数节点」（比如梯度累加器、上一个操作的 grad_fn）；
第二个元素（input_index）：该节点在父操作中的输入索引（仅多输入操作有用，单输入操作下为 0）。
```

view: Returns a new tensor with the same data as the self tensor but of a different shape.

detach: Returned Tensor shares the same storage with the original one.

与普通的 Python 列表不同，torch.zeros 在调用时会立即在内存（RAM）或显存（VRAM）中开辟连续的物理空间，并将所有位（bits）初始化为 0。

# utils

hooks.py

```
class RemovableHandle:
    next_id: int = 0

    def __init__(self, hooks_dict: Any, *, extra_dict: Any = None) -> None:
        self.hooks_dict_ref = weakref.ref(hooks_dict)
        self.id = RemovableHandle.next_id
        RemovableHandle.next_id += 1
    def remove(self) -> None:
        hooks_dict = self.hooks_dict_ref()
        if hooks_dict is not None and self.id in hooks_dict:
            del hooks_dict[self.id]
```

