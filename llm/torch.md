Caffe2的GitHub仓库直接并入PyTorch主库

c10(caffe2):tensor and storage struct

aten(a tensor library):tensor operator

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

# build

3090, 570.195.03, 12.8

至少30GiB内存

```
git clone https://github.com/pytorch/pytorch.git
cd pytorch
git checkout v2.13.0
git submodule sync
git submodule update --init --recursive
DEBUG=1 MAX_JOBS=1 nohup python -m pip install --no-build-isolation -v -e . > cyyinstall.log 2>&1 &
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

# aten/src/ATen

op 会 dispatch 多次，嵌套 autograd，TorchDispatchMode，cuda

## core
### boxing
#### BoxedKernel_impl.h
torch.matmul->Tensor::matmul->matmul::call->dispatcher->KernelFunction->BoxedKernelWrapper->BoxedKernel

torch.matmul->Tensor::matmul->matmul::call->dispatcher->KernelFunction->BoxedKernelWrapper->BoxedKernel->PythonTLSSnapshot->redispatchBoxed->BoxedKernel

```
inline BoxedKernel::BoxedKernel(
    std::unique_ptr<OperatorKernel> functor,
    InternalBoxedKernelFunction* boxed_kernel_func)
    : functor_(std::move(functor)), boxed_kernel_func_(boxed_kernel_func) {}

inline void BoxedKernel::callBoxed(
    const OperatorHandle& opHandle,
    DispatchKeySet dispatchKeySet,
    Stack* stack) const {
  (*boxed_kernel_func_)(functor_.get(), opHandle, dispatchKeySet, stack);
```
#### impl
##### boxing.h
torch.matmul->Tensor::matmul->matmul::call->dispatcher->KernelFunction->BoxedKernelWrapper
```
struct BoxedKernelWrapper<
    Result(Args...),
    std::enable_if_t<
        can_box_all<Args...>::value && can_unbox<Result>::value &&
            !is_tuple_of_mutable_tensor_refs<Result>::value,
        void>> {
  static Result call(
      const BoxedKernel& boxed_kernel_func,
      const OperatorHandle& opHandle,
      DispatchKeySet dispatchKeySet,
      Args... args) {
    torch::jit::Stack stack = boxArgs<Args...>(std::forward<Args>(args)...);
    boxed_kernel_func.callBoxed(opHandle, dispatchKeySet, &stack);


```

##### make_boxed_from_unboxed_functor.h
torch.matmul->Tensor::matmul->matmul::call->dispatcher->KernelFunction->BoxedKernelWrapper->BoxedKernel->PythonTLSSnapshot->redispatchBoxed->BoxedKernel->unboxed
```
template <class KernelFunctor, bool AllowDeprecatedTypes>
struct make_boxed_from_unboxed_functor final {
  static void call(
      OperatorKernel* functor,
      const OperatorHandle& /*unused*/,
      DispatchKeySet dispatchKeySet,
      Stack* stack) {
      ReturnType_ output = call_functor_with_args_from_stack<
          KernelFunctor,
          AllowDeprecatedTypes>(functor, dispatchKeySet, stack);

call_functor_with_args_from_stack(
    OperatorKernel* functor,
    DispatchKeySet dispatchKeySet,
    Stack* stack) {
  using ArgTypes = typename c10::remove_DispatchKeySet_arg_from_func<
      Functor>::parameter_types;
  constexpr size_t num_ivalue_args = guts::typelist::size<ArgTypes>::value;
  return call_functor_with_args_from_stack_<Functor, AllowDeprecatedTypes>(
      functor,
      dispatchKeySet,
      stack,
      std::make_index_sequence<num_ivalue_args>(),
      static_cast<ArgTypes*>(nullptr));
}

call_functor_with_args_from_stack_(
    OperatorKernel* functor,
    DispatchKeySet dispatchKeySet,
    Stack* stack,
    std::index_sequence<ivalue_arg_indices...> /*unused*/,
    guts::typelist::typelist<ArgTypes...>* /*unused*/) {
  return wrap_kernel_functor_unboxed<Functor>::call(
      functor,
      dispatchKeySet,
      ivalue_to_arg<
          typename decay_if_not_tensor<ArgTypes>::type,
          AllowDeprecatedTypes>::
          call(torch::jit::peek(
              *stack, ivalue_arg_indices, sizeof...(ivalue_arg_indices)))...);
}

struct wrap_kernel_functor_unboxed_<
  static ReturnType call(
      OperatorKernel* functor,
      DispatchKeySet /*unused*/,
      ParameterTypes... args) {
    KernelFunctor* functor_ = static_cast<KernelFunctor*>(functor);
    // Note [Plumbing Keys Through The Dispatcher 2]
    // See Note [Plumbing Keys Through The Dispatcher] for the background.
    // This functor explicitly takes in a dispatchKeySet and drops it on the
    // floor- it does not forward it to the registered kernel.
    //
    // This is due to the calling convention within the dispatcher, which
    // expects all registered kernels to have a first argument of type
    // DispatchKeySet.
    // This is not the case for pretty much all manually written kernels,
    // however- this functor serves to separate the calling convention of the
    // dispatcher from the calling convention of manually written kernels.
    return (*functor_)(std::forward<ParameterTypes>(args)...);
  }
```
##### WrapFunctionIntoFunctor.h
```
class WrapFunctionIntoFunctor_<
    FuncPtr,
    ReturnType,
    guts::typelist::typelist<Parameters...>>
    final : public c10::OperatorKernel {
 public:
  C10_ALWAYS_INLINE decltype(auto) operator()(Parameters... args) {
    return (*FuncPtr::func_ptr())(std::forward<Parameters>(args)...);
  }
};
} // namespace detail
```
#### KernelFunction_impl.h
torch.matmul->Tensor::matmul->matmul::call->dispatcher->KernelFunction

torch.matmul->Tensor::matmul->matmul::call->dispatcher->KernelFunction->BoxedKernelWrapper->BoxedKernel->PythonTLSSnapshot->redispatchBoxed->callBoxed
```
C10_ALWAYS_INLINE Return KernelFunction::call(
    const OperatorHandle& opHandle,
    DispatchKeySet dispatchKeySet,
    Args... args) const {
  return impl::BoxedKernelWrapper<Return(Args...)>::call(
      boxed_kernel_func_,
      opHandle,
      dispatchKeySet,
      std::forward<Args>(args)...);

inline void KernelFunction::callBoxed(
    const OperatorHandle& opHandle,
    DispatchKeySet dispatchKeySet,
    Stack* stack) const {
  boxed_kernel_func_.callBoxed(opHandle, dispatchKeySet, stack);
}

```
### dispatch
#### Dispatcher.h
根据输入tensors生成dispatchKeySet，选KernelFunction

torch.matmul->Tensor::matmul->matmul::call->dispatcher

torch.matmul->Tensor::matmul->matmul::call->dispatcher->KernelFunction->BoxedKernelWrapper->BoxedKernel->PythonTLSSnapshot->redispatchBoxed

torch.matmul->Tensor::matmul->matmul::call->dispatcher->KernelFunction->BoxedKernelWrapper->BoxedKernel->PythonTLSSnapshot->redispatchBoxed->BoxedKernel->unboxed->native::matmul->native::_matmul_impl->mm::call->...->autograd::mm->at::redispatch::mm->mm::redispatch->...


```
class TORCH_API Dispatcher final {
  LeftRight<ska::flat_hash_map<OperatorName, OperatorHandle>>
      operatorLookupTable_;
  std::list<OperatorDef> operators_;

  struct OperatorDef final {
    explicit OperatorDef(OperatorName&& op_name) : op(std::move(op_name)) {}

    impl::OperatorEntry op;


C10_ALWAYS_INLINE_UNLESS_MOBILE Return Dispatcher::call(
    const TypedOperatorHandle<Return(Args...)>& op,
    Args... args) const {
  auto dispatchKeySet =
      op.operatorDef_->op.dispatchKeyExtractor()
          .template getDispatchKeySetUnboxed<Args...>(args...);
  if (show_dispatch_trace()) {
    detail::_print_dispatch_trace(
        "[call]", toString(op.operator_name()), dispatchKeySet);
  }
  const KernelFunction& kernel = op.operatorDef_->op.lookup(dispatchKeySet);
  return kernel.template call<Return, Args...>(
      op, dispatchKeySet, std::forward<Args>(args)...);

inline void Dispatcher::redispatchBoxed(
    const OperatorHandle& op,
    DispatchKeySet dispatchKeySet,
    Stack* stack) const {
  if (show_dispatch_trace()) {
    detail::_print_dispatch_trace(
        "[redispatchBoxed]", toString(op.operator_name()), dispatchKeySet);
  const auto& kernel = entry.lookup(dispatchKeySet);
  kernel.callBoxed(op, dispatchKeySet, stack);

inline Return Dispatcher::redispatch(
  return kernel.template call<Return, Args...>(
      op, currentDispatchKeySet, std::forward<Args>(args)...);

class TORCH_API OperatorHandle {
  explicit OperatorHandle(
      std::list<Dispatcher::OperatorDef>::iterator operatorIterator)
      : operatorDef_(&*operatorIterator), operatorIterator_(operatorIterator) {}

  void redispatchBoxed(DispatchKeySet ks, Stack* stack) const {
    c10::Dispatcher::singleton().redispatchBoxed(*this, ks, stack);
  }

class TypedOperatorHandle<Return(Args...)> final : public OperatorHandle {
  C10_ALWAYS_INLINE Return call(Args... args) const {
    return c10::Dispatcher::singleton().call<Return, Args...>(
        *this, std::forward<Args>(args)...);
  }
  C10_ALWAYS_INLINE Return
  redispatch(DispatchKeySet currentDispatchKeySet, Args... args) const {
    return c10::Dispatcher::singleton().redispatch<Return, Args...>(
        *this, currentDispatchKeySet, std::forward<Args>(args)...);
  }

```
#### Dispatcher.cpp
dispatcher记录lib.namespace，记录op，kernel注册到op

算子: 做什么，例如 add

backend: 在哪里做，例如 CPU/CUDA

kernel: 具体怎么做，例如 add_cpu/add_cuda，有些 kernel 依赖 backend

一个算子可以对应多个 kernel，这些对应关系组成该算子的 dispatch table

输入tensors和线程状态 (autocast) 带有keys，算子的 DispatchKeyExtractor 提取并过滤 (fallthrough) 本次调用的 keys

Dispatcher 选择优先级最高的 key (kernel)，包装层 kernel 可以主动 redispatch，因此一次算子调用可能经过多次选择

```
Dispatcher
    libraries_
    std::list<OperatorDef> operators_
    LeftRight<ska::flat_hash_map<OperatorName, OperatorHandle>> operatorLookupTable_;

OperatorHandle
    OperatorDef
        OperatorEntry
            DispatchKeyExtractor
                dispatch_arg_indices_reverse
                nonFallthroughKeys_
            dispatchTable_
            kernels_
```

```
RegistrationHandleRAII Dispatcher::registerLibrary(std::string ns, std::string debug) {
  libraries_.emplace(ns, std::move(debug));
  return RegistrationHandleRAII([guard = this->guard_, this, ns] {
    std::lock_guard<std::mutex> lock(guard->mutex);
    if (!guard->alive.load()) {
      return;
    }
    deregisterLibrary_(ns);
  });
RegistrationHandleRAII Dispatcher::registerDef(FunctionSchema schema, std::string debug, std::vector<at::Tag> tags) {
  OperatorName op_name = schema.operator_name();
  auto op = findOrRegisterName_(op_name);
  op.operatorDef_->op.registerSchema(std::move(schema), std::move(debug), std::move(tags));
  ++op.operatorDef_->def_count;
  ++op.operatorDef_->def_and_impl_count;
  return RegistrationHandleRAII([guard = this->guard_, this, op, op_name] {
    // we need a lock to avoid concurrent writes
    std::lock_guard<std::mutex> lock(guard->mutex);
    if (!guard->alive.load()) {
      return;
    }
    deregisterDef_(op, op_name);
  });

OperatorHandle Dispatcher::findOrRegisterName_(const OperatorName& op_name) {
  const auto found = findOp(op_name);
  if (found.has_value()) {
    return *found;
  }

  operators_.emplace_back(OperatorName(op_name));
  OperatorHandle handle(--operators_.end());
  operatorLookupTable_.write([&] (ska::flat_hash_map<OperatorName, OperatorHandle>& operatorLookupTable) {
    operatorLookupTable.emplace(op_name, handle);
  });

std::optional<OperatorHandle> Dispatcher::findOp(const OperatorName& overload_name) {
  return operatorLookupTable_.read([&] (const ska::flat_hash_map<OperatorName, OperatorHandle>& operatorLookupTable) -> std::optional<OperatorHandle> {
    auto found = operatorLookupTable.find(overload_name);
    if (found == operatorLookupTable.end()) {
      return std::nullopt;
    }
    return found->second;
  });
}

RegistrationHandleRAII Dispatcher::registerImpl(
  auto op = findOrRegisterName_(op_name);
  auto handle = op.operatorDef_->op.registerKernel(
    *this,
    dispatch_key,
    std::move(kernel),
  ++op.operatorDef_->def_and_impl_count;
  return RegistrationHandleRAII([guard = this->guard_, this, op, op_name, dispatch_key, handle] {


 [call] op=[aten::matmul], key=[PythonTLSSnapshot]
  [redispatchBoxed] op=[aten::matmul], key=[AutogradCUDA]
   [call] op=[aten::mm], key=[PythonTLSSnapshot]
    [redispatchBoxed] op=[aten::mm], key=[AutogradCUDA]
     [redispatch] op=[aten::mm], key=[Python]
      [callBoxed] op=[aten::mm], key=[CUDA]
      [call] op=[aten::detach], key=[ADInplaceOrView]
       [redispatch] op=[aten::detach], key=[CUDA]
bool show_dispatch_trace() {
  static auto envar = c10::utils::get_env("TORCH_SHOW_DISPATCH_TRACE");
```
#### DispatchKeyExtractor.cpp
```
void DispatchKeyExtractor::setOperatorHasFallthroughForKey(DispatchKey k, bool has_fallthrough) {
  if (has_fallthrough) {
    nonFallthroughKeys_ = nonFallthroughKeys_.remove(k);
  } else {
    nonFallthroughKeys_ = nonFallthroughKeys_.add(k);
  }
  if (isPerBackendFunctionalityKey(toFunctionalityKey(k))) {
    if (has_fallthrough) {
      nonFallthroughKeysPerBackend_[backend_idx] = nonFallthroughKeysPerBackend_[backend_idx].remove(k);
    } else {
      nonFallthroughKeysPerBackend_[backend_idx] = nonFallthroughKeysPerBackend_[backend_idx].add(k);
    }
//如果所有backend相同，用全局即可
    for (const auto i : c10::irange(nonFallthroughKeysPerBackend_.size() - 1)) {
      if (nonFallthroughKeysPerBackend_[i] != nonFallthroughKeysPerBackend_[i+1]) {
        requiresBitsetPerBackend_ = true;
        return;
      }
    }
    requiresBitsetPerBackend_ = false;
    return;
  } else {
//所有backend都适用
    if (has_fallthrough) {
      for (const auto i : c10::irange(nonFallthroughKeysPerBackend_.size())) {
        nonFallthroughKeysPerBackend_[i] = nonFallthroughKeysPerBackend_[i].remove(k);
      }
    } else {
      for (const auto i : c10::irange(nonFallthroughKeysPerBackend_.size())) {
        nonFallthroughKeysPerBackend_[i] = nonFallthroughKeysPerBackend_[i].add(k);
      }
    }
  }
```
#### OperatorEntry.cpp
```
OperatorEntry
    DispatchKeyExtractor
    std::array<KernelFunction, c10::num_runtime_entries> dispatchTable_//最新kernel
    kernels_//一个dispatchkey上可以注册多个kernel，使用最新的，可删除后使用旧的
```
```
OperatorEntry::OperatorEntry(OperatorName&& operator_name)
: name_(std::move(operator_name)), dispatchTable_(), dispatchKeyExtractor_(DispatchKeyExtractor::makeUninitialized()), is_observed_(ObservedOperators::isObserved(name_))
{
  const auto& dispatcher = c10::Dispatcher::singleton();
  const auto& dispatch_table_index_to_key = getDispatchTableIndexToKey();
//生成所有可dispatch的key
  for (const auto dispatch_ix: c10::irange(dispatcher.backendFallbackKernels_.size())) {
//当某个算子没有该 key 的专用 kernel 时，可以使用该 key 的通用 fallback。保底
    const auto& bfk = dispatcher.backendFallbackKernels_[dispatch_ix];
    if (bfk.kernel.isValid()) {
      dispatchTable_[dispatch_ix] = bfk.kernel;
      if (bfk.kernel.isFallthrough()) {
//跳过
        dispatchKeyExtractor_.setOperatorHasFallthroughForKey(dispatch_table_index_to_key[dispatch_ix], true);
      }
    } else {
      dispatchTable_[dispatch_ix] = missingKernel().kernel;
    }
  }
}

static const auto& getDispatchTableIndexToKey() {
  static const auto result = []() {
    using result_type = std::array<DispatchKey, c10::num_runtime_entries>;
    result_type arr;
    arr.fill(DispatchKey::Undefined);
    for (const auto dispatch_key: allDispatchKeysInFullSet()) {
      const auto index = getDispatchTableIndexForDispatchKey(dispatch_key);
      TORCH_INTERNAL_ASSERT(arr.at(index) == DispatchKey::Undefined);
      arr.at(index) = dispatch_key;
    }
    return arr;
  }();
  return result;
}

static const std::vector<DispatchKey>& allDispatchKeysInFullSet() {
  static const auto result = []() {
    std::vector<DispatchKey> vec;
    for (const auto dispatch_key: DispatchKeySet(DispatchKeySet::FULL)) {
      vec.push_back(dispatch_key);
    }
    return vec;
  }();
  return result;
}

void OperatorEntry::registerSchema(FunctionSchema&& schema, std::string&& debug, std::vector<at::Tag> tags) {
  TORCH_INTERNAL_ASSERT(!schema_.has_value());
  dispatchKeyExtractor_.registerSchema(schema);
  schema_ = AnnotatedSchema(std::move(schema), std::move(debug));
}

//增加kernel，更新DispatchTable
OperatorEntry::AnnotatedKernelContainerIterator OperatorEntry::registerKernel(
  const c10::Dispatcher& dispatcher,
  std::optional<DispatchKey> dispatch_key,
  KernelFunction kernel,
  std::optional<CppSignature> cpp_signature,
  std::unique_ptr<FunctionSchema> inferred_function_schema,
  std::string debug
) {
  auto& k = dispatch_key.has_value() ? kernels_[*dispatch_key] : kernels_[DispatchKey::CompositeImplicitAutograd];
  k.emplace_front(std::move(kernel), std::move(inferred_function_schema), std::move(debug));
  AnnotatedKernelContainerIterator inserted = k.begin();
  if (dispatch_key.has_value()) {
    updateDispatchTable_(dispatcher, *dispatch_key);
  } else {
    updateDispatchTableFull_(dispatcher);
  }
  return inserted;

void OperatorEntry::updateDispatchTable_(const c10::Dispatcher& dispatcher, DispatchKey dispatch_key) {
  if (dispatch_key == DispatchKey::Undefined) {
    updateDispatchTableEntry_(dispatcher, dispatch_key);
    return;
  }
  for (auto k : c10::getRuntimeDispatchKeySet(dispatch_key)) {
    updateDispatchTableEntry_(dispatcher, k);
  }
  if (dispatch_key == DispatchKey::CompositeImplicitAutograd
   || dispatch_key == DispatchKey::CompositeExplicitAutograd
   || dispatch_key == DispatchKey::CompositeExplicitAutogradNonFunctional) {
    updateDispatchTableEntry_(dispatcher, DispatchKey::Undefined);
  }
  if (c10::isBackendDispatchKey(dispatch_key)) {
    DispatchKey autograd_key = getAutogradKeyFromBackend(toBackendComponent(dispatch_key));
    updateDispatchTableEntry_(dispatcher, autograd_key);
  }
}

void OperatorEntry::updateDispatchTableEntry_(const c10::Dispatcher& dispatcher, DispatchKey dispatch_key) {
  dispatchTable_[dispatch_ix] = computeDispatchTableEntry(dispatcher, dispatch_key);
  dispatchKeyExtractor_.setOperatorHasFallthroughForKey(dispatch_key, dispatchTable_[dispatch_ix].isFallthrough());

const KernelFunction& OperatorEntry::computeDispatchTableEntry(const c10::Dispatcher& dispatcher, DispatchKey dispatch_key) const {
  return computeDispatchTableEntryWithDebug(dispatcher, dispatch_key).first.kernel;
}

std::pair<const AnnotatedKernel&, const char*> OperatorEntry::computeDispatchTableEntryWithDebug(const c10::Dispatcher& dispatcher, DispatchKey dispatch_key) const {
  if (auto direct_registration = getKernelForDispatchKey(dispatch_key)) {
    return {*direct_registration, "kernel"};
  }

const AnnotatedKernel* OperatorEntry::getKernelForDispatchKey(DispatchKey dispatch_key) const{
  auto kern_it = kernels_.find(dispatch_key);
  if (kern_it != kernels_.end()) {
    return &kern_it->second.front();
  }
  return nullptr;
}
```
#### RegistrationHandleRAII.h
```
class RegistrationHandleRAII final {
  explicit RegistrationHandleRAII(std::function<void()> onDestruction)
      : onDestruction_(std::move(onDestruction)) {}
  ~RegistrationHandleRAII() {
    if (onDestruction_) {
      onDestruction_();
    }
  }
```
### library.cpp
DEF注册lib.namespace到dispatcher，记录句柄（用于lib释放时注销），FRAGMENT，IMPL啥也不干

注册算子签名到dispatcher，记录句柄

注册KernelFunction到dispatcher，记录句柄
```
Library::Library(Kind kind, std::string ns, std::optional<c10::DispatchKey> k, const char* file, uint32_t line)
  : kind_(kind)
  , ns_(ns == "_" ? std::nullopt : std::make_optional(std::move(ns)))
  , dispatch_key_(k.value_or(CatchAll) == CatchAll ? std::optional<c10::DispatchKey>() : k)
    switch (kind_) {
      case DEF:
        // Only DEFs require library uniqueness; fragments
        // don't register a library
        registrars_.emplace_back(
          c10::Dispatcher::singleton().registerLibrary(
            // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
            ns_.value(), debugString(file_, line_)
          )
        );
        [[fallthrough]];

Library& Library::_def(c10::FunctionSchema&& schema, c10::OperatorName* out_name, const std::vector<at::Tag>& tags, _RegisterOrVerify rv) & {
  switch (rv) {
    case _RegisterOrVerify::REGISTER:
      registrars_.emplace_back(
        c10::Dispatcher::singleton().registerDef(
          std::move(schema),
          debugString(file_, line_),
          tags
        )
      );
      break;
Library& Library::_impl(const char* name_str, CppFunction&& f, _RegisterOrVerify rv) & {
  switch (rv) {
    case _RegisterOrVerify::REGISTER:
      registrars_.emplace_back(
        c10::Dispatcher::singleton().registerImpl(
          std::move(name),
          dispatch_key,
          std::move(f.func_),
          f.cpp_signature_,
          std::move(f.schema_),
          debugString(std::move(f.debug_), file_, line_)
        )
      );
      break;
```

### PythonFallbackKernel.cpp
torch.matmul->Tensor::matmul->matmul::call->dispatcher->KernelFunction->BoxedKernelWrapper->BoxedKernel->PythonTLSSnapshot


torch.matmul->Tensor::matmul->matmul::call->dispatcher->KernelFunction->BoxedKernelWrapper->BoxedKernel->PythonTLSSnapshot->redispatchBoxed->BoxedKernel->unboxed->native::matmul->native::_matmul_impl->mm::call->...->autograd::mm->at::redispatch::mm->mm::redispatch->...->pythonFallback
```
TORCH_LIBRARY_IMPL(_, PythonTLSSnapshot, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&pythonTLSSnapshotFallback>());
}

void pythonTLSSnapshotFallback(const c10::OperatorHandle &op, c10::DispatchKeySet dispatch_keys, torch::jit::Stack* stack) {
  at::impl::MaybeSetTLSOnEntryGuard guard;

  op.redispatchBoxed(dispatch_keys & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::PythonTLSSnapshot), stack);
}

TORCH_LIBRARY_IMPL(_, Python, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&pythonFallback>());
}

void pythonFallback(const c10::OperatorHandle& op, c10::DispatchKeySet dispatch_keys, torch::jit::Stack* stack) {
  const auto mode_stack_len = c10::impl::TorchDispatchModeTLS::stack_len();
  if (mode_stack_len > 0) {
    RECORD_FUNCTION("PythonDispatchMode", torch::jit::last(*stack, num_arguments));
    const auto& cur_torch_dispatch_mode_state = c10::impl::TorchDispatchModeTLS::get_stack_at(mode_stack_len - 1);
    cur_torch_dispatch_mode_state->pyinterpreter()->dispatch(op, stack);
    return;
  }

```
### TensorBody.h
torch.matmul->Tensor::matmul

torch.matmul->Tensor::matmul->matmul::call->dispatcher->KernelFunction->BoxedKernelWrapper->BoxedKernel->PythonTLSSnapshot->redispatchBoxed->BoxedKernel->unboxed->native::matmul->native::_matmul_impl->mm::call->...
```
inline at::Tensor Tensor::matmul(const at::Tensor & other) const {
    return at::_ops::matmul::call(const_cast<Tensor&>(*this), other);
}

inline at::Tensor Tensor::mm(const at::Tensor & mat2) const {
    return at::_ops::mm::call(const_cast<Tensor&>(*this), mat2);
}
```
## native
###　LinearAlgebra.cpp
torch.matmul->Tensor::matmul->matmul::call->dispatcher->KernelFunction->BoxedKernelWrapper->BoxedKernel->PythonTLSSnapshot->redispatchBoxed->BoxedKernel->unboxed->native::matmul->native::_matmul_impl
```
Tensor matmul(const Tensor & tensor1, const Tensor & tensor2) {
  at::Tensor result, unused;
  result = at::native::_matmul_impl(unused, tensor1, tensor2);
  return result;
}

static Tensor _matmul_impl(
    return has_out ? at::mm_out(out, tensor1, tensor2) : tensor1.mm(tensor2);

```
## Operators_3.cpp
torch.matmul->Tensor::matmul->matmul::call->dispatcher->KernelFunction->BoxedKernelWrapper->BoxedKernel->PythonTLSSnapshot->redispatchBoxed->BoxedKernel->unboxed->native::matmul->native::_matmul_impl->mm::call->...->autograd::mm->at::redispatch::mm->mm::redispatch

```
at::Tensor mm::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & mat2) {

    static auto op = create_mm_typed_handle();
    return op.redispatch(dispatchKeySet, self, mat2);
}
```
## Operators_4.cpp
torch.matmul->Tensor::matmul->matmul::call
```
static C10_NOINLINE c10::TypedOperatorHandle<matmul::schema> create_matmul_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(matmul::name, matmul::overload_name)
      .typed<matmul::schema>();
}
at::Tensor matmul::call(const at::Tensor & self, const at::Tensor & other) {

    static auto op = create_matmul_typed_handle();
    return op.call(self, other);
}
```
## RedispatchFunctions.h
torch.matmul->Tensor::matmul->matmul::call->dispatcher->KernelFunction->BoxedKernelWrapper->BoxedKernel->PythonTLSSnapshot->redispatchBoxed->BoxedKernel->unboxed->native::matmul->native::_matmul_impl->mm::call->...->autograd::mm->at::redispatch::mm->mm::redispatch

```
    inline at::Tensor mm(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & mat2) {
        return at::_ops::mm::redispatch(dispatchKeySet, self, mat2);
    }

```
## RegisterCompositeImplicitAutograd_0.cpp
torch.matmul->Tensor::matmul->matmul::call->dispatcher->KernelFunction->BoxedKernelWrapper->BoxedKernel->PythonTLSSnapshot->redispatchBoxed->BoxedKernel->unboxed->native::matmul
```
at::Tensor wrapper_CompositeImplicitAutograd__matmul(const at::Tensor & self, const at::Tensor & other) {
    // No device check
  // DeviceGuard omitted
  return at::native::matmul(self, other);
}
```
## SavedTensorHooks.cpp
```
std::optional<std::pair<SafePyObject, SafePyObject>> SavedTensorDefaultHooks::get_hooks(bool ignore_is_tracing) {
  if (!is_initialized || tls.stack.empty() || (!ignore_is_tracing && tls.is_tracing)) {
    return std::nullopt;
  }
  return tls.stack.top();
}

void SavedTensorDefaultHooks::push_hooks(SafePyObject pack_hook, SafePyObject unpack_hook) {
  TORCH_INTERNAL_ASSERT(is_initialized);
  assertSavedTensorHooksNotDisabled();
  tls.stack.emplace(std::move(pack_hook), std::move(unpack_hook));
}
```
# c10
## core
### DispatchKeySet.cpp
```
DispatchKeySet::iterator& DispatchKeySet::iterator::operator++() {
//选出functionality的位
  uint64_t masked_functionality_bits =
      llvm::maskTrailingZeros<uint64_t>(next_functionality_) & *data_ptr_;
  uint64_t masked_backend_bits =
      llvm::maskTrailingZeros<uint64_t>(next_backend_) & full_backend_mask &
      *data_ptr_;
//找第一个为1的位
  uint64_t first_functionality_idx =
      llvm::findFirstSet(masked_functionality_bits);
  uint64_t first_backendcomponent_idx = llvm::findFirstSet(masked_backend_bits);

  auto new_next_functionality = first_functionality_idx + 1;
//在dispatchset里下标
  auto new_backendcomponent_idx = first_backendcomponent_idx + 1;
  auto next_dispatchkey_idx = new_next_functionality - num_backends;
//在dispatchkey里面的值

  if (isPerBackendFunctionalityKey(
          static_cast<DispatchKey>(next_dispatchkey_idx))) {

    current_dispatchkey_idx_ = next_dispatchkey_idx;
    current_backendcomponent_idx_ = new_backendcomponent_idx;
//在dispatchkey里面的值，要加上backend偏移

    uint64_t next_backendcomponent_bits =
        llvm::maskTrailingZeros<uint64_t>(first_backendcomponent_idx + 1) &
        full_backend_mask & *data_ptr_;
    uint64_t next_backendcomponent_idx =
        llvm::findFirstSet(next_backendcomponent_bits);
//没有下一个backend了，更新functionality下标
    if (next_backendcomponent_idx == std::numeric_limits<uint64_t>::max()) {
      next_functionality_ = new_next_functionality;
      next_backend_ = 0;
    } else {
//还有下一个backend，更新backend下标
      next_backend_ = first_backendcomponent_idx + 1;
    }
  } else {
//增加functionality bit
    current_dispatchkey_idx_ = next_dispatchkey_idx;
    next_functionality_ = new_next_functionality;
  }
  return *this;
}


initializeFunctionalityOffsetsAndMasks() {
  std::array<FunctionalityOffsetAndMask, num_functionality_keys>
      offsets_and_masks;
//基础key
  // manually set the first entry, which corresponds to Undefined.
  offsets_and_masks[0] = FunctionalityOffsetAndMask(0, 0);
  for (const auto functionality_idx : c10::irange(1, num_functionality_keys)) {
    // functionality_idx should be Dense -> 1, ...
    auto prev_offset_and_mask = offsets_and_masks[functionality_idx - 1];
    auto k = static_cast<DispatchKey>(functionality_idx);

    auto next_offset = prev_offset_and_mask.offset +
        (prev_offset_and_mask.mask == 0 ? 1 : num_backends);
//offset表示在dispatchtable里的起点，要考虑backend
    auto next_mask = isPerBackendFunctionalityKey(k) ? full_backend_mask : 0;
    offsets_and_masks[functionality_idx] =
        FunctionalityOffsetAndMask(next_offset, next_mask);
  }
  
```
### impl
#### TorchDispatchModeTLS.cpp
```
void TorchDispatchModeTLS::push_non_infra_mode_onto_stack(
    std::shared_ptr<PyObject_TorchDispatchMode> mode) {
  torchDispatchModeState.stack_.push_back(std::move(mode));

```
## util
加锁，被处理
```
template <typename T>
class Synchronized final {
  mutable std::mutex mutex_;
  T data_;
  template <typename CB>
  auto withLock(CB&& cb) {
    std::lock_guard<std::mutex> guard(this->mutex_);
    return std::forward<CB>(cb)(this->data_);
  }
```
# torch
## csrc

### autograd
#### engine.cpp
```
void Engine::thread_init(
  thread_main(graph_task);

auto Engine::thread_main(const std::shared_ptr<GraphTask>& graph_task) -> void {
  while (graph_task == nullptr || !graph_task->future_result_->completed()) {
      NodeTask task = local_ready_queue->pop();
      if (task.fn_ && !local_graph_task->has_error_.load()) {
            evaluate_function(
                local_graph_task,
                task.fn_.get(),
                task.inputs_,

void Engine::evaluate_function(
  auto outputs = call_function(graph_task, func, inputs);

static variable_list call_function(
    outputs = fn(std::move(inputs));

```
#### generated
##### Functions.h
```
struct TORCH_API MmBackward0 : public TraceableFunction {
  SavedVariable mat2_;
  SavedVariable self_;
```
##### Functions.cpp
反向节点的计算：save_tensor.unpack()，。。。
save_tensor.unpack()
```
variable_list MmBackward0::apply(variable_list&& grads) {
  auto mat2 = mat2_.unpack();
  auto self = self_.unpack();
  return MmBackward0_apply_functional(std::move(grads), needs_input_grad, mat2, mat2_layout, mat2_sym_sizes, mat2_sym_strides, self, self_layout, self_sym_sizes, self_sym_strides);
```
##### VariableType_8.cpp
torch.matmul->Tensor::matmul->matmul::call->dispatcher->KernelFunction->BoxedKernelWrapper->BoxedKernel->PythonTLSSnapshot->redispatchBoxed->BoxedKernel->unboxed->native::matmul->native::_matmul_impl->mm::call->...->autograd::mm->at::redispatch::mm

创建反向节点，包含save_tensor，dispatch，连接结果和反向节点的
```
at::Tensor mm(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & mat2) {
  c10::intrusive_ptr<MmBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = c10::make_intrusive<MmBackward0>();
    grad_fn->set_next_edges(collect_next_edges( self, mat2 ));
    if (grad_fn->should_compute_output(0)) {
      grad_fn->mat2_ = SavedVariable(mat2, false);
    }
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::mm(ks & c10::after_autograd_keyset, self_, mat2_);
  })();
  auto result = std::move(_tmp);
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
```
#### init.cpp
```
  m.def(
      "_make_saved_tensor",
      [](const at::Tensor& tensor,
         bool is_output,
         bool is_inplace_on_view) -> torch::autograd::SavedVariable {
        return torch::autograd::SavedVariable(
            tensor, is_output, is_inplace_on_view);
      },
      py::arg("tensor"),
      py::arg("is_output"),
  m.def(
      "_push_saved_tensors_default_hooks",
      [](py::function& pack_hook, py::function& unpack_hook) {
        torch::autograd::PyDefaultSavedVariableHooks::push_hooks(
            pack_hook, unpack_hook);
      });

    {"_push_on_torch_dispatch_stack",
     push_on_torch_dispatch_stack,
     METH_O,
     nullptr},

static PyObject* push_on_torch_dispatch_stack(
    PyObject* _unused,
    PyObject* arg) {
      c10::impl::TorchDispatchModeTLS::push_non_infra_mode_onto_stack(
          std::make_shared<c10::impl::PyObject_TorchDispatchMode>(
              arg, getPyInterpreter()));
}
```
#### node.h
```
struct TraceableFunction : public Node {

struct TORCH_API Node : c10::intrusive_ptr_target {
  variable_list operator()(variable_list&& inputs) {
      return apply(std::move(inputs));
```
#### python_engine.cpp
```
void PythonEngine::thread_init(
  Engine::thread_init(device, ready_queue, false);

std::unique_ptr<SavedVariableHooks> PythonEngine::
    get_default_saved_variable_hooks() {
  return PyDefaultSavedVariableHooks::get_hooks();
}
```
#### python_saved_variable_hooks.h
保存pack之后的结果

pack(x)->data_->unpack(data_)->x
```
struct PySavedVariableHooks : public SavedVariableHooks {
 private:
  PyObject* pack_hook_;
  PyObject* unpack_hook_;
  PyObject* data_ = nullptr;
};
```
#### python_saved_variable_hooks.cpp
```
void PyDefaultSavedVariableHooks::push_hooks(
    py::function& pack_hook,
    py::function& unpack_hook) {
  at::SavedTensorDefaultHooks::lazy_initialize();
  at::SavedTensorDefaultHooks::push_hooks(
      c10::SafePyObject(pack_hook.release().ptr(), getPyInterpreter()),
      c10::SafePyObject(unpack_hook.release().ptr(), getPyInterpreter()));
}
std::unique_ptr<SavedVariableHooks> PyDefaultSavedVariableHooks::get_hooks() {
  auto out = at::SavedTensorDefaultHooks::get_hooks();

void PySavedVariableHooks::call_pack_hook(const at::Tensor& tensor) {
  py::gil_scoped_acquire acquire;
  THPObjectPtr obj(THPVariable_Wrap(tensor));
  THPObjectPtr packed(
      PyObject_CallFunctionObjArgs(pack_hook_, obj.get(), nullptr));
  data_ = packed.release();

at::Tensor PySavedVariableHooks::call_unpack_hook() {
  THPObjectPtr res(PyObject_CallFunctionObjArgs(unpack_hook_, data_, nullptr));
```
#### python_torch_functions_manual.cpp

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

#### python_variable.cpp

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

#### python_variable.h

```
struct THPVariable {
  PyObject_HEAD
  // Payload
  at::Tensor cdata;
```
#### python_variable_methods.cpp
torch.matmul的cpp接口
```
static PyObject * THPVariable_matmul(PyObject* self_, PyObject* args, PyObject* kwargs)
  auto dispatch_matmul = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.matmul(other);
  };
  return wrap(dispatch_matmul(self, _r.tensor(0)));

template <PyObject* (*Func)(PyObject*, PyObject*, PyObject*)>
static PyObject * TypeError_to_NotImplemented_(PyObject* self, PyObject* args, PyObject* kwargs) {

  PyObject* ret = Func(self, args, kwargs);
  if (!ret && PyErr_ExceptionMatches(PyExc_TypeError)) {
    PyErr_Clear();
    Py_INCREF(Py_NotImplemented);
    ret = Py_NotImplemented;
  }
  return ret;
}
```
#### saved_variable.cpp
无hook: 保存tensor
有hook: save_tensor在创建时保存save_tensor_hooks，并调用pack_hook，结果存在hook中
```
namespace torch::autograd {

SavedVariable::SavedVariable(
    const Variable& variable,
    bool is_output,
    bool is_inplace_on_view) {
    std::unique_ptr<SavedVariableHooks> maybe_hooks =
        at::SavedTensorDefaultHooks::is_enabled() ? get_default_hooks()
                                                  : nullptr;
    if (maybe_hooks && !variable.unsafeGetTensorImpl()->is_wrapped_number()) {
      save_metadata(variable);
      set_hooks_and_pack_data(std::move(maybe_hooks), variable);
      TORCH_INTERNAL_ASSERT(!data_.defined());
      return;
    }
    if (!is_output || is_leaf_) {
      saved_original_ = true;
      data_ = variable;
      return;
    }
    save_metadata(variable);
    data_ = variable.tensor_data();
std::unique_ptr<SavedVariableHooks> SavedVariable::get_default_hooks() {
  return Engine::get_default_engine().get_default_saved_variable_hooks();
}
void SavedVariable::set_hooks_and_pack_data(
    std::unique_ptr<SavedVariableHooks>&& hooks,
    const Variable& data) {
  hooks_ = std::move(hooks);
  hooks_->call_pack_hook(saved_original_ ? data.detach() : data);
Variable SavedVariable::unpack(c10::intrusive_ptr<Node> saved_for) const {
  auto data = hooks_ ? hooks_->call_unpack_hook() : data_;
```

### Module.cpp

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
### PyInterpreter.cpp
torch.matmul->Tensor::matmul->matmul::call->dispatcher->KernelFunction->BoxedKernelWrapper->BoxedKernel->PythonTLSSnapshot->redispatchBoxed->BoxedKernel->unboxed->native::matmul->native::_matmul_impl->mm::call->...->autograd::mm->at::redispatch::mm->mm::redispatch->...->pythonFallback->TorchDispatch
```
void ConcretePyInterpreterVTable::dispatch(
  PyObject* obj = handle_torch_function_no_python_arg_parser(
      overloaded_args,
      args.ptr(),
      kwargs.ptr(),
      nullptr,
      torch_api_function_overload.ptr(),
      nullptr,
      &op,
      &arguments,
      TorchFunctionName::TorchDispatch);
```
### stub.c

```
PyMODINIT_FUNC PyInit__C(void)
{
  return initModule();
}
```
### utils
#### python_arg_parser.cpp
torch.matmul->Tensor::matmul->matmul::call->dispatcher->KernelFunction->BoxedKernelWrapper->BoxedKernel->PythonTLSSnapshot->redispatchBoxed->BoxedKernel->unboxed->native::matmul->native::_matmul_impl->mm::call->...->autograd::mm->at::redispatch::mm->mm::redispatch->...->pythonFallback->TorchDispatch
```
auto handle_torch_function_no_python_arg_parser(
    std::tie(ret, mode_obj) = dispatch_on_mode(
        args,
        kwargs,
        py_types,
        torch_api_function,
        is_torch_function,
        torch_function_name_str);

torch_function_name_str(torch_api_function,args,kwargs)
static std::tuple<py::object, py::object> dispatch_on_mode(
    ret = py::reinterpret_steal<py::object>(PyObject_CallMethod(
        mode_obj.ptr(),
        torch_function_name_str,
        "OOOO",
        torch_api_function,
        py_types.ptr(),
        args,
        kwargs));
```
#### python_dispatch.cpp
```
CppFunction
    KernelFunction func_
        BoxedKernel boxed_kernel_func_
            PythonKernelHolder:OperatorKernel functor_
                c10::SafePyObject func_
    dispatch_key_;
```
```
static c10::Synchronized<std::vector<std::unique_ptr<std::string>>>
    leaked_python_filenames_;

static c10::Synchronized<ska::flat_hash_map<
    c10::OperatorName,
    ska::flat_hash_map<c10::DispatchKey, std::shared_ptr<c10::SafePyObject>>>>
    python_registrations_;

  m.def(
      "_dispatch_library",
      [](const char* kind,
         std::string name,
         const char* dispatch,
         const char* file,
         uint32_t linenum) {
        HANDLE_TH_ERRORS
        // Store the file string in global storage to ensure it remains valid
        // for the lifetime of the Library object
        const char* leaked_file =
            leaked_python_filenames_.withLock([&](auto& filenames) {
              filenames.push_back(std::make_unique<std::string>(file));
              return filenames.back()->c_str();
            });

        return std::make_unique<torch::Library>(
            parseKind(kind),
            std::move(name),
            std::string(dispatch).empty()
                ? std::nullopt
                : std::make_optional(c10::parseDispatchKey(dispatch)),
            leaked_file,
            linenum);
        END_HANDLE_TH_ERRORS_PYBIND
      },
      .def(
          "define",
          [](const py::object& self,
             const char* schema,
             const char* alias_analysis,
             const std::vector<at::Tag>& tags) {
            auto parsed_schema =
                torch::schema(schema, parseAliasAnalysisKind(alias_analysis));
            self.cast<torch::Library&>().def(
                std::move(parsed_schema), tags, register_or_verify());
            // TODO: this is dumb, had to make a second copy
            return torch::schema(schema, parseAliasAnalysisKind(alias_analysis))
                .name();
          },
      .def(
          "impl",
          [](const py::object& self,
             const char* name,
             // TODO: empty string no longer works
             c10::DispatchKey dispatch,
             py::object func,
             bool with_keyset) {
            HANDLE_TH_ERRORS
            auto& lib = self.cast<torch::Library&>();
            if (func.is(py::module::import("torch.library")
                            .attr("fallthrough_kernel"))) {
              lib.impl(
                  name,
                  torch::dispatch(dispatch, CppFunction::makeFallthrough()),
                  register_or_verify());
            } else {
              lib.impl(
                  name,
                  torch::dispatch(
                      dispatch,
                      CppFunction::makeFromBoxedFunctor(
                          std::make_unique<PythonKernelHolder>(
                              func, dispatch, with_keyset))),
                  register_or_verify());
              python_registrations_.withLock([&](auto& regs) {
                regs[lib._resolve(name)].insert_or_assign(
                    dispatch,
                    std::make_shared<c10::SafePyObject>(
                        func.release().ptr(), getPyInterpreter()));
              });
            }
            END_HANDLE_TH_ERRORS_PYBIND
          },

class PythonKernelHolder : public c10::OperatorKernel {
  void operator()(
      const c10::OperatorHandle& op,
      c10::DispatchKeySet keyset,
      torch::jit::Stack* stack) {
```

## function.py
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
## graph.py
设置save_tensor_hooks

前向保存的save_tesor实际上是经过save_tensor_hooks的结果
```
class saved_tensors_hooks:
    In that context, the ``pack_hook`` function will be called every time an
    operation saves a tensor for backward (this includes intermediary results
    saved using
    :func:`~torch.autograd.function._ContextMethodMixin.save_for_backward` but
    also those recorded by a PyTorch-defined operation). The output of
    ``pack_hook`` is then stored in the computation graph instead of the
    original tensor.
    def __enter__(self) -> None:
        torch._C._autograd._push_saved_tensors_default_hooks(
            self.pack_hook, self.unpack_hook
        )

    def __exit__(self, *args: object) -> None:
        torch._C._autograd._pop_saved_tensors_default_hooks()

```



## compile
时间流程：
```
Python
   │字节码
Dynamo
   │FX Graph
AOTAutograd(Ahead Of Time Autograd)
   │AOTAutograd 并不是自己推导梯度，它依赖 Autograd Mode 建好的图，然后把这张动态图提取、追踪成一张静态的 Backward FX Graph
Inductor
    │
    ├──调用 cuBLAS
    │
    ├──调用 cuDNN
    │
    ├──生成 Triton
    │
    └──生成 C++
```

## cuda
### memory

https://pytorch.org/memory_viz

```
_record_memory_history()
_dump_snapshot(file_name)
_record_memory_history(enabled=None)
```

## distributed

### fsdp.fully_shard
Each call to fully_shard creates one communication group containing all parameters in the module that are not already assigned to a group from an earlier call on a submodule. Each group’s parameters are all-gathered together in one collective before forward, and their gradients are reduce-scattered together in one collective after backward. In forward, all-gathers run on a separate CUDA stream, so the next module’s all-gather can overlap with the current module’s forward compute. Each module’s pre-forward hook issues its own all-gather and waits for it to complete before running the module.

mesh (Optional[DeviceMesh]) – This data parallel mesh defines the sharding and device. If 1D, then parameters are fully sharded across the 1D mesh (FSDP) with (Shard(0),) placement. If 2D, then parameters are sharded across the 1st dim and replicated across the 0th dim (HSDP) with (Replicate(), Shard(0)) placement. The mesh’s device type gives the device type used for communication; if a CUDA or CUDA-like device type, then we use the current device.
### run

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

## export
```
Python
   │
Dynamo
   │
ExportedProgram
   │
保存
```


## include
### Aten
#### core
##### boxing
###### BoxedKernel_impl.h
```
template <class KernelFunctor>
inline BoxedKernel BoxedKernel::makeFromFunctor(
    std::unique_ptr<KernelFunctor> kernelFunctor) {
  return BoxedKernel(
      std::move(kernelFunctor),
      [](OperatorKernel* kernel,
         const OperatorHandle& op,
         DispatchKeySet ks,
         Stack* stack) {
        (*static_cast<KernelFunctor*>(kernel))(op, ks, stack);
      });
}
```
###### KernelFunction.h
```
class TORCH_API KernelFunction final {
  BoxedKernel boxed_kernel_func_;

```
###### KernelFunction_impl.h
```

inline KernelFunction KernelFunction::makeFromBoxedKernel(
    BoxedKernel boxed_fn) {
  return KernelFunction(
      std::move(boxed_fn), nullptr); // no unboxed function pointer
}

inline KernelFunction::KernelFunction(
    BoxedKernel boxed_fn,
    void* unboxed_kernel_func,
    void* sym_unboxed_kernel_func = nullptr)
    : boxed_kernel_func_(std::move(boxed_fn)),
      unboxed_kernel_func_(unboxed_kernel_func),
      sym_unboxed_kernel_func_(sym_unboxed_kernel_func) {}
```
##### dispatch
###### DispatchKeyExtractor.h

```
struct TORCH_API DispatchKeyExtractor final {
  DispatchKeySet nonFallthroughKeys_;
  std::array<DispatchKeySet, num_backends> nonFallthroughKeysPerBackend_;

  static DispatchKeyExtractor makeUninitialized() {
    return DispatchKeyExtractor(c10::utils::bitset());
  }
  explicit DispatchKeyExtractor(c10::utils::bitset dispatch_arg_indices_reverse)
      : dispatch_arg_indices_reverse_(dispatch_arg_indices_reverse),
#参与dispatch的参数位置
        nonFallthroughKeys_(DispatchKeySet::FULL) {
#所有 DispatchKey 都不是 fallthrough
    for (const auto i : c10::irange(nonFallthroughKeysPerBackend_.size())) {
      nonFallthroughKeysPerBackend_[i] = DispatchKeySet::FULL;
    }
  }

  void registerSchema(const FunctionSchema& schema) {
    dispatch_arg_indices_reverse_ = makeBitsetForDispatchArgs(schema);
  }
  static c10::utils::bitset makeBitsetForDispatchArgs(
      const FunctionSchema& schema) {
    c10::utils::bitset dispatch_arg_indices_reverse;
    for (const auto index : c10::irange(schema.arguments().size())) {
      if (isDispatchType(*schema.arguments()[index].type())) {
        dispatch_arg_indices_reverse.set(schema.arguments().size() - 1 - index);
      }
    }
    return dispatch_arg_indices_reverse;
  }
  static bool isDispatchType(const Type& type) {
    return type.isSubtypeOf(*TensorType::get()) ||
        type.isSubtypeOf(*ListType::ofTensors()) ||
        type.isSubtypeOf(*ListType::ofOptionalTensors()) ||
        type.isSubtypeOf(*OptionalType::ofTensor());

```

### c10
#### core
##### DispatchKey.h
基础key：一些需要与backend组合才能用于dispatch，一些不用

包含基础和复合
```
constexpr uint8_t num_functionality_keys =
    static_cast<uint8_t>(DispatchKey::EndOfFunctionalityKeys);

#define C10_FORALL_FUNCTIONALITY_KEYS(_) \
  _(Dense, )                             \
  _(Quantized, Quantized)                \
  _(Sparse, Sparse)                      \
  _(SparseCsr, SparseCsr)                \
  _(NestedTensor, NestedTensor)          \
  _(AutogradFunctionality, Autograd)


enum class DispatchKey : uint16_t {
  Undefined = 0,
  CatchAll = Undefined,
  ...
  EndOfFunctionalityKeys, // End of functionality keys. //基础

//自动生成 functionality 和 backend 的“笛卡尔积”

#define DEFINE_PER_BACKEND_KEYS_FOR_BACKEND(n, prefix) prefix##n,

#define DEFINE_PER_BACKEND_KEYS(fullname, prefix)      \
  StartOf##fullname##Backends,                         \
      C10_FORALL_BACKEND_COMPONENTS(                   \
          DEFINE_PER_BACKEND_KEYS_FOR_BACKEND, prefix) \
          EndOf##fullname##Backends = prefix##Meta,

  C10_FORALL_FUNCTIONALITY_KEYS(DEFINE_PER_BACKEND_KEYS)

      EndOfRuntimeBackendKeys = EndOfAutogradFunctionalityBackends, //复合
    ...
};


constexpr uint8_t num_backends =
    static_cast<uint8_t>(BackendComponent::EndOfBackendKeys);

#define C10_FORALL_BACKEND_COMPONENTS(_, extra) \
  _(CPU, extra)                                 \
  _(CUDA, extra)                                \
  _(HIP, extra)                                 \
  _(XLA, extra)                                 \
  _(MPS, extra)                                 \
  _(IPU, extra)                                 \
  _(XPU, extra)                                 \
  _(HPU, extra)                                 \
  _(VE, extra)                                  \
  _(Lazy, extra)                                \
  _(MTIA, extra)                                \
  _(MAIA, extra)                                \
  _(PrivateUse1, extra)                         \
  _(PrivateUse2, extra)                         \
  _(PrivateUse3, extra)                         \
  _(Meta, extra)

enum class BackendComponent : uint8_t {
  InvalidBit = 0,
#define DEFINE_BACKEND_COMPONENT(n, _) n##Bit,
  C10_FORALL_BACKEND_COMPONENTS(DEFINE_BACKEND_COMPONENT, unused)
#undef DEFINE_BACKEND_COMPONENT
  EndOfBackendKeys = MetaBit,
};

class DispatchKeySet final {
//根据dispatchkey生成dispatchset
  constexpr explicit DispatchKeySet(DispatchKey k) {
    if (k == DispatchKey::Undefined) {
      // Case 1: handle Undefined specifically
      repr_ = 0;

//根据dispatchkeyset下标生成dispatchkey
constexpr DispatchKey toRuntimePerBackendFunctionalityKey(
    DispatchKey functionality_k,
    BackendComponent backend_k) {
  if (functionality_k == DispatchKey::Dense) {
    return static_cast<DispatchKey>(
        static_cast<uint8_t>(DispatchKey::StartOfDenseBackends) +
        static_cast<uint8_t>(backend_k));
  }

//需要复合的基础functionality不占位
constexpr uint16_t num_runtime_entries = num_functionality_keys +
    (numPerBackendFunctionalityKeys() * (num_backends - 1));

```
##### DispatchKeySet.h
低位：backend bits

高位：functionality bits（基础）

存储所有可能的dispatchkey
```
class DispatchKeySet final {
  constexpr DispatchKeySet(Full /*unused*/)
      : repr_((1ULL << (num_backends + num_functionality_keys - 1)) - 1) {}
//EndOfFunctionalityKeys也占1位，要减去，backend没有
  iterator begin() const {
    return iterator(&repr_);
  }
  class iterator {
    explicit iterator(
        const uint64_t* data_ptr,
        uint8_t next_functionality = num_backends,
        uint8_t next_backend = 0)
        : data_ptr_(data_ptr),
          next_functionality_(next_functionality),
          next_backend_(next_backend) {
      ++(*this);
    }

    DispatchKey operator*() const {
      auto functionality_key =
          static_cast<DispatchKey>(current_dispatchkey_idx_);//在dispatchkey里面的索引
      if (isPerBackendFunctionalityKey(functionality_key)) {
        auto next_key = toRuntimePerBackendFunctionalityKey(
            functionality_key,
            static_cast<BackendComponent>(current_backendcomponent_idx_));
        return next_key;
      } else {
        return functionality_key;
      }
    }

//bit越高，优先级越高？
    int getDispatchTableIndexForDispatchKeySet() const {
        auto functionality_idx =
            DispatchKeySet(repr_ >> num_backends).indexOfHighestBit();
//找最高1，基础dispatchkey
        auto offset_and_mask = offsetsAndMasks()[functionality_idx];
//dispatchkey在表里的下标
        auto backend_idx =
            DispatchKeySet((repr_ & offset_and_mask.mask) >> 1).indexOfHighestBit();
        return offset_and_mask.offset + backend_idx;
    }

inline int getDispatchTableIndexForDispatchKey(DispatchKey k) {
  return DispatchKeySet(k).getDispatchTableIndexForDispatchKeySet();
}

C10_ALWAYS_INLINE static const std::
    array<FunctionalityOffsetAndMask, num_functionality_keys>&
    offsetsAndMasks() {
  static auto offsets_and_masks_ = initializeFunctionalityOffsetsAndMasks();
  return offsets_and_masks_;
}
```
## library.h

```
class TORCH_API Library final {
  std::vector<c10::RegistrationHandleRAII> registrars_;
  Library& def(
      c10::FunctionSchema&& s,
      const std::vector<at::Tag>& tags = {},
      _RegisterOrVerify rv = _RegisterOrVerify::REGISTER) & {
    return _def(std::move(s), nullptr, tags, rv);
  }
  template <typename Name, typename Func>
  Library& impl(
      Name name,
      Func&& raw_f,
      _RegisterOrVerify rv = _RegisterOrVerify::REGISTER) & {
    CppFunction f(std::forward<Func>(raw_f));
#endif
    return _impl(name, std::move(f), rv);
  }

class TORCH_API CppFunction final {
  template <class KernelFunctor>
  static CppFunction makeFromBoxedFunctor(
      std::unique_ptr<KernelFunctor> kernelFunctor) {
    return makeFromBoxedKernel(
        c10::BoxedKernel::makeFromFunctor(std::move(kernelFunctor)));
  static CppFunction makeFromBoxedKernel(c10::BoxedKernel kernel) {
    return CppFunction(
        c10::KernelFunction::makeFromBoxedKernel(std::move(kernel)),
        /* cpp_signature */ std::nullopt, // not known for boxed functions
        /* schema */ nullptr);
  }
  std::optional<c10::DispatchKey> dispatch_key_;

inline CppFunction dispatch(c10::DispatchKey k, Func&& raw_f) {
  CppFunction f(std::forward<Func>(raw_f));
  if (k == c10::DispatchKey::CatchAll) {
    f.dispatch_key_ = std::nullopt;
  } else {
    f.dispatch_key_ = k;
  }
  return f;
}

```
## library.py
实现、替换原有算子

注册lib、op签名、kernel到dispatcher

计算？？？
```
class Library:
    def __init__(self, ns, kind, dispatch_key=""):
        filename, lineno = f.f_code.co_filename, f.f_lineno
        self.m: Any | None = torch._C._dispatch_library(
            kind, ns, dispatch_key, filename, lineno
        )
        self.ns = ns
        self._op_defs: set[str] = set()
        self._op_impls: set[str] = set()
        self._registration_handles: list[torch._library.utils.RegistrationHandle] = []
        self.kind = kind
        self.dispatch_key = dispatch_key
        weakref.finalize(
            self,
            _del_library,
    def define(self, schema, alias_analysis="", *, tags=()):
        result = self.m.define(schema, alias_analysis, tuple(tags))
        qualname = self.ns + "::" + name
        self._op_defs.add(qualname)
        _defs.add(qualname)
    def impl(
        self, op_name, fn, dispatch_key="", *, with_keyset=False, allow_override=True
    ):
        key = self.ns + "/" + name.split("::")[-1] + "/" + dispatch_key
        self.m.impl(
            name,
            dispatch_key if dispatch_key != "" else "CompositeImplicitAutograd",
            fn,
            with_keyset,
        )
        _impls.add(key)
        self._op_impls.add(key)

        
@functools.singledispatch
def impl(
    return _impl(qualname, types, func, lib=lib, disable_dynamo=False)

@impl.register
def _(
    lib: Library, name: str, dispatch_key: str = ""
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:

    def wrap(f: Callable[_P, _T]) -> Callable[_P, _T]:
        lib.impl(name, f, dispatch_key)
        return f

    return wrap
```
## nn

### module

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

## profiler

timeline:

https://ui.perfetto.dev

chrome://tracing

memory:

pip install torch_tb_profiler

tensorboard --logdir dir_name
```
p=profile(schedule,on_trace_ready,profile_memory)

p.step()

tb_handler=tensorboard_trace_handler(dir_name)
def on_trace_ready_fn(p):
    p.export_chrome_trace(file_name)
    #tb_hander(p)

schedule(wait,warmup,active,repeat)
```
active结束调用一次on_trace_ready，step一次表示走完一步

## Tensor

```
next_functions = [
    (grad_fn_node, input_index),  
    ...
]
第一个元素（grad_fn_node）：真正的「梯度函数节点」（比如梯度累加器、上一个操作的 grad_fn）；
第二个元素（input_index）：该节点在父操作中的输入索引（仅多输入操作有用，单输入操作下为 0）。
```

chunk: view，全部释放才释放显存

detach: Returned Tensor shares the same storage with the original one. 不再有grad_fn，反向没法找到下一个函数

items(): D2H+sync

view: Returns a new tensor with the same data as the self tensor but of a different shape.

## utils
### _python_dispatch.py

```
def _push_mode(mode: TorchDispatchMode) -> None:
    k = mode._dispatch_key if hasattr(mode, "_dispatch_key") else None
    if k is not None and k != torch._C.DispatchKey.PreDispatch:
        raise AssertionError(
            "mode._dispatch_key must be None or DispatchKey.PreDispatch"
        )
    if k is None:
        _push_on_torch_dispatch_stack(mode)


class TorchDispatchMode:
    def __enter__(self):
        _push_mode(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _pop_mode(mb_dk_or_mode_key)
```
## checkpoint

发现没保存checkpoint输入? 

Use MUST_* over PREFER_* to indicate that the policy should not be overridden by other subsystems like torch.compile.

```
def context_fn():
    def policy_fn(ctx, op, *args, **kwargs):
        global forward_idx, recompute_idx

        if op != torch.ops.aten.mm.default:
            return CheckpointPolicy.PREFER_SAVE

        if ctx.is_recompute:
            recompute_idx += 1
            idx = recompute_idx
        else:
            forward_idx += 1
            idx = forward_idx

        if idx % 2 == 0:
            return CheckpointPolicy.PREFER_RECOMPUTE
        else:
            return CheckpointPolicy.MUST_SAVE
    return create_selective_checkpoint_contexts(policy_fn)
y = checkpoint(
    model,
    x,
    use_reentrant=False,
    context_fn=context_fn,
    early_stop=False
)
loss = y.sum()
loss.backward()
```

### checkpoint.py
_CheckpointFrame保存checkpoint输入tensor，前向save_tensor不保存任何tensor，save_op保存输出

反向一开始就执行整个checkpoint的前向，_CheckpointFrame保存反向需要的tensor，recompute_op计算，TorchDispatchMode返回输出

如果checkpoint里面所有算子都不需要save_tensor的话，没有hook调用，没有frame引用，不会保存输入，不会重算，比如torch.add

旧版重算会进policy
```
new_frame.save_inputs(*args)保存输入tensor
_checkpoint_hook.__enter__给设置save_tensor_hooks
user_forward_context.__enter__设置_CachingTorchDispatchMode
前向: save_tensor.pack_hook没有保存tensor，TorchDispatchMode调用算子，TorchDispatchMode保存save_op输出
user_forward_context.__exit__移除TorchDispatchMode
_checkpoint_hook.__exit__移除hooks
反向: save_tensor.unpack_hook
    once:
        _recomputation_hook.__enter__给设置save_tensor_hooks
        user_recompute_context.__enter__设置_CachedTorchDispatchMode
        前向: save_tensor.pack_hook将tensor保存到frame，TorchDispatchMode调用recompute_op，返回save_op,recompute_op输出
        user_recompute_context.__exit__移除TorchDispatchMode
        _recomputation_hook.__exit__移除hooks
    从frame取结果
```
```
from torch._C._autograd import _make_saved_tensor, SavedTensor

@torch._disable_dynamo
def checkpoint(
    context_fn: Callable[[], Tuple[ContextManager, ContextManager]] = noop_context_fn,
    if use_reentrant:
        if context_fn is not noop_context_fn or debug is not False:
            raise ValueError(
                "Passing `context_fn` or `debug` is only supported when "
                "use_reentrant=False."
            )
        return CheckpointFunction.apply(function, preserve, *args)
    else:
        gen = _checkpoint_without_reentrant_generator(
            function, preserve, context_fn, determinism_check, debug, early_stop, *args, **kwargs
        )
        # Runs pre-forward logic
        next(gen)
        try:
            ret = function(*args, **kwargs)
        try:
            next(gen)
        except StopIteration:
            return ret

class _CheckpointFrame:
    def __init__(self, recompute_fn, early_stop, unpack_error_cb, metadata_fn) -> None:
        self.recompute_fn = recompute_fn
        self.saved_args: List[Any] = []
        self.recomputed: DefaultDict[
            int, weakref.WeakKeyDictionary[_Handle, torch.Tensor]
        ] = defaultdict(weakref.WeakKeyDictionary)
#弱引用，随着反向进行可以释放中间激活
    def save_inputs(self, *args):
        self.saved_args = [
            _make_saved_tensor(arg, is_output=False)
            if isinstance(arg, torch.Tensor) else arg
            for arg in args
        ]

class _checkpoint_hook(torch.autograd.graph.saved_tensors_hooks):
    def __init__(self, frame) -> None:
        def pack_hook(x):
            # See Rule 4 above
            holder = _Holder()
            frame.weak_holders.append(weakref.ref(holder))
            # Save metadata to detect non-determinism
            if frame.metadata_fn is not None:
                with torch.no_grad():
                    frame.x_metadatas.append(frame.metadata_fn(x))
            return holder

        def unpack_hook(holder):
            gid: GraphExecGroup | None | int = GraphExecGroup._get_current_group()
            if not frame.is_recomputed[gid]:
                args = frame.get_inputs()

                try:
                    with _recomputation_hook(
                        weakref.ref(frame), gid
                    ), torch.autograd.enable_grad():
                        # See Note: [compiled autograd and checkpoint unpack hook]
                        _run_fn_with_dynamo_disabled(frame.recompute_fn, *args)
                frame.is_recomputed[gid] = True
            ret = frame.recomputed[gid][holder.handles[gid]]
            holder.handles[gid] = None
            return ret

        if frame.unpack_error_cb is not None:
            def unpack_hook_with_error_cb(holder):
                try:
                    return unpack_hook(holder)
                except CheckpointError as e:
                    frame.unpack_error_cb(e)
            super().__init__(pack_hook, unpack_hook_with_error_cb)
        else:
            super().__init__(pack_hook, unpack_hook)

class _recomputation_hook(torch.autograd.graph.saved_tensors_hooks):
    def __init__(self, target_frame_ref: ReferenceType, gid: GraphExecGroup | int) -> None:
        @torch._dynamo.disable
        def pack_hook(x):
            x = x.detach() if x.requires_grad else x
            if holder is not None:
                _internal_assert(holder.handles.get(gid, None) is None)
                holder.handles[gid] = _Handle()
                target_frame.recomputed[gid][holder.handles[gid]] = x
            return x

        def unpack_hook(x):
            return x

        super().__init__(pack_hook, unpack_hook)

def _checkpoint_without_reentrant_generator(
    forward_context, recompute_context = context_fn()
    def recompute_fn(*args) -> None:
            with (
                device_autocast_ctx,
                torch.amp.autocast("cpu", **cpu_autocast_kwargs),
                recompute_context,
                device_ctx,
                nested_fx_trace_ctx,
            ):  # type: ignore[attr-defined]
                fn(*args, **kwargs)
    new_frame = _CheckpointFrame(
        recompute_fn,
        _enable_checkpoint_early_stop if _enable_checkpoint_early_stop is not None else early_stop,
        unpack_error_cb,
        metadata_fn
    )
    new_frame.save_inputs(*args)
    with _checkpoint_hook(new_frame), forward_context:
        try:
            yield
    new_frame.forward_completed = True

class _CachingTorchDispatchMode(TorchDispatchMode):
    def __init__(self, policy_fn, storage, ac_graph_id=None) -> None:
        self.policy_fn = policy_fn
        self.storage = storage
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **kwargs)
        policy = self.policy_fn(SelectiveCheckpointContext(is_recompute=False, op_output=out),
                                func, *args, **kwargs)
        if policy in (CheckpointPolicy.MUST_SAVE, CheckpointPolicy.PREFER_SAVE) or is_compiling:
            self.storage[key][idx] = tree_map(lambda x: _VersionWrapper(_detach_helper(x)), out)
        else:
            self.storage[key][idx] = _RECOMPUTE
        return out

class _CachedTorchDispatchMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        func_storage = self.storage.get(key)
        entry = func_storage.get(idx)
        elif entry is _RECOMPUTE:
            kwargs = {} if kwargs is None else kwargs
            return func(*args, **kwargs)
        else:
            func_storage[idx] = _CONSUMED
            return tree_map(lambda x: x.get_val(self.allow_cache_entry_mutation), entry)

def create_selective_checkpoint_contexts(policy_fn_or_list, allow_cache_entry_mutation=False):
    return (
        _CachingTorchDispatchMode(policy_fn, storage),
        _CachedTorchDispatchMode(None, storage, allow_cache_entry_mutation),
    )
```

## data

iterableDataset


# ...

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
register_full_backward_pre_hook
