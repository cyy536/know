# LLVM

```
[ 源代码 ]
     ↓
[ 前端（Frontend） ] → 语言特定（C、Rust、Swift...）
     ↓
[ 中间表示（IR） ]   → LLVM IR（统一、语言无关）
     ↓
[ 中端（Optimizer） ] → 通用优化（死代码消除、循环优化等）
     ↓
[ 后端（Backend） ]   → 平台特定（x86、ARM、RISC-V...）
     ↓
[ 机器码 ]
```

LLVM IR 是设计为公开、稳定、可编程的中间语言，Clang 是 LLVM 的 C/C++/Objective-C/CUDA C++ 前端

MLIR 是 LLVM 项目的一部分，它允许编译器在不同抽象层级定义自己的 IR，称为 Dialect

Triton 编译器使用 MLIR 基础设施定义了自己的 Dialect，例如 Triton IR 和 TritonGPU IR。它逐步完成张量运算优化、GPU 布局选择和线程映射，再降低到更底层的 IR，最终生成 GPU 可执行代码。

# AI编译器

```
torch.compile
  -> TorchDynamo 捕获计算图 torch.FX.GraphModule
  -> 图编译器（如 Inductor、XLA）优化
  -> 后端生成/调用高性能 kernel
  -> Triton / CUDA / cuDNN 等执行
CUDA Graph 可减少重复 launch 开销
```

现代图编译器既会调用现成的高性能 kernel，也会在需要时调用 kernel 编译器动态生成新 kernel

Inductor 会调用 Triton，XLA 是 Google 开发的图级编译器，最初为 TensorFlow/JAX 设计，后支持 PyTorch

PyTorch 默认不编译，需显式调用 torch.compile 才能启用编译优化，JAX 强制使用 XLA 进行编译优化

TensorRT 可以看作 NVIDIA GPU 上的推理编译器。它把 ONNX 或 TensorRT network 编译成 engine，核心工作包括图优化、层融合、精度/layout 优化、tactic 选择和 kernel profiling。它主要复用并选择高度优化的 kernel 实现，但也可能涉及 fused kernel 和 JIT-compiled code。

# 静态图

静态图是指在执行前构建完成的计算图

静态图执行流程：图捕获、静态图表示、图优化、静态图执行器

## torch.compile

torch.compile(f)不修改 f 的字节码

```
def f(x):
    y = torch.relu(x)
    return y + 1
```

首次调用 compiled_f(x)

```
读取原始字节码
生成新字节码
替换函数的 __code__
等价于
def f_instrumented(x):
    proxy_x = make_proxy(x)          
    proxy_y = dynamo_record(torch.relu, proxy_x)                   
    proxy_out = dynamo_record(operator.add, proxy_y, 1)
    return proxy_out
执行重写后的字节码，构建 FX Graph：每个 dynamo_record 调用向 Graph 添加一个 Node
```

非 PyTorch 操作不可捕获，强制将图分割为多段，导致多个编译单元，无法跨段优化

```
执行流到达 print 时
停止当前图构建
将已捕获的子图编译并执行（得到真实张量 a）
在原始 Python 环境中执行 print
捕获后续代码
```

## torch.jit.script

纯静态分析不执行，不是所有 python 操作都能映射成 TorchScript

```
script 装饰器调用 inspect.getsource(f) 获取函数源码
使用 Python 标准库 ast.parse() 将源码转为 AST
将 AST 节点映射到 TorchScript 算子（if x > 0 → prim::If）
当遇到不支持的 AST 节点时，script 会抛出 UnsupportedNodeError
生成序列化图
```

## torch.jit.trace

```
example_input = torch.tensor([1.0, 2.0])
traced_f = torch.jit.trace(f, example_input)
真实执行代码（包括控制流判断），但只记录“张量操作”，不记录“控制流逻辑本身”
生成序列化图
```
