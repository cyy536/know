# c

ANSI C = C89，是 C 语言标准化的起点

C89 在 1990 年被 ISO/IEC 采纳为国际标准，称为 ISO/IEC 9899:1990，简称 C90

ISO C，通常指最新或指定版本的 C 标准（如 C99、C11）。

GNU C 不是标准，而是 GNU Compiler Collection（GCC）对 ISO C 的超集扩展

默认启用：当你用 gcc hello.c 编译时，实际使用的是 GNU C 模式。

```
栈内存布局（简化版）：
|-------------------| 高地址
| 环境变量字符串... | 比如 "HOME=/root"
| 环境变量字符串... | 比如 "PATH=/usr/bin"
| 参数字符串...     | 比如 "c"
| 参数字符串...     | 比如 "b"
| 参数字符串...     | 比如 "a"
| 参数字符串...     | 比如 "./test"
|-------------------|
| envp 指针数组     | env_start = &envp[0]，env_end = &envp[n]（最后一个元素是NULL）
| argv 指针数组     | arg_start = &argv[0]，arg_end = &argv[argc]（argc是参数个数）
| argc（参数个数）  | 比如 4（对应 ./test a b c）
|-------------------| 低地址
```
## static

全局变量具有全局唯一性，多个文件可通过 extern 共享同一个变量

每个文件中的 static 全局变量是独立的副本，函数也可以加 static

全局变量，static 局部变量存储空间在程序加载时就分配在 .data 或 .bss 中

在标准C语言中，全局变量，static 局部变量的初始化值必须在编译时或链接时确定，不能依赖运行时才能计算的值

.data / .bss 的数据是所有线程共享的，普通局部变量在用户栈

inline：给编译器在本 TU 内展开函数体用，不能内联时不会生成函数

An inline function definition does not provide an external definition for the function. Any function call appearing in the source file containing an inline definition will either be satisfied by inlining the function definition at the call site, or by a reference to an externally defined function.
Therefore any program containing inline definitions should link with an object file containing an extern function definition.
Use of both the keywords extern and inline with a function definition (or on any prototype in the file continuing the function definition) will result in an external function being defined in that object file. 
编译器必须生成一个可链接的函数实体

```
static inline int add(int a, int b) {
    return a + b;
}
编译器按需生成一个局部函数实体
```

## 链接

```
-U__CUDA_NO_HALF_OPERATORS__ -DCUDA_ARCH=xx
#undefine __CUDA_NO_HALF_OPERATORS__
#define CUDA_ARCH xx

-c只编译不链接，需要头文件不需要库，链接和运行不需要头文件
链接阶段解决符号 → 静态库和动态库都用
运行阶段执行动态加载 → 只对动态库有要求
静态链接将所有依赖的库代码（如函数、全局变量等）直接复制到最终的可执行文件中

-I是头文件路径
-L是编译器找库的路径，-l是需要链接的库名
编译器将所有找到的库文件完整路径（如 /path/to/libs/libfoo.so）传给 ld (静态链接器 / 链接器)
-Wl是 GCC/Clang 的选项，用于将逗号分隔的参数传递给ld
-Wl,-rpath -Wl,/usr/local/openmpi-4.1.5/lib -Wl,--enable-new-dtags
ld -rpath /usr/local/lib --enable-new-dtags
-rpath用于指定.so的搜索路径，让 ld 将 /path/to/lib 写入可执行文件的 RPATH

ld.so (动态链接器 / 装载器) 查找路径顺序大致如下：
LD_LIBRARY_PATH → 用户环境变量，最高优先级
RPATH / RUNPATH → 可执行文件内置路径
系统默认路径 → /lib, /usr/lib, /usr/local/lib 等

链接器的符号解析是从左到右的，因此目标文件应该放在库前面
overlap.o -o overlap \
-L/usr/local/cuda/lib64 \
-L/root/paddlejob/workspace/env_run/output/chenyuyan/Nccl/nccl/build/lib \
-lcudart -lnccl
```

## gdb

```
gdb -p PID1

gdb -ex "break aclnnGroupedMatmul" \
    -ex "break aclnnGroupedMatmulV2" \
    -ex "break aclnnGroupedMatmulV3" \
    -ex "break aclnnGroupedMatmulV4" \
    -ex "break aclnnGroupedMatmulV5" \
    -ex "run" \
    --args python example1.py

backtrace
break test.c:10 if i >= 100 
continue
delete 2
delete break
finish
info breakpoints
list
next
step
set scheduler-locking on
GDB will only allow the current thread to run when the program is resumed (e.g., with continue, next, or step commands). All other threads are held stopped by the operating system's scheduler. 
```

操作系统在程序发生异常而异常在进程内部又没有被捕获的情况下，会把进程此刻内存、寄存器状态、运行堆栈等信息转储保存在一个文件里。

gdb driver-app core.5616，gdb会显示程序crash的位置

## ...

```
1000000000000LL

void foo(int a[4]);
void foo(int a[]);
void foo(int *a);
等价

scanf() 跳过空白读字符串，后续空白留在缓冲区
c = getchar();
putchar(c);

#pragma unroll
for (...) {
    ...
}
要求编译器展开循环。
```

# cpp

C++ 编译器本质上是 “增强版 C 编译器”，它在兼容 C 语言编译流程的基础上，额外增加了大量针对 C++ 特性的处理逻辑。

## 参数

```
.h写参数默认值
void f(int a);   
void f(int a, int b = 2); 
编译器生成不同函数

f(10); // error
编译器在调用点展开，实际上生成的函数只接受完整参数列表

template <typename... Args> // 声明 Args 是类型参数包
void bar(Args... args) {    // 声明 args 是函数参数包
    func(args...);          // 展开 args 为 func(a1, a2, ...)
    std::tuple<Args...> t;  // 展开 Args 为 std::tuple<T1, T2, ...>
}
参数包在编译期展开，无法运行时动态增减
```

## 引用

两个变量共享一个地址

通常是用指针实现的

左值表示可寻址的对象，其余右值

常量引用 (const int&)：可以绑定左值和右值，但不能修改值。

右值引用 (int&&)：只能绑定右值。

引用必须在声明时立即初始化，成员引用可以在类的构造函数中初始化

```
Fragment_q frag_q[Mma_tile_p::MMAS_M]
func(frag_q);
void func(Fragment (&a)[Mma_tile::MMAS_M]);
```

## 类

声明.h，函数写在.h是内联

成员函数，静态变量定义在.cpp

const成员函数不允许修改类的任何非mutable成员变量，指针成员和引用成员指向的非const变量还是可以修改的

const ClassName * const this

const对象只能调用const成员函数

没有手写构造函数，会自动生成一个默认构造函数，只要有一个就不会自动生成

= default 用于显式请求编译器生成默认的构造函数、拷贝构造函数、拷贝赋值运算符等

没有初始化列表就看声明初始值，否则按照默认初始化

```
拷贝构造
A(const A&a)
移动构造(用于指针转移的场景，区别于拷贝)
A(A&&a):_p(a.p){}
调用移动构造A a(std::move(b));
std::move(b)返回 b的右值引用，本身仅改变值的类别，用于触发移动构造或拷贝
被移动对象的析构会在它的作用域结束时触发
```

```
A& operator=(const A&a)
A& operator=(A&&a)
```

```
友元声明打破private，通常写在类的声明中
friend std::ostream& operator<<(std::ostream&o,const A&a);
std::ostream& operator<<(std::ostream& o, const A& a) {
    o << "A(value=" << a.value << ")";
    return o;
}
```

父类函数都继承，都可以重写

虚函数通过基类的指针或引用调用实例函数，非虚函数看类型

每个含有虚函数的类在编译时会生成一个虚函数指针数组，存储该类所有虚函数的地址

派生类Derived若重写基类虚函数，其虚函数表中对应的地址会被替换为Derived::func1等新函数地址

派生类会为每个基类维护独立的虚函数表，子类独有的虚函数存放在第一个基类的虚函数表中

每个对象实例的内存起始位置存储一个指向虚函数表的指针vptr

当通过基类指针调用虚函数时，通过对象的vptr找到虚函数表

抽象类：含纯虚函数（virtual void func() = 0;），强制子类实现特定逻辑，不能实例化

子类的变量会隐藏父类的同名变量，但父类的变量依然存在，只是需要通过作用域解析运算符

子类对象的大小是父类与子类所有成员的总和

如果父类有默认构造函数（无参数），则子类构造函数会自动调用父类的默认构造函数。如果父类没有默认构造函数（只有带参数的构造函数），则子类必须显式调用父类的某个构造函数，否则编译报错。父类构造函数总是先于子类构造函数执行，且执行顺序遵循继承声明顺序。

## 单例

```
class Singleton {
public:
    // 删除拷贝和赋值操作（C++11 起）
    Singleton(const Singleton&) = delete;
    Singleton& operator=(const Singleton&) = delete;

    // 全局访问点
    static Singleton& getInstance() {
        static Singleton instance;  // 局部静态变量，C++11 起保证线程安全初始化
        return instance;
    }

private:
    // 私有构造函数
    Singleton() = default;
};
```

## 函数对象

```
std::bind用于绑定可调用对象的部分参数，生成一个新的可调用对象
std::function通过类型擦除机制，可以包装任何可调用实体
函数对象是通过重载operator()的类实例
[捕获列表] (参数列表) mutable -> 返回类型 { 函数体 }
Lambda在编译时被转换为匿名类，捕获列表初始化成员变量
mutable 用于修饰成员变量，operator()为const函数
悬空引用：Lambda被调用时外部变量已销毁，引发未定义行为
```

## 模板

.h

完美转发是为了转发参数

```
template <typename T>
void foo(T&& arg) {
    // 将 arg 按原值类别转发给 target
    target(std::forward<T>(arg));
} 
int x = 42;
const int y = 100;
foo(x);       // x 是左值 → T = int&,      T&& = int&
foo(y);       // y 是 const 左值 → T = const int&, T&& = const int&
foo(100);     // 100 是右值 → T = int,      T&& = int&&
foo(std::move(x)); // std::move(x) 是右值 → T = int, T&& = int&&
```

实现泛型函数（如 std::make_shared、emplace_back），转发参数到某个类的构造器

```
std::make_unique<T>(42)==std::unique_ptr<T>(new T(42));
```

## new

malloc仅分配内存，不调用构造函数或初始化内存

new对基本类型（如 int）可显式初始化（如 new int(5)），对类对象自动调用构造函数完成初始化，释放内存，自动调用析构函数。

```
int* arr = new int[10];
```

智能指针是对new指针的管理

unique_ptr 包含一个指针，禁用拷贝语义，支持移动语义。析构delete。

shared_ptr 支持拷贝、移动语义。强引用计数记录当前共享对象的shared_ptr数量，归零时释放对象。弱引用计数记录观察该对象的weak_ptr数量，归零时释放控制块。引用计数的增减通过原子操作保证线程安全。

```
template<typename T>
struct ControlBlock {
    T* ptr_;                                  // 托管的对象指针
    std::atomic<size_t> shared_count_;        // 强引用计数（shared_ptr 数量）
    std::atomic<size_t> weak_count_;          // 弱引用计数（weak_ptr 数量 + 1 个占位）
这个“额外的 1”表示：只要还有 shared_ptr 存在，control block 就不能被销毁。当最后一个 shared_ptr 释放时，它会减少这个占位，此时若 weak_count_ == 0，才真正删除 control block。
T& operator*() const { return *ptr_; }
T* operator->() const { return ptr_; }
get()可获得原指针

memory_order_relaxed:编译器和 CPU 可以自由地将该操作与其他内存操作重排序
memory_order_acquire:当前线程中后续的所有读写操作不能被重排到该原子操作之前
memory_order_release:当前线程中之前的所有读写操作不能被重排到该原子操作之后
memory_order_acq_rel:之前的操作不能重排到它之后，之后的操作不能重排到它之前
void producer() {
    data = 42;                                       
    ready.store(true, std::memory_order_release);   
}
void consumer() {
    while (!ready.load(std::memory_order_acquire)) {}
    assert(data == 42); 
}
```

weak_ptr 包含控制块指针。从shared_ptr创建。通过 lock() 获取临时 shared_ptr，若对象已释放则返回空

```
class A { std::shared_ptr<B> b; };
class B { std::shared_ptr<A> a; };
auto a = std::make_shared<A>();
auto b = std::make_shared<B>();
a->b = b;
b->a = a; 
//指针析构后不能delete Ap,Bp

class A { std::shared_ptr<B> b; };
class B { std::weak_ptr<A> a; }; 

std::weak_ptr<MyClass> w = s;
if (auto tmp = w.lock()) {  // 升级为shared_ptr
    tmp->doSomething();     // 安全使用对象
}
```

```
std::move(shared_ptr)避免引用计数原子操作（拷贝会增加计数，移动不会）。

class ResourcePool : public std::enable_shared_from_this<ResourcePool<T>> {
    std::shared_ptr<ResourcePool<int>> getPtr() {
        return shared_from_this();  
    }
auto pool = std::make_shared<ResourcePool>();
auto p2 = pool->getPtr();  // p2 和 pool 共享控制块

enable_shared_from_this 内部保存一个弱引用 _weak_this
当对象被第一个 shared_ptr 管理时：构造 shared_ptr 会把控制块的弱引用赋给 _weak_this
std::shared_ptr<T> shared_from_this() {
    return _weak_this.lock(); // 返回 shared_ptr
}
```

## ...

```
auto：忽略引用和顶层 const
decltype：精确返回表达式的声明类型
using NewType = OldType;类型别名（支持模板）
using namespace std;引入命名空间/类型
typedef OldType NewType;
```

vector

```
ans.emplace_back(newInterval);
template <class InputIterator>    void insert (iterator position, InputIterator first, InputIterator last);
tmp.clear();清空vector中的所有元素
vector的reserve增加了vector的capacity，但是它的size没有改变！ 而resize改变了vector的capacity同时也增加了它的size
```

queue

```
front()->T&
emplace
```

```
static bool down(const ListNode*a,const ListNode*b){}
priority_queue<ListNode*,vector<ListNode*>,decltype(&down)>pq(&down);
struct compare{bool operator()(const ListNode*l,const ListNode*r){}}
priority_queue<ListNode*,vector<ListNode*>,compare>pq;
top()->const T&
```

deque

set

```
for(auto i=ss.begin();i!=ss.end();i++){}
count()
insert()
unordered_set<string>ws(wordDict.begin(),wordDict.end());
```

unordered_map

```
earse
tmp.first,tmp.second
unordered_map<long long,int>val_co={{0,1}};
```

string

```
stoi();stof
to_string()
(n,'a');
substr(i,len);
```

algorithm

```
template< class RandomIt, class Compare >
void sort( RandomIt first, RandomIt last, Compare comp );
当两个元素相等时，必须返回 false
可以传静态函数
```

## thread

```
std::lock_guard<std::mutex> lock(comm_task_list_mutex_);
在构造时锁定 comm_task_list_mutex_
在 lock_guard 析构时（如离开作用域）自动释放锁

std::unique_lock<std::mutex> lock(comm_task_list_mutex_);
//构造时自动锁定互斥锁
comm_task_list_cv_.wait_for(
    lock,
    std::chrono::milliseconds(loop_thread_sleep_millis),
    [&]() -> bool { return terminated_.load(); });
它会自动释放锁lock并使当前线程进入等待状态
直到以下情况之一发生：
被其他线程通过notify_one()或notify_all()唤醒
指定的超时时间到达
谓词条件满足（第三个参数返回true）
返回时会重新获取锁
```

std::async支持两种策略（通过 std::launch参数）：

std::launch::async：强制异步执行（新线程）。

std::launch::deferred：延迟执行（调用 future.get()时在当前线程运行）。

```
if (future.valid()) {  // 检查 future 是否关联了异步任务
    future.wait();     // 阻塞等待任务完成
}
future = std::async(std::launch::async, [&]() { task->ClearRecord(); });
```

## gflags

是 Google 开源的一个 C++ 库，专门用于高效管理命令行参数。它的核心目标是简化命令行参数的解析、存储和访问

```
#include <gflags/gflags.h>
DEFINE_string(config_path, "config.yaml", "Path to config file");
DECLARE_string(config_path);
gflags::ParseCommandLineFlags(&argc, &argv, true);
std::cout << FLAGS_config_path << std::endl;
```

## glog

```
#include "glog/logging.h"
LOG(INFO) << "CommTaskManager init success.";
VLOG(3) << "IsTimeout: " << IsTimeout();
//Verbose Log
```

# python

C实现的

## 对象

```
//一切皆对象，对象一定有类型，类型也是对象，类型的类型是“纯类型”
typedef struct {
    Py_ssize_t ob_refcnt;       // 引用计数
    PyTypeObject *ob_type;       // 指向对象类型的指针
} PyObject;
typedef struct {
    PyObject ob_base;      
    Py_ssize_t ob_size;     // 元素数量（如列表的长度）
} PyVarObject;
typedef struct _typeobject {
    // 1. 所有对象都有的头部
    PyObject_VAR_HEAD

    // 2. 类型基本信息
    const char *tp_name;        // 类型名："int"、"str"、"list"
    Py_ssize_t tp_basicsize;    // 一个实例占多大内存
    Py_ssize_t tp_itemsize;     // 变长元素大小

    // 3. 生命周期函数
    destructor tp_dealloc;      // 销毁对象时调用
    allocfunc tp_alloc;         // 分配内存
    newfunc tp_new;             // 创建实例（如 int()）

    // 4. 通用方法
    reprfunc tp_repr;           // repr()
    hashfunc tp_hash;           // hash()
    getattrfunc tp_getattr;     // 获取属性__getattr__
    setattrfunc tp_setattr;     // 设置属性

    // 5. 三大功能接口（决定对象是什么类型的行为）
    PyNumberMethods *tp_as_number;    // 数字行为：+ - * /
    PySequenceMethods *tp_as_sequence; // 序列行为：[] len()
    PyMappingMethods *tp_as_mapping;  // 映射行为：键值对

    // 6. 继承相关
    struct _typeobject *tp_base;  // 父类
    PyObject *tp_dict;            // 类的字典（存放属性、方法）

    // ... 还有很多标志、缓存、槽位，但不重要
} PyTypeObject;

//类型链的最大节点数是3，可通过metaclass扩展
PyTypeObject PyType_Type = {
    PyVarObject_HEAD_INIT(&PyType_Type, 0)  // 自引用：ob_type 指向自己
    "type",                                 // 类型名称
    .tp_base = &PyBaseObject_Type,         
};
type
|          \          \
人类      苹果类      PyBaseObject_Type
|   \     |     \
人a，人b，苹果a，苹果b

//python设置了类型继承的起点
//继承链就可以很深了
PyTypeObject PyBaseObject_Type = {
    PyVarObject_HEAD_INIT(&PyType_Type, 0)  // 类型是 type
    "object",                               // 类型名
    .tp_base = NULL,                        // 父类为 NULL（它是根）
};
PyBaseObject_Type
|          \         \
人类      苹果类      type
| \       | \  

大对象使用C malloc，销毁时调用free，小对象自己管理

大部分对象用引用计数立即释放，可能形成环的对象额外加入 GC 跟踪列表
GC 周期性运行从“可能有环”的对象中找：refcnt > 0 但从根（栈、全局变量）不可达的对象
找到 → 拆环 → 手动 decref → 释放

[ PyGC_Head ][ PyObject ][ 对象特有数据 ]
```

```
可以通过写C创建新类型，和内置类型一样，属性和方法存在结构体字段里
也可以写python class创建，解释器在内存里 new 了一个 PyTypeObject
类的属性、方法→ 存在类的 __dict__

python class实例化，解释器在内存里 new 了一个 PyInstanceObject
typedef struct {
    PyObject_HEAD;          // 引用计数 + 类型
    PyObject *ob_dict;      // 实例属性字典 __dict__
} PyInstanceObject;
实例的属性→ 存在实例的 __dict__
```

## 拷贝

不可变对象：任何修改操作都会生成新对象，原对象保持不变，可哈希（字符串）

元组本身不可变，但若其元素是可变对象，则这些元素的内容可修改，元组对元素的引用地址不变

可变对象：修改操作直接作用于原对象，不可哈希

Python 自定义 class 的实例默认是可变的，浅拷贝会为新实例创建一个新的 ob_dict 字典（地址不同），但字典里的可变值（如列表、字典）仍引用原实例的对象

赋值（=）：只是给同一个对象绑定新的变量名，新旧变量共享同一个内存地址，修改一个会影响另一个；

浅拷贝：创建新对象，但新对象的子元素仍引用原对象的子元素（只拷贝 “第一层”）；

深拷贝：创建全新对象，递归拷贝所有层级的子元素，新旧对象完全独立。

浅拷贝：仅复制对象的第一层结构，内层嵌套对象与原对象共享内存地址

深拷贝：递归复制所有层级的对象

对于不可变对象，拷贝不会创建新对象，本质还是赋值；

```
当两个字符串字面量仅用空格或换行分隔时，Python 会在编译时自动合并它们
如果需要在字符串中保留换行符，可以用三引号 """..."""或 '''...'''

list[start:end:step]

OrderedDict([('Linear', QuantizationScheme()),()])严格保留键值对的插入顺序
sorted() 只消费“可迭代对象”里的元素，而 OrderedDict 被迭代时只产出“键”

defaultdict(new_comm_group)
创建一个默认值为 new_comm_group() 的字典
```

## 描述符

```
Python 里的描述符协议只有三个可选方法：
__get__(self, instance, owner) → 读属性时触发
__set__(self, instance, value) → 写属性时触发
__delete__(self, instance) → 删除属性时触发
只要实现了其中至少一个（通常是 __get__），这个对象就是描述符
# 描述符类（实现了 __get__）
class MyDesc:
    def __get__(self, instance, owner):
        return 10

# 这个对象就是描述符
desc_obj = MyDesc()
```

内置类型的C方法被包装成描述符，每次通过内置实例访问C方法，描述符通过tp_descr_get生成一个方法实例（绑定内置实例的C方法）

```
PyTypeObject PyLong_Type = {
    ...
    .tp_methods = long_methods, 
    ...
};
static PyMethodDef long_methods[] = {
    {"bit_length", (PyCFunction)long_bit_length, METH_VARARGS, doc},
    {"__add__",     (PyCFunction)long_add,       METH_O,         doc},
    {NULL, NULL}, // 结尾
};
a.bit_length
去 PyLong_Type.tp_methods 里找 "bit_length"
根据这个 PyMethodDef，创建一个描述符(单例可以缓存)
typedef struct {
    PyObject_HEAD
    PyMethodDef *m_def;   
} PyMethodDescrObject;
PyMethodDescr_Type 有 tp_descr_get
调用tp_descr_get
// 伪代码简化版
PyObject *method_descr_get(PyObject *desc, PyObject *obj, PyObject *cls) {
	//obj是a，cls是int
    if (obj == NULL) {
        // 从类访问：返回描述符自身（PyMethodDescrObject）
        Py_INCREF(desc);
        return desc;
    }
    // 从实例访问：创建并返回 builtin_function_or_method 对象
    return PyMethod_New(desc, obj);
}
builtin_function_or_method是可调用对象（能加 () 执行）；自动绑定实例；调用时都自动把绑定的实例传给底层函数
typedef struct {
    PyObject_HEAD
    PyObject *im_self;       // 绑定的实例（比如 123 这个 int 对象）
    PyMethodDef *im_meth;    // 指向 C 函数的描述表（包含 long_bit_length 指针）
} PyCFunctionObject;  // builtin_function_or_method 对应的 C 结构体
print天生就是 builtin_function_or_method
```

python函数本身就是描述器

```
def f：PyFunction_Type 有 tp_descr_get
tp_descr_get逻辑类似method_descr_get
给 Python 函数用的 PyMethod_New：创建 method 对象；（PyMethodObject）
给内置方法用的 PyMethod_New：创建 builtin_function_or_method 对象
typedef struct {
    PyObject_HEAD
    PyObject *im_func;  // 指向原函数（PyFunctionObject）→ 对应 __func__
    PyObject *im_self;  // 指向绑定的实例 → 对应 __self__
} PyMethodObject;

obj.f(10)
m.__func__=f
m.__self__=obj
m(10)
m.__func__(obj, 10)
obj.f(10)执行 Python 字节码
a.bit_length()执行 C 机器码
```

method_descriptor，function生成绑定实例的方法，staticmethod返回function，classmethod生成绑定类的方法

```
// staticmethod 的 __get__ 实现（简化版）
PyObject *staticmethod_descr_get(PyObject *self, PyObject *obj, PyObject *type) {
    // self：staticmethod 对象，内部保存原函数
    PyObject *func = PyStaticMethod_GET_FUNC(self);
    Py_INCREF(func);
    return func;  // 直接返回原函数，不绑定任何实例/类
}
// classmethod 的 __get__ 实现（简化版）
PyObject *classmethod_descr_get(PyObject *self, PyObject *obj, PyObject *type) {
    // self：classmethod 对象，内部保存原函数
    PyObject *func = PyClassMethod_GET_FUNC(self);
    // 若没传类（type 为 NULL），用实例的类
    if (type == NULL) type = Py_TYPE(obj);
    // 绑定类（type）而非实例，生成方法对象
    return PyMethod_New(func, type);
}
```

method_descriptor，function，staticmethid，classmethod 只有 get，没有 set，是非数据描述符

property不返回方法，返回函数结果

```
property 有 __get__ 和 __set__，是数据描述符，不能被实例覆盖
class my_property:
    def __init__(self, fget=None, fset=None):
        self.fget = fget  # 读的函数
        self.fset = fset  # 写的函数
        
	#把函数执行后返回结果
    def __get__(self, instance, owner):
        if instance is None:
            return self
        return self.fget(instance)

	def __set__(self, instance, value):
        if self.fset is None:
            raise AttributeError("只读属性，不能赋值")
        self.fset(instance, value)

    # 装饰器语法：@x.setter
    def setter(self, fset):
        self.fset = fset
        return self
```

```
obj.attr==getattr(obj,"attr")
obj->ob_type->tp_getattro(obj, "attr")
PyObject_GenericGetAttr(obj, "attr") {
    1. 调用 obj.__getattribute__("attr")（如果自定义了）；
       → 若没自定义，执行默认逻辑：
    2. 遍历 obj 的类 + 所有父类的属性，找 "attr"：
        a. 找到 → 判断是否是「数据描述符」（有 __set__）：
           - 是 → 调用 attr.__get__(obj, obj.__class__) → 返回结果，流程结束；
           - 否 → 先暂存这个 attr，继续下一步；
        b. 没找到 → 暂存“未找到”，继续下一步；
    3. 检查 obj 的 __dict__（实例属性）：
        a. 找到 "attr" → 返回该值，流程结束；
        b. 没找到 → 回到步骤 2 暂存的结果：
            i. 若步骤 2 找到了非数据描述符 → 调用 attr.__get__(obj, obj.__class__) → 返回结果；
            ii. 若步骤 2 找到了普通类属性 → 返回该属性值；
            iii. 若步骤 2 没找到 → 继续下一步；
    4. 调用 obj.__getattr__("attr")（如果定义了）→ 返回结果；
    5. 若没定义 __getattr__ → 抛出 AttributeError；
}
```

```
def tp_setattro(obj, name, value):
    # 步骤 1：获取对象的类（及父类）
    cls = type(obj)
    mro = cls.__mro__  # 类的继承链（从子类到父类）
    
    # 步骤 2：遍历继承链，查找是否有该属性的「数据描述符」
    desc = None
    for base in mro:
        if name in base.__dict__:
            attr = base.__dict__[name]
            # 判断是否是数据描述符（有 __set__ 方法）
            if hasattr(attr, '__set__'):
                desc = attr
                break  # 找到第一个数据描述符就停止
    
    # 步骤 3：如果找到数据描述符 → 调用其 __set__
    if desc is not None:
        desc.__set__(obj, value)
        return  # 流程结束
    
    # 步骤 4：没有数据描述符 → 直接写入实例 __dict__
    obj.__setattr__('name', value)
    
    # 步骤 5：object 已经内置了原生的 __setattr__ 实现
    # 注意：如果实例是不可变对象（如 int/str），这里会抛异常
    if hasattr(obj, '__dict__'):
        obj.__dict__[name] = value
    else:
        raise AttributeError(f"{cls.__name__} 对象不支持属性赋值")
```

```
hasattr(obj, name)
首先检查对象 obj 的实例属性字典（__dict__），看是否存在 name 对应的键；
如果实例字典中没有，会去对象所属的类（obj.__class__）的属性字典中查找；
如果类的字典中也没有，会继续向上遍历类的继承链（父类、祖父类等），直到 object 类；
如果以上都没找到，会检查对象是否实现了 __getattr__ 魔法方法
```

## 函数

你执行 def add(a,b): ... 时，解释器先编译生成 PyCodeObject，再创建 PyFunctionObject，并把 func_code 指向这个代码对象

```
typedef struct _PyCodeObject {
    PyObject_HEAD
    int co_argcount;        // 位置参数个数（如add的2）
    int co_kwonlyargcount;  // 仅限关键字参数个数
    PyObject *co_names;     // 函数内引用的名字（如全局变量名）
    PyObject *co_varnames;  // 局部变量名元组（如('a','b','c')）
    PyObject *co_consts;    // 常量元组（如函数内的字面量1、"hello"）
    PyObject *co_code;      // 核心：字节码指令（bytes类型）
    int co_stacksize;       // 执行该代码需要的求值栈大小
    int co_flags;           // 标志位（如是否是生成器、闭包等）
    // 其他：co_filename（代码所在文件）、co_firstlineno（起始行号）等
} PyCodeObject;
```

```
typedef struct _PyFunctionObject {
    PyObject_HEAD          // Python对象公共头（引用计数等）
    PyObject *func_code;    // 指向PyCodeObject（函数的静态字节码）
    PyObject *func_globals;// 指向函数的全局作用域（字典）
    PyObject *func_defaults;// 默认参数元组（如def f(a=1): ... 中的(1,)）
    PyObject *func_closure;// 闭包的自由变量（cell对象元组，非闭包为NULL）
    PyObject *func_name;    // 函数名字符串（如"add"）
    // 其他：func_doc（文档字符串）、func_kwdefaults（关键字默认参数）等
} PyFunctionObject;
```

每次调用PyFunctionObject新建一个帧对象，加入线程的 “帧栈”，执行完后出栈，销毁或被垃圾回收。

```
typedef struct _PyFrameObject {
    PyObject_HEAD
    PyObject *f_back;       // 指向上一层帧（调用者的帧，如主程序帧）
    PyCodeObject *f_code;   // 指向对应的PyCodeObject
    PyObject *f_locals;     // 局部变量字典（如{'a':1, 'b':2, 'c':3}）
    PyObject *f_globals;    // 全局变量字典（复用函数对象的func_globals）
    PyObject *f_builtins;   // 内置变量字典（如__builtins__）
    Py_ssize_t f_lasti;     // 程序计数器：记录当前执行到的字节码位置
    int f_lineno;           // 当前执行的源代码行号
    PyObject *f_valuestack; // 求值栈（操作字节码时的临时栈，存储中间值）
    Py_ssize_t f_stacksize; // 求值栈的大小
    // 其他：f_closure（闭包的cell对象）、f_state（帧状态，如是否暂停）等
} PyFrameObject;
```

```
不带参数调用：vars() → 返回当前作用域的局部变量字典（等价于 locals()）；
带参数调用：vars(obj) → 返回对象 obj 的实例属性字典（等价于 obj.__dict__），前提是对象有 __dict__ 属性（大部分自定义类实例、命名空间对象都有）。
```

```
typedef struct _ts {
    struct _ts *next;
    PyFrameObject *frame;   // 当前正在执行的帧（帧栈顶）
    struct _frame *frame_stack; // 线程的帧栈（所有未执行完的帧）
    PyEvalFrameFunc eval_frame; // 帧执行器（核心函数，执行字节码）
    // 其他：全局解释器锁GIL、异常状态、线程ID等
} PyThreadState;
```

```
PyEvalFrameFunc是一个函数指针
接收 PyFrameObject 作为参数；
循环读取 f_code->co_code 中的字节码指令；
逐行执行指令，操作帧的 f_valuestack 和 f_locals；
遇到 RETURN_VALUE 时终止执行，返回结果。

进入 eval_frame 前必须获取 GIL，执行期间（默认）持有 GIL；遇到 I/O、睡眠等场景时，eval_frame 会临时释放 GIL，执行完后重新进入；
eval() 是 Python 的一个内置函数，它会将传入的字符串当作有效的 Python 表达式来执行，并返回结果。
```

局部变量：生存周期==PyFrameObject

闭包变量：嵌套函数中，内层函数引用的外层函数的局部变量，生存周期==内层函数

闭包变量本身是外层函数的局部变量，默认会随外层帧对象销毁。而 cell 对象的存在，就是为了 “把闭包变量从外层帧对象中抽离出来，封装成独立容器，绑定到内层函数对象上”。

即使多次调用内层函数，复用的是同一个 cell 对象里的闭包变量。

全局变量：生存周期==模块

```
解释器扫描到 `class ClassName(metaclass=Meta): ...`
编译类体代码 → 生成类体的 PyCodeObject
准备类创建的核心参数：类名(name)、基类(bases)、空的命名空间(attrs)
执行类体的 PyCodeObject → 把执行结果（变量、方法等）填充到 attrs 中
调用元类的 `__new__(cls, name, bases, attrs)` → 创建类对象
调用元类的 `__init__` → 初始化类对象
类对象创建完成，绑定到当前作用域的类名

metaclass默认是type，类的类
metaclass要带__call__: 使生成的实例可调用
```

```
MRO（Method Resolution Order）：即方法解析顺序，简单说就是 Python 查找类中方法 / 属性的 “优先级列表”。每个类都有 __mro__ 属性，可以直接查看这个列表。
super(type, obj) 的核心是调整方法查找起点：从 obj 的 MRO 列表中跳过 type，找后续第一个类型；
找到目标类型后，调用方法时会自动把 obj 传给该方法
需要精确匹配类型（不关心子类）    type(obj) == dict
兼容子类（如 defaultdict、OrderedDict）    isinstance(obj, dict)
```

```
当你执行 import module_name 或直接运行 python module.py 时：
解释器先检查 sys.modules 是否有 module_name？有，直接返回缓存的模块对象，不做后续操作；
解释器先读取模块的 .py 文件内容（源代码）；将整个模块的源代码编译成一个顶级的 PyCodeObject
创建空的模块对象（`module` 实例，`__dict__` 为空）；
将这个空模块对象立即加入 sys.modules；
调用求值循环（eval_frame）执行这个代码对象的字节码：
执行模块中的变量赋值（如 g = 100）→ 存入模块的全局字典（module.__dict__）；
执行函数定义（def func(): ...）→ 编译函数体生成函数级 PyCodeObject，再创建 PyFunctionObject 存入模块全局字典；
执行类定义（class Cls: ...）→ 编译类体生成类体级 PyCodeObject，执行后创建类对象存入模块全局字典；
PyCodeObject 执行后不会销毁，会被模块对象保留，且可通过 .pyc 文件缓存；
如果循环导入中依赖了未执行的内容，会直接报错。
```

## 模块

module查找：sys.modules，builtin（编译进解释器），sys.path（当前目录，PYTHONPATH，python安装路径，site-packages）

```
export PYTHONPATH=$root_path/paddle/Paddle/build/python:$PYTHONPATH
export PYTHONPATH=$root_path/paddle/PaddleNLP:$PYTHONPATH
```

只要模块的物理路径相同，Python 都会将其视为同一个模块​​。

主脚本 `A.py`的模块名是 `__main__`，而 `B.py`导入的是 `"A"`，所以 Python 认为 `"A"` 未被缓存，导致重新执行。

```
当使用 from module import * 时，Python 默认会导入模块中所有不以下划线（_）开头的名称。通过定义 __all__，可以显式指定哪些名称应该被导出，避免意外暴露内部变量或函数。
__all__ 仅作用于 from module import *。用户仍可通过 from module import _private_func 直接导入未列出的名称。
```

## ...

```
add = lambda x, y: x + y
sorted(tasks, key=lambda x: x["start_time"])

partial(model_provider, gpt_builder) 后，会得到一个新函数（比如叫 new_func）：
调用 new_func() 等价于调用 model_provider(gpt_builder)；
如果 model_provider 还需要其他参数，调用 new_func(参数1, 参数2) 等价于 model_provider(gpt_builder, 参数1, 参数2)。
```

装饰器

```
@functools.wraps(A)
B
#C=functools.wraps(A)
#B=C(B)
将B的一些属性改成A的，这里面一共4个函数
```

迭代器

```
# for x in obj: ... 的实际执行逻辑
_iterator = iter(obj)  # 调用 obj.__iter__()
while True:
    try:
        x = next(_iterator)  # 调用 _iterator.__next__()
    except StopIteration:
        break
```

生成器是实现迭代器的一种简便手段

调用生成器函数（带yield）或生成器表达式，会生成生成器对象，next(生成器对象)

```
itertools.chain：将多个可迭代对象（比如列表、元组、字符串、生成器等）“串联” 成一个连续的迭代器
map(function, iterable)：将第二个可迭代参数中的每个元素，依次传入第一个函数参数中执行，返回一个迭代器
filter(function, iterable)：用 function 对 iterable 中的每个元素做判断，只保留返回 True 的元素，返回一个包含符合条件元素的迭代器
def my_filter(function, iterable):
    if function is None:
        function = lambda x: bool(x)
 
    for item in iterable:
        if function(item):
            yield item
```

上下文

```
with 上下文对象 as 上下文对象.__enter()__返回值:
with退出前执行上下文对象.__exit__()
with中发生异常时，异常会传递到__exit__处理，若返回True，执行with之后到的语句，
否则报错

#简单实现
@contextmanager
def 上下文函数():
    ret=enter()
    try:
        yield ret
    except x:
        pass
    exit()
```

## pip

```
python -m venv paddle_env
pip install virtualenv
virtualenv -p /usr/bin/python3.10 paddle_env
source ./paddle_env/bin/activate
```

```
my_venv/
├── bin/          # 核心执行文件（Windows 是 Scripts/）
│   ├── python    # 指向当前 Python 版本的解释器副本/软链接
│   ├── pip       # 绑定到该环境的 pip 工具
│   └── activate  # 激活虚拟环境的脚本（关键）会把虚拟环境的 bin/ 目录添加到 PATH 的最前面
├── lib/          # 该环境的依赖包存放目录
│   └── pythonX.X/
│       └── site-packages/
```

网络：

pip install package 优先下载预编译 whl 包（无编译），仅在无匹配 whl 包时才会下载源码包并本地编译；少数 whl 包会包含 “部分源码”（比如 Cython 代码），安装时需要在本地做少量编译（称为 “就地编译”）；编译的核心目的是将包中的 C/C++/Cython 等扩展源码，转换成系统可执行的 .so/.pyd 二进制文件；

```
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install xxx -i https://www.paddlepaddle.org.cn/packages/nightly/cu118/
--extra-index-url http://xxx/simple
--trusted-host xxx 使用http需要加
```

```
pip install git+https://github.com/psf/requests.git
git+ 是 pip 的协议标识前缀，用于告诉 pip 该地址需要通过 Git 工具克隆仓库，而非普通 HTTP 下载；
```

本地：

```
pip install requests-2.31.0-py3-none-any.whl
pip install django-5.0.1.tar.gz
```

whl包括.py、.so、dist-info（依赖）

pip install时会安装依赖，pip uninstall不会移除依赖

```
pip show/pip list/pip freeze 的核心逻辑是：读取元数据文件（如 .dist-info/.egg-info 目录），这些文件记录了包的版本、路径、依赖等信息。
只要目录在 sys.path 中，Python 就能导入
pip show megatron 无结果是正常现象，因为 megatron 不是独立 pip 包，需用 pip show megatron-core 查看核心包信息；
import megatron 能成功，是因为 megatron-core 安装后生成了 megatron 命名空间目录，Python 可正常识别
```

## build

pip install .优先查找pyproject.toml，否则setup.py

```
[build-system]
requires = [
    "setuptools>=70.1.0",
    "cmake>=3.27",
]
build-backend = "setuptools.build_meta"
它让 pip 把构建控制权交给 setuptools，而 setuptools 会自动调用 setup.py 执行实际编译
```

setup.py指定.py，编译.c

```
setup.py
py_package/
csrc/
build：.py/.so
dist: whl
```

```
setup(
    name='lut_kernels',
    ext_modules=[
        cpp_extension.CUDAExtension(
            name='lut_kernels._CUDA',
            sources=[
                'csrc/lut_kernels.cu',
                'csrc/bindings.cpp',
            ],
            include_dirs=['csrc'],
            extra_link_args=['-lcublas', '-lcuda', '-lcudart'],
            extra_compile_args={'cxx': ['-std=c++17', '-O3'],
                                'nvcc': ['-O3', '-std=c++17', '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_HALF2_OPERATORS__', f'-DCUDA_ARCH={cuda_arch}', f'-arch=sm_{cuda_arch//10}']},
        ),
    ],
    cmdclass={
        'build_ext': cpp_extension.BuildExtension.with_options(use_ninja=False)
    },
    packages=find_packages(),
)
```

cmdclass 是 setuptools 中的一个核心参数，用于自定义 setup.py 中的打包 / 安装命令。

setuptools 读取 cmdclass 中的 build_ext 类，按照该类的规则，编译 ext_modules 中定义的扩展模块；

当你安装需要编译的包（比如包含 C/C++ 扩展的 Python 包，如 numpy、pytorch、psycopg2 等）时，pip 会默认开启 “构建隔离”：pip 会临时创建一个独立的虚拟环境（仅用于编译这个包）；从包的 pyproject.toml/setup.py 中读取编译所需的依赖（比如 setuptools、wheel、Cython 等）；在这个临时环境中安装这些编译依赖，然后用这个环境编译包；编译完成后，把包安装到你指定的主环境，再删除临时环境。

```
-e：
把本地 Megatron-LM 源码目录以 “可编辑模式” 安装到 Python 的 site-packages/dist-packages；
自动编译源码中的 C++/CUDA 扩展（包括 helpers_cpp.so），生成到本地源码的 megatron 目录中；
在 site-packages 中生成一个 *.egg-link 文件，指向本地源码目录 —— 此时 Python 导入 megatron 时，加载的是本地源码目录，但该目录已包含编译后的 helpers_cpp 扩展

pip install --no-build-isolation megatron-core[mlm,dev]
--no-build-isolation：这个参数主要是停止自动下载 / 安装 “编译构建所需的工具依赖”（如 setuptools、wheel、Cython 等），而非停止下载包的运行时依赖（如 requests 依赖 urllib3）。
安装 megatron-core 的 “基础套餐” + 同时加购 mlm 扩展套餐 + dev 开发套餐。

--ignore-installed 是 pip 的一个强制参数，作用是：忽略已安装的同名包（无论它是用什么工具装的），直接安装新版本，覆盖可覆盖的文件。
```

## pybind

Python 本身提供了一套 C 语言接口，pybind11 的底层就是调用这些 C API，但用 C++ 的特性（模板、RAII、lambda、类型推导）把繁琐的 C 风格代码封装得更易用。

py::def 的作用是将 C++ 函数（普通函数、lambda、类成员函数等）注册到 Python 模块中，让 Python 可以像调用原生函数一样调用它。

```
import example
print(example.add(1, 2))  

PYBIND11_MODULE(example, m) {
    // 基础注册：函数名（Python 中）、C++ 函数指针
    m.def("add", &add, "A function that adds two integers");
```

```
py::def 会为目标 C++ 函数生成一个符合 Python C API 规范的包装函数
PyObject* add_wrapper(PyObject* self, PyObject* args, PyObject* kwargs) {
    try {
        // 1. 解析 Python 参数（通过 pybind11 的类型转换器）
        int a, b;
        // pybind11 会根据函数签名自动解析 args/kwargs，替代原生 PyArg_ParseTuple
        if (!py::detail::parse_args(args, kwargs, "add", &a, &b)) {
            return nullptr; // 解析失败返回空，Python 会抛出 TypeError
        }
        
        // 2. 调用原始 C++ 函数
        int result = add(a, b);
        
        // 3. 将 C++ 返回值转为 Python 对象（类型转换器）
        return py::cast(result).release().ptr(); // 释放 py::object 所有权，返回 PyObject*
    } catch (const std::exception& e) {
        // 4. 异常桥接：C++ 异常转为 Python 异常
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}
py::def 最终会调用 Python C API，将生成的包装函数注册到模块对象中：
void def_impl(py::module_& m, const char* name, void* wrapper) {
    // 1. 构建 PyMethodDef 结构体（Python C API 要求）
    PyMethodDef method = {
        const_cast<char*>(name),          // 函数名（Python 中可见）
        (PyCFunction) wrapper,            // 包装函数指针
        METH_VARARGS | METH_KEYWORDS,     // 支持位置+关键字参数
        const_cast<char*>(docstring)      // 函数文档字符串
    };
    
    // 2. 将函数添加到模块的方法表中
    PyModule_AddObject(m.ptr(), name, PyCFunction_New(&method, nullptr));
}
```

py::class_ 是 pybind11 提供的模板类，核心作用是将 C++ 类（包括普通类、模板类、继承类、抽象类等）完整映射为 Python 中的类

```
PYBIND11_MODULE(example, m) {
    // 1. 定义 Python 类：模板参数 <C++类名, 父类（无则省略）>
    py::class_<Person>(m, "Person", "A class representing a person")
        // 2. 注册构造函数（支持多个重载构造函数）
        .def(py::init<const std::string&, int>(), 
             py::arg("name"), py::arg("age"), 
             "Create a Person object with name and age")
```

```
PyTypeObject* create_python_type() {
    static PyTypeObject type = {
```

内存嵌入，C++ 对象的生命周期完全由 Python 对象的引用计数决定，无法单独销毁 C++ 对象。

```
struct PyPersonObject {
    PyObject_HEAD
    Person cpp_obj; // 直接嵌入 C++ 对象
};
PyObject* person_new(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
    PyPersonObject* self = (PyPersonObject*) type->tp_alloc(type, 0);
    new (&self->cpp_obj) Person(name, age);
    return (PyObject*) self;
}
void person_dealloc(PyPersonObject* self) {
    self->cpp_obj.~Person();
    Py_TYPE(self)->tp_free((PyObject*) self);
}
```

Python 对象中仅存储 C++ 对象的指针（裸指针 /shared_ptr）

```
创建：Python 对象分配内存时，仅存储 C++ 对象的指针，不构造 C++ 对象（C++ 对象已提前创建）；
销毁：Python 对象销毁时，根据「生命周期策略」决定是否释放 C++ 对象；
```

```
automatic（默认，适配 shared_ptr）
Python 创建对象 → shared_ptr 引用计数 +1；
Python 销毁对象 → shared_ptr 引用计数 -1；

take_ownership
Python 对象存储裸指针 → 销毁时调用 delete 释放 C++ 对象

reference
Python 仅存储裸指针，不负责销毁
```

```
PYBIND11_MODULE(libpaddle, m) {
生成一个符合 Python C API 标准的模块初始化函数 PyInit_libpaddle()。

python可以import .so文件：
解释器加载动态库（如 libpaddle.so）
Python 查找并解析 .so 文件中的 PyInit_<module> 函数
调用模块初始化函数（如 PyInit_libpaddle()）
将生成的 PyModuleObject 加入 sys.modules
```

```
在cpp中可以直接执行python的函数：
PyObject_Call调用函数对象的tp_call
    为这次调用创建一个新的帧对象
    _PyEval_EvalFrameDefault(frame)
```

## argparse

```
parser = argparse.ArgumentParser()
parser.add_argument('--xxx', type=type_func, default=xxx, 
                       help='xxx')
parser.add_argument("-o", "--output",
                     nargs='?',          # 参数可选，可以不带值
                     default="result.txt",  # 不提供 -o 时，默认值
                     const="result.txt",    # 提供 -o 但不给值时，使用该值
                     help="可选,若不指定则为result.txt"
                   )     
#下划线和连字符可互换
assert callable(type_func)==True
当用户输入命令行参数时，所有参数最初都是字符串。type参数指定的函数会被调用，将这个字符串转换为目标类型
arg_group = parser.add_argument_group(title=title, description=self.src_cfg_class.__doc__)
arg_group.add_argument(*arg_names, **add_arg_kwargs)
#注册形参
args = parser.parse_args()
#字符串实参转parser形参
```

## ast

```
source = inspect.getsource(src_cfg_class)
通过传入的类对象（TrainingConfig），找到它对应的源代码文件路径；
打开这个文件，读取文件中定义该类的那一段原始文本；
返回这段文本（保留所有手写内容，包括装饰器、注释、缩进）。
# AST 解析时会直接忽略 # 注释，不会生成任何节点
tree = ast.parse(source)
root_node = tree.body[0]
#第一个顶级语法单元。
a_cond = isinstance(a, ast.AnnAssign) and isinstance(a.target, ast.Name)
#带注解的赋值语句，赋值语句的目标，标识符
b_cond = isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant)
#独立表达式（如文档字符串），表达式的值，常量
isinstance(a.target.id, str)
isinstance(b.value.value, str)
```

## asyncio

单线程

调用异步函数（async def+return）生成协程对象

```
async def my_coro():
    print("start")
    await fut  # 暂停 = yield
    print("resume")

c = my_coro()
c.send(None)  # 启动
c.send(None)  # 恢复
```

```
Future 
标记状态：等待中 / 已完成 / 失败
存储结果 / 异常
fut = asyncio.Future()  # pending
fut.set_result(123)     # 变成 done
存储完成回调

#await Future
if fut.done():
    pass
else:
    yield fut
    
#await 协程对象
直接执行
```

```
create_task 只是把任务加入事件循环的就绪队列，等待wakeup，然后立刻继续往下执行当前代码。
class Task(Future):  
def _wakeup(self):
    self.coro.send(None) 
    协程暂停，返回 Future 给 Task
    Task 给 Future 加回调：done → _wakeup(self)
	return #回到event loop
```

3种回调：future、定时器、IO

```
async def sleep(time):
    fut = loop.create_future()
    loop.call_later(time, fut.set_result, None)#创建asyncio定时器
    #定时器堆.append((到期时间, callback))
    await fut 
```

```
async def read_from_socket(sock):
    fut = asyncio.Future()
    def on_readable():
        data = sock.recv(1024)
        fut.set_result(data)          
    loop = asyncio.get_running_loop()
    loop.add_reader(sock.fileno(), on_readable)
    #把这个 socket 加入操作系统的监听列表（epoll / kqueue）
	#记录映射关系：socket 可读 → 执行 on_readable
    data = await fut
    loop.remove_reader(sock.fileno())
    return data
```

```
while True:
    # ==========================
    # 阶段 A：先把所有“已经就绪”的东西跑完
    # ==========================
    1. 执行所有已经到期的定时器回调（来自 call_later）
       → 这里面就包含了 sleep 对应的 fut.set_result()

    2. 执行【所有已 done 的 Future 回调】
       → 这些回调的内容 = 把对应的 Task 放进就绪队列！

    3. 执行就绪队列里的所有 Task！
       → 这里才调用 task._wakeup()
       → wakeup 里执行 coro.send() → 恢复协程

    # ==========================
    # 阶段 B：没事干了，开始阻塞等待
    # ==========================
    算出最近一个定时器还剩多久 → 超时时间
    events = epoll_wait(所有注册的fd, 超时=超时时间)

    # ======================
    # 阶段 C：处理 IO 事件
    # ======================
    for fd in events:
        run_callback(fd)  # <-- add_reader 的回调在这里执行
```

await asyncio.gather等待多个task完成

asyncio.Queue() 的 put()/get() 是协程函数，空和满会挂起，put_nowait/get_nowait 不是协程函数

```
异步迭代器 = 实现了 __aiter__ 和 __anext__ 方法的对象
__anext__ 是一个协程函数，可以被await挂起
async for 异步迭代器 == await __anext__()，不挂起就获取结果
```

调用异步生成函数（async def+yeild）生成异步生成器，异步生成器本身不能被 await，async for每次调用anext()，异步生成器每次运行到yield

```
def make_async(func: Callable[P, T]) -> Callable[P, Awaitable[T]]:
    """Take a blocking function, and run it on in an executor thread.

    This function prevents the blocking function from blocking the
    asyncio event loop.
    The code in this function needs to be thread safe.
    """

    def _async_wrapper(*args: P.args, **kwargs: P.kwargs) -> asyncio.Future:
        loop = asyncio.get_event_loop()
        p_func = partial(func, *args, **kwargs)
        return loop.run_in_executor(executor=None, func=p_func)

    return _async_wrapper
```



## dataclasses

```
from dataclasses import dataclass, field, fields
当用 @dataclass 装饰一个类时，类中直接定义的变量会自动转换为实例属性，所有字段默认会包含在 __init__ 方法中。
如果需要对某些属性进行控制，就需要用 field() 来标记。
@dataclass
class SimpleExample:
    sequence_parallel_nccl_config: Optional[Dict[str, Dict[str, int]]] = field(
        default=None,
        metadata={"help": "sequence parallel nccl config",},
    )
    #Optional 指 None，不会影响默认值，类型注解在运行时会被忽略
field_info = fields(SimpleExample)[0]  # 获取第一个字段
print("原始 field.type：", field_info.type)
# 输出：原始 field.type： Optional[Dict[str, Dict[str, int]]]
# 注意：这里是字符串形式，不是实际的类型对象

type_hints = get_type_hints(SimpleExample)  # 提取类的类型提示（解析后的）
print("解析后的类型：", type_hints["sequence_parallel_nccl_config"])
# 输出：解析后的类型： typing.Optional[typing.Dict[str, typing.Dict[str, int]]]
```

## logging

```
logger = logging.getLogger("strobelight_function_profiler")

console_handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(name)s, line %(lineno)d, %(asctime)s, %(levelname)s: %(message)s"
)
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.setLevel(logging.INFO)
logger.propagate = False
#阻止当前 logger 的日志消息向上传递给父级 logger
```

根 logger 是最顶层（logging.getLogger() 无参数时获取）；

子 logger 按 “点分隔” 命名（比如 logging.getLogger("a.b") 是 logging.getLogger("a") 的子级，a 又是根 logger 的子级）；

默认情况下，子 logger 输出日志时，会把日志消息向上传播给父级 logger，父级再按自己的配置处理（比如输出到控制台 / 文件）。

warnings 模块发出的警告，默认输出到标准错误流（stderr）

## multitask

```
thread创建子线程，执行python代码
multiprocessing创建子进程，执行python代码
subprocess创建进程
concurrent提供高级的线程/进程池接口（基于 ThreadPoolExecutor和 ProcessPoolExecutor）
```

## pandas

Cython 是 Python 的超集（语法兼容 Python），Cython 新增了一套语法，允许你直接声明 C 类型（如 cdef int a）、调用 C 函数（如 from libc.stdio cimport printf）、定义 C 风格的函数（cdef 函数），Cython 本身不直接运行，而是把 .pyx 文件编译成 C/C++ 代码，再通过 C 编译器（如 gcc）编译成 .so（Linux）/.pyd（Windows）二进制扩展模块，供 Python 调用。

```
纯 Python 语法用 Python C API 模拟 Python 解释器的执行逻辑，性能几乎和原生 Python 一样
扩展语法转换为原生 C 逻辑
```

Pandas 本身不直接写大量纯 C 代码，而是通过 Cython 作为 “桥梁” 封装 / 调用 C 代码（包括 NumPy 的 C 接口），numpy 核心是纯 C 写的

Series: 一维数组。它由两部分组成：索引 (Index) 和 数值 (Values)。

DataFrame: 二维表格型数据结构。它由三部分组成：行索引 (Index)、列名 (Columns) 和 数值 (Values)。

如果你从 DataFrame 中提取单独的一列，返回的对象就是一个 Series。

```
s = pd.Series(len_list).value_counts(normalize=True).sort_index()

csv文件类似数据库：
姓名,年龄,成绩
张三,20,85
李四,22,92
df = pd.read_csv('data.csv')

行
data = [
    {'姓名': '张三', '年龄': 25, '城市': '北京'},
    {'姓名': '李四', '年龄': 30, '城市': '上海'},
    {'姓名': '王五', '年龄': 28, '城市': '广州'}
]
df = pd.DataFrame(data)

列
data = {
    '姓名': ['张三', '李四', '王五'],
    '年龄': [25, 30, 28],
    '城市': ['北京', '上海', '广州']
}
df = pd.DataFrame(data)
```

## pdb

pdb 是解释器的 “插件”，被调试程序和 pdb 运行在同一个进程中

你用 b 行号/函数名 设置断点时，pdb 会先解析源码，找到断点对应的行号和字节码位置；通过 sys.settrace() 注册的跟踪函数，会在执行每一行代码前检查 “断点表”：如果当前行是断点且启用，则暂停执行，进入 pdb 交互模式。

单步执行：跟踪函数会设置 “仅执行一行代码” 的标记，解释器执行完当前行的字节码后，立即触发暂停，回到 pdb 交互界面；

```
import pdb
import socket

def pre_pdb(local_rank, port, addr='127.0.0.1'):
    port += local_rank
    handle_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    handle_socket.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,True)
    handle_socket.bind((addr,port))
    handle_socket.listen(1)
    print("pdb rank %d %s:%d" % (local_rank,addr,port))
    (client,address) = handle_socket.accept()
    handle = client.makefile('rw')
    return handle
```

```
if args.local_rank==0:
    handle = pre_pdb(args.local_rank,16890)
    pdb.Pdb(stdin=handle,stdout=handle).set_trace()

apt install -y netcat
nc localhost 16890
```

## plot

2D = 平面画画；3D = 空间建模 + 拍照

矢量图形：用数学公式、坐标、几何描述来定义的图形

AGG 是 2D 矢量图形渲染库，FreeType 是字体光栅化库；AGG 不内置字体解析，通过封装 FreeType 实现高质量字体渲染

AGG 中文字 = 特殊矢量路径，由 FreeType 解析字体生成，和几何图形无本质区别；

矢量路径 → 光栅化（几何 → 像素位置）→ 渲染（填色到画布）→ 最终图像

```
// 第一步：渲染图形（矩形）
ras.reset(); // 清空光栅化器缓存
ras.add_path(rect_path); // 提交图形路径
render_scanlines_aa_solid(ras, sl, ren_base, 红色); // 渲染图形

// 第二步：渲染文字
ras.reset(); // 再次清空光栅化器
ras.add_path(text_path); // 提交文字路径
render_scanlines_aa_solid(ras, sl, ren_base, 黑色); // 渲染文字
```

pyplot show()、save(png/jpg)依靠AGG，save(pdf)直接保存「折线、文字、坐标轴」的矢量描述

```
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']  
plt.rcParams['axes.unicode_minus'] = False  
中文字体

fig_la, axs_la = plt.subplots(len(model_rate),2,figsize=(8, 5),width_ratios=[3,1])
figure代表整图，axs代表子图

axs_la[idx,0].plot(bs, la, label=f"{name}",color=color_map[color_id])
画线
bars=axs_la[idx,1].bar([0],[qua_qua_t["smooth"]["lut"]],width=0.8,edgecolor=color_map[0],facecolor='white',hatch="...")
画柱状图
axs_la[idx,1].text(bar.get_x() + bar.get_width()/2, min(y_max,yval+y_max*0.14), round(yval,1), ha='center', va='top', fontsize=10)
写字
plt.axvline(mean_len, color='red', linestyle='--', linewidth=2, label=f'平均长度: {mean_len:.2f}')
一条竖线

ax2.legend(bbox_to_anchor=(0.45, 1.0), fontsize=8)
legend将已定义的label汇总为一个图例框，显示在图表中
axs_th[idx].grid(True)
axs_th[idx].grid(axis='y', linestyle='--', alpha=0.7)
axs_th[idx].set_xticks(list(range(2,max_bs+1,2)))
axs_la[idx,1].set_xticklabels(["LU", "WD", "AQ+AD"])
axs_th[idx].set_xlabel('Batch Size')
y_max = axs_th[idx].get_ylim()[1]
axs_la.set_yticks(np.linspace(0, ymax, 10))
axs_la.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
axs_la[idx,0].tick_params(axis='y', labelsize=12) 
axs_th[idx].set_ylabel('Tokens/Millisecond')
axs_th[idx].set_title(f"{model_name[model]}")

plt.tight_layout()
fig_la.subplots_adjust(left=0.06, right=0.94, bottom=0.1, top=0.95, wspace=0.3, hspace=1)
会覆盖tight_layout

fig_la.savefig(latency_path)
```

## queue

线程安全

```
Queue()队列无限大
Queue.put(item, block=True, timeout=None)
阻塞最多 timeout 秒并会在没有可用的空闲槽位时引发 Full 异常
Queue.put_nowait(item) 相当于 put(item, block=False)
```

## re

```
*0或多
+1或多
?0或1
? 跟在 * 或 + 后面时，表示尽可能少地匹配字符（非贪婪模式）
(?:MemOpt)非捕获组
(?P<name>xxx)命名组

r''取消python转义\，\留给re转义，否则要加\\

NUM_PATTERN = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'
time_pattern = re.compile(
    r'.*?gpu_task_manager.cc.*?'
    r'(?P<nccl_id>nccl_ids/\d+/\d+(?:->\d+)?),'
    r'(?P<seq>\d+),'
    r'(?P<comm_type>\d+),'
    r'(?P<numel>\d+),'
    r'(?P<start_time>{num}),'
    r'(?P<end_time>{num})'
    r'(?:,(?P<use_calc_stream>\d+))?' 
    r'(?:,(?P<end_time_start_time>{num}))?'
    r'(?:,(?P<elapsed_ms>{num}))?'
    r'(?:,(?P<ratio>{num}))?'.format(num=NUM_PATTERN)
)

init_match = initRank_start_pattern.search(line)
if init_match:
    rank0_comm = init_match.group(1)

match = re.search(r'cudaDev (\d+)', line)

group(0)返回整个匹配的字符串
group(1)返回第1组
```

## sys

```
print("全部参数：", sys.argv)
print("脚本名：", sys.argv[0])
```

## typing

```
from typing import TYPE_CHECKING 运行时永远是 False，但类型检查工具（如 mypy）会把它当作 True，常用于解决「导入循环」或「仅类型检查需要的导入」

在定义类或函数时，如果类型注解中引用了尚未定义的类，直接写类名会导致运行时错误
通过 from __future__ import annotations 自动将所有注解转换为字符串(前向引用)

from typing import get_type_hints
从函数、类或模块中提取类型提示。
def user_info(name: str, age: int) -> Dict[str, List[str]]:
hints = get_type_hints(user_info)
{'name': <class 'str'>, 'age': <class 'int'>, 'return': typing.Dict[str, typing.List[str]]}

isinstance(A, type) 的作用
如果 True：说明 A 是一个普通的 Python 类型（如 int、str）。
如果 False：说明 A 可能是泛型、前向引用等。

lst_type = List[int]
print(lst_type.__origin__)  # <class 'list'>（原始容器类型）
print(lst_type.__args__)    # (<class 'int'>,)（泛型参数）
union_type = Union[int, str]
print(union_type.__origin__) # typing.Union
print(union_type.__args__)   # (<class 'int'>, <class 'str'>)（Union包含的所有类型）
```

# shellscript

Shell 所有变量都是字符串

-gt -ge -lt -le -eq -ne 这些都是专门给整数用的比较符，Shell 会自动把两边的数字字符串转成整数比较

```
manage_types=("std" "space" "pre_swap_out")
for manage_type in "${manage_types[@]}"; do
    while true; do
        if [ "$count" -ge "$TARGET_COUNT" ]; then
            break
        fi
        sleep 5
    done
done

${manage_types[0]} → 只取第 1 个：std
${manage_types[@]} → 取全部：std space pre_swap_out

for ((M=32; M<=512; M*=2)); do
    for ((K=32; K<=512; K*=2)); do
        for ((N=32; N<=512; N*=2)); do
            ./a.out "$M" "$K" "$N"
        done
    done
done

(( ... ))表示算术上下文，里面可以写整数运算、比较、递增、乘法赋值等
```

```
'...'   几乎完全原样，不展开变量，不执行命令
"..."   保留空格，但会展开变量和命令替换
$(...)  命令替换：执行命令，把输出放回来
`...`   老式命令替换
```

把中间那几行文本写入到 $config_json 这个变量指向的文件里
```
cat <<EOF >"$config_json"
{
"model":2
}
EOF
```

# javascript

除了基本类型，都是对象

没有类，只有原型，每个对象都有proto域指向另一个对象，可以把它当成父对象

Google的v8引擎：js编译成字节码，解释执行，热点字节码编译成机器码

v8是cpp实现，用于浏览器、node.js

解析HTML和执行js在一个线程

```
解析 HTML
处理脚本 A：
    执行同步代码，打印 A1: 同步。
    将 A2 加入微任务队列，A3 加入宏任务队列。
    脚本 A 同步代码结束。
清空微任务（脚本 A 产生）：
	打印 A2: 微任务。
继续解析 HTML
处理脚本 B：
    执行同步代码，打印 B1: 同步。
    将 B2 加入微任务队列。
    脚本 B 同步代码结束。
清空微任务（脚本 B 产生）：
	打印 B2: 微任务。
事件循环后续阶段：
	打印 A3: 宏任务。
```

浏览器是增量渲染的，只要当前 DOM 树和 CSSOM（CSS 树）结合后能计算出一些内容，浏览器就会尝试把这部分内容画到屏幕上。

# latex

pdfLaTeX：把 LaTeX 代码编译成 PDF 的程序，传统引擎，不支持中文

XeLaTeX：现代、支持 Unicode、支持中文的编译器

你的 LaTeX 代码 → pdfLaTeX 解析成 TeX 底层指令 → TeX 引擎执行排版计算 → pdfLaTeX 把结果直接输出为 PDF

TeX Live：一整套完整环境：引擎 + 编译器 + 宏包 + 工具

tlmgr 是 TeX Live 的官方包管理器，核心作用是安装 / 更新 / 卸载 LaTeX 宏包；

```
# 步骤1：第一次编译，生成参考文献引用标记
xelatex -synctex=1 -interaction=nonstopmode -file-line-error test.tex
# 步骤2：用 bibtex 解析 .bib 文件，生成参考文献中间文件
bibtex test
# 步骤3：第二次编译，把参考文献插入文档
xelatex -synctex=1 -interaction=nonstopmode -file-line-error test.tex
# 步骤4：第三次编译，修正参考文献的引用编号
xelatex -synctex=1 -interaction=nonstopmode -file-line-error test.tex
```

# conda

anaconda distribution: 软件包合集+conda(默认从defaults下载)

miniconda: conda(默认从defaults下载)

anaconda.org: defaults(收费), conda-forge(社区版)

miniforge: conda(默认从conda-forge下载), mamba(cpp版本)

可以管理不同版本gcc

所有通过 conda install安装的包（包括依赖库）的原始文件都存放在 pkgs/下，多个 Conda 环境可以共享 pkgs/中的同一包文件

当创建 Conda 环境时，lib/ bin/会包含硬链接，指向 pkgs/中的实际文件

```
/opt/conda/   
├── bin/python3.10
├── lib/              
│   ├── libcudart.so.12 
│   └── python3.10/
|            ├──site-packages
|					├── torch/
└── pkgs/
```

通过conda install，PyTorch GPU 版本依赖 pytorch-cuda，后者需要 cuda-runtime，而 cuda-runtime 又依赖 cuda-cudart，CUDA 运行时库会被安装到环境的全局 lib/ 目录下

```
/opt/conda/envs/myenv/    
├── bin/python3.10
├── lib/              
│   ├── libcudart.so.12 
│   └── python3.10/
|            ├──site-packages
|					├── torch/   
```

```
PyTorch 的 Conda 包在构建时主动包含 Pip 格式的元数据，pip show 有效
```

通过pip install，libcudart会安装在torch/lib

```
/opt/conda/envs/myenv/              
├── bin/python3.10                  
├── lib/python3.10/
			 ├──site-packages/           
                    ├── torch/                
                    │   ├── __init__.py       
                    │   ├── lib/              
                    │   │    ├── libcudart.so.12
```

# protobuf

Protobuf，全称 Protocol Buffers，是 Google 的一种数据序列化机制。它用 .proto 文件定义数据结构和服务接口，可以生成不同语言的代码，比如 Go、Java、Python、C++。

```
message GetUserRequest {
  int32 id = 1;
}
message User {
  int32 id = 1;
  string name = 2;
}
service UserService {
  rpc GetUser (GetUserRequest) returns (User);
  rpc CreateUser (User) returns (User);
}
//service = 一组 RPC 方法的集合
```

Protobuf 生成的是 message 相关代码：GetUserRequest和User的序列化方法、反序列化方法、字段访问方法

gRPC 会在 Protobuf message 代码基础上，再根据 service 生成远程调用相关代码 (基于 HTTP/2)

```
lis, err := net.Listen("tcp", ":50051")
err = grpcServer.Serve(lis)

conn, err := grpc.Dial("localhost:50051", ...)
client := pb.NewUserServiceClient(conn)
resp, err := client.GetUser(...)
```

# vscode

```
my-language-plugin/
├── package.json                # 插件配置（声明高亮+LSP贡献点）
├── syntaxes/
│   └── mylang.tmLanguage.json  # TextMate 语法规则
├── server/                     # Language Server 代码/可执行文件
│   ├── out/
│   │   └── server.js           # Server 逻辑代码
│   └── package.json
└── extension.js                # 插件主逻辑（启动/管理 Server）
```

语法高亮的核心是 TextMate 正则规则：通过精准的正则匹配不同语法元素，给每个元素打上唯一的 scope 标签；

高亮的呈现依赖主题样式映射：主题文件将 scope 标签与颜色 / 字体样式绑定，最终渲染到编辑器；

color-theme：tokenColors 控制「代码文本的语法高亮」（内容层），colors 控制「编辑器界面的配色」（UI 层）；

编辑器侧（客户端）：只负责 UI 交互（触发请求、渲染结果）、文本管理；

语言服务器侧：只负责代码分析（AST 生成、语义检查、补全 / 跳转计算）；

二者通过 LSP（Language Server Protocol） 通信（基于 JSON-RPC 2.0 的标准化协议）

「全量同步」：客户端首次打开文件时，发送完整文件内容给 Server；

「增量同步」：客户端只发送文件修改的部分（如某几行的增删改），Server 局部更新，避免重复传输整个文件（性能优化核心）；

remote-ssh: 本地 VS Code（客户端） ↔ 远程 VS Code Server（服务端），通过 SSH 隧道传输数据

先检查远程服务器是否已安装 VS Code Server，通过 SSH 将压缩包传输到远程服务器，在远程服务器上启动 VS Code Server 进程

```
Error: LocalDownloadFailed (Failed to download VS Code Server (Failed to fetch))

Remote.SSH: Local Server Download
Whether the extension can download the VS Code Server on the client and transfer it to the host with scp, instead of downloading it on the host.
```

```
版本：help->about
Ctrl+Shift+P 包含了操作 VS Code 的所有指令
Ctrl+, 是打开图形化设置界面的快捷方式，settings.json 是所有配置的最终存储文件
word wrap
Scroll Editor With Preview
```

