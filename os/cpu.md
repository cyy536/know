# 指令集架构

CISC：x86（闭源）

RISC：RISC-V（开源），ARM（付费），MIPS（衰落）

为防断供或垄断，芯片厂商（Intel）常被要求授权另一家公司（AMD）生产兼容芯片

Intel 与 AMD 的 CPU「实现」（即微架构）完全不同，但它们都遵循同一个「x86 指令集架构」规范

x86的汇编语法有两种：Intel 和 AT&T（Unix）

非 x86 架构通常只有一种主流汇编语法

# 汇编

字：能够由cpu一次完成处理的数据

32位cpu指的是字长32位，通用寄存器的大小和字长相同

汇编指令和机器指令一一对应

伪指令是给汇编器看的

```
assume cs:xxx
//声明段类型，cpu不会给cs赋值，需要手动赋值
xxx segment
...
xxx ends
//定义一个段
xxx:
...
end xxx
//执行入口，默认第一条指令
mov cx, 5
xxx:
...
loop xxx
//loop是cpu指令
dw xxx,...
//声明字(16bit)类型，数据和指令可以混合放在一个段，但分开更规范
//栈段和数据段没有实际差别
db 200 dup(0)
//200个字节都是0
```

三种中断：除0中断，int，外设中断

CPU加电后跳转到BIOS程序，BIOS会在内存地址0设置中断向量表（函数表），跳转中断处理程序由CPU自动完成，iret返回

CPU通过端口（寄存器）和外设进行联系

键盘内置芯片会扫描按键情况，扫描码会写到端口，int 9会读取扫描码，写入键盘缓冲区

int 16h可读取键盘缓冲区，int 13h可读写磁盘

BIOS被直接跳转，Bootloader被BIOS加载，内核被Bootloader加载，可执行文件被操作系统加载，它们的格式不同

编译器和链接器本质上是文件处理程序，算法本身（解析、优化、链接）可以跨平台实现，输出结果必须严格遵守目标平台的二进制规范，与宿主平台无关

第一个汇编器：设计汇编语言，用汇编语言编写汇编器，手工处理成机器码

第一个 C 编译器是用汇编写的，可以用C写C编译器，用初始编译器编译新版 C 编译器

C 是“平台无关”的高级语言，平台相关的操作必须由汇编表达，例如C没有int的语法，无法实现write()，只能内联汇编，使用系统调用需要链接系统库

标准库是由编程语言标准（如 ISO C、ISO C++） 定义的库，所有符合标准的实现都必须提供

系统库是由操作系统或POSIX 等系统标准提供的库，被语言标准库底层使用

标准库和系统库合一非常常见，是主流做法

glibc = ISO C 标准库的实现 + POSIX 系统接口的实现 + GNU 扩展

内核不依赖glibc，操作系统的用户态组件依赖，如 init、bash、sshd、systemd

操作系统的用户态组件和你自己写的程序，在 CPU 权限级别上没有区别，它们是不需要写在内核里的系统功能，以 root 或特权用户运行

内核在执行系统调用时，会查看当前正在执行该系统调用的进程所关联的用户身份信息

# 实模式

“实模式”、“保护模式”和“长模式”这些术语是 x86 架构所独有的，而“特权级别”是所有现代 CPU 的通用安全机制。

实模式：物理地址 = 段寄存器 × 16 + 偏移

16 位 CPU，1 MB 内存，寄存器是 16 位的，直接寻址能力只有 64 KB，不能访问 1MB 内存，用两个寄存器

保护模式下程序员使用的地址称为逻辑地址（无论是否分页）

```
mov eax, [0x0804a01c]   ; ← 默认使用 DS
mov ebx, [esp + 4]      ; ← 默认使用 SS

段选择子：存放在段寄存器（CS, DS, SS 等）中
┌──────────────┬──────┬──────┐
|    Index     |  TI  | RPL  |
└──────────────┴──────┴──────┘
  13 位         1 位   2 位
Index：GDT/LDT 中的表项索引
TI：0 = GDT，1 = LDT
RPL：请求特权级（0–3）
```

GDT/LDT 中的表项

```
段基地址（32 位）：段的起始物理地址
段限长（20 位）
特权级（DPL, 2 位）
```

线性地址 = 段描述符.Base + 偏移地址

启用分页：

设置 CR0.PG = 1（Paging Enable），MMU 开始使用 CR3 指向的页表进行线性地址翻译

用高 10 位索引页目录 → 得到页表物理地址

用中间 10 位索引页表 → 得到页帧物理地址

用低 12 位作为页内偏移 → 物理地址 = 页帧地址 + 偏移

在启用分页的现代 OS，所有段（CS/DS/SS/ES）都设为 flat 段

```
Base = 0
Limit = 4 GB
DPL = 0（内核）或 3（用户）
```

逻辑地址 = 偏移，线性地址 = 偏移

C 语言程序中看到的所有地址（指针值、变量地址、函数地址等），也就是虚拟地址，在数值上就等于汇编中的“偏移”

每个进程有独立的用户页表，但共享内核页表

长模式是 AMD 为 x86 添加 64 位能力时设计的，保留对 32/16 位程序的兼容，切换到长模式必须要开启分页

可以从实模式直接切换到长模式，也可以从保护模式切换长模式

支持 48 位虚拟地址：如果支持完整 64 位虚拟地址，页表和 TLB 的开销会剧增

# 特权

Ring 0：最高特权。操作系统内核、驱动程序运行在此。

Ring 3：最低特权。用户态应用程序运行在此。

CPU 在地址翻译时会检查当前特权级，内核空间的页在页表中被标记为 U/S = 0

Ring 3 不能使用特殊指令和寄存器

从 Ring 3 切换到 Ring 0：通过系统调用（如 syscall / int 0x80），由 CPU 自动完成上下文切换。

不同系统调用有不同的系统调用号。系统调用号，参数，返回值通过寄存器传递

int $0x80的中断处理程序是system_call

# 缓存-锁

3级缓存，l3多核心共享

```
I（Invalid） 是“未使用”或“已失效”状态；
E（Exclusive） 是“干净 + 独占”，可无通知地升级为 M；
M（Modified） 是“脏 + 独占”，是写操作后的典型状态；
S（Shared） 是“干净 + 共享”，允许多读；
O（Owned） 是“脏 + 共享”，用于避免频繁写回；
```

所有核心通过缓存一致性协议（如 MESI、MOESI）组成一个连贯的缓存系统。当一个核心修改了 L1（或 L2）缓存中的数据不需要先写回主内存（DRAM）；其他核心在读取同一地址时，会通过缓存一致性协议，直接从修改核心的缓存中“获取最新值”；主内存（DRAM）可能长期保持旧值，直到该缓存行被驱逐（evict）时才写回（write-back）。

缓存一致性协议由硬件自动实现，确保对同一个缓存行（Cache Line）的修改，在所有核心间最终一致，不保证多个地址的操作顺序被其他核心以相同顺序观察到

原因：写入缓存的顺序不等于指令执行顺序，Store Buffer 导致不同地址的写操作以不同延迟提交到缓存，而弱内存模型允许这种“可见性重排序”。

Core 0 执行 x = 1;（x 缓存状态为 Shared，y 缓存状态为 Exclusive）；Core 0 将 (地址=x, 值=1) 放入自己的 Store Buffer；同时向其他核心发送 Invalidate 消息（请求使 x 的缓存副本失效）；不等待响应，立即执行下一条指令（y = 1，无需 Invalidate，立即写入缓存）；当收到所有核心的 ACK 后，CPU 才将 Store Buffer 中的 x=1 写入本地 L1 缓存，状态变为 Modified；此时，该写操作才真正对其他核心可见，在此之前，其他核心读 x 仍然会看到旧值，即使 Core 0 的程序已经“执行完”了 x = 1

ARM 等弱内存模型把控制权交给程序员：需要顺序时加屏障，不需要时尽情并行

x86 通过硬件机制，强制实现了“顺序可见性”；必须等 x=1 完成 Invalidate、写入缓存后，y=1 才被允许提交；其他核一定先看到 x=1，再看到 y=1。

x86 可以乱序执行 x=1; y=1; 但对外可见顺序 = 编译器重排后的顺序。两条 store 指令的地址计算、写入 Store Buffer 等执行步骤可以乱序进行，以提升性能。

x86 只允许 StoreLoad 重排序（即写操作之后的读操作可能提前执行）。

内存屏障阻止某些类型的内存操作重排序，CPU 架构都提供显式的内存屏障指令

写屏障（Store Fence）强制 Store Buffer 中的前序写操作全部提交到缓存后，才允许后续写入 Store Buffer

```
[应用程序]
    ↓ 使用 std::atomic_thread_fence(memory_order_release)
[编译器 (GCC/Clang)]
    ↓ compiler barrier + (可能的) 硬件指令
    → x86: 可能无指令
    → ARM: 生成 dmb st
```

Compiler barrier（编译器屏障）是一种仅作用于编译阶段的内存屏障，它的唯一作用是阻止编译器对内存访问进行重排序或优化，但不会生成任何硬件指令

CPU 都提供不可中断的读-修改-写（RMW）指令，这是原子操作的基础，无需进入内核

```
x.store(1, std::memory_order_release);
```

C++ 标准库的 std::atomic 由编译器替换成原子指令 + 内存屏障，不是函数调用，不需要链接库

volatile 是 C 和 C++ 中的一个类型限定符（type qualifier）

volatile 保证：每次读取都从内存加载（不使用寄存器缓存）；每次写入都立即存到内存（不合并、不重排、不优化掉）；访问顺序与源码一致（编译器不会对两个 volatile 变量的访问进行重排）。

所有锁的实现都基于硬件提供的原子操作，这些指令能原子地检查并修改一个标志位

```
互斥锁加锁：
原子 CAS 设置锁状态；隐含 acquire 语义（防止临界区操作重排到 lock 之前）。
加锁失败：调用系统调用（如 futex）让线程睡眠
解锁：
原子 store 释放锁；隐含 release 语义（防止临界区操作重排到 unlock 之后）；
如果有等待者，调用 futex(FUTEX_WAKE)。
```

# SMP

多个处理单元（CPU 核心或物理处理器） 具有相同的功能和权限；共享同一物理内存空间；运行同一个操作系统实例，由 OS 统一调度任务到任意处理单元；每个处理单元可以执行任意任务，没有主/从之分（“对称”）。

CCD 将计算核心（如 CPU 核心）分散到多个独立的 Die上，这些 Die 可以集成在同一个Package内。Package 让 Die 能够正常工作，提供外部连接，并可能包含多个Die。

核显和CPU可以在同一片芯片（Die），或是不同的芯片，封装在同一个Package内。

SoC 将CPU、GPU、NPU、内存控制器、I/O接口等多个功能模块封装到一个Package。

苹果把DRAM 芯片与 SoC 芯片被封装在同一个 package 内

服务器中的多个 CPU 通常是位于 不同的 Package 中。这些 CPU 可以被安装在同一个主板上的不同插槽里，从而形成一个多处理器系统。

在多插槽（Socket）系统中，虽然整体是 SMP（逻辑对称），但内存访问延迟可能不一致（本地内存快，远端内存慢）→ 这种系统称为 NUMA（Non-Uniform Memory Access）。NUMA 中两个 CPU 通过 UPI（Intel）或 Infinity Fabric（AMD）这种“缓存一致性互连”直连。

多插槽需要cpu和主板支持

# boot

分区是将一块物理磁盘逻辑上划分成多个独立区域的操作，每个区域可以单独格式化（创建文件系统）

分区不是物理切割，而是在磁盘的“分区表”中记录一段连续的扇区范围，操作系统看到分区后，会把它当作一个独立的“块设备”

分区信息存储在磁盘的分区表中，GPT 是 UEFI 标准的一部分，主 GPT 在磁盘开头，备份 GPT 在磁盘末尾，中间是各个分区

ESP 是一个 分区类型，FAT32 是一种 文件系统格式。ESP 必须格式化为 FAT32，但 FAT32 分区不一定是 ESP

每个 Linux 会在 ESP 中创建自己的目录，有自己的 grubx64.efi，保留一个即可

UEFI 在 ESP 分区的特定路径下查找 .efi 可执行文件，加载到内存，直接执行它（此时 CPU 已运行在 64 位模式）

GRUB 内置了 ext2/ext4、Btrfs、XFS 等文件系统驱动，对每个分区，依次尝试挂载，寻找 grub.cfg 

grub.cfg 告诉 GRUB：有哪些操作系统内核、windows bootloader可以启动

```
一般boot分区独立于根文件系统：
/dev/sda1：/boot（ext4，500MB）
/dev/sda2：/（根文件系统，ext4/Btrfs/XFS）
文件位置：
grub.cfg → /dev/sda1:/grub/grub.cfg
内核（vmlinuz） → /dev/sda1:/vmlinuz-...
操作系统文件（/bin, /etc） → /dev/sda2:/
```

```
set root='hd0,gpt1'
接下来要从 /dev/sda1 读文件

linux /vmlinuz... root=UUID=xyz
GRUB 从 /dev/sda1 读取 /vmlinuz-5.15.0 到内存，并告诉内核：“你的根文件系统在 UUID=xyz 的设备上”

initrd /initrd...
GRUB 从 /dev/sda1 读取 /initrd.img-5.15.0 到内存
内核需要驱动才能挂载根文件系统 → 但驱动又在根文件系统的 /lib/modules/ 里 → 需要额外提供驱动

/
├── init           # 核心！第一个用户进程
├── bin/
│   ├── busybox    # 集成了所有命令：ls、mount、mkdir、insmod 等
│   ├── modprobe
│   ├── mount
│   ├── umount
│   └── switch_root
├── etc/
│   ├── fstab
│   └── init.d/...
├── lib/
│   └── modules/   # 驱动！磁盘驱动、USB、RAID、文件系统驱动
├── dev/           # 空目录，用来挂载 devtmpfs
├── sys/           # 空目录，用来挂载 sysfs
├── proc/          # 空目录，用来挂载 procfs
├── run/
└── mnt/
    └── root/      # 真正的根分区会挂载到这里
    
跳转到内核入口
内核启动与底层初始化: 初始化 CPU、内存管理（MMU）、中断向量表等硬件最底层。
The kernel initializes the device model
挂载 rootfs 类型的文件系统到 /
The initramfs archive is decompressed into the rootfs.
内核运行 /init 脚本
mounts virtual filesystems such as sysfs (/sys), procfs (/proc), and devtmpfs (/dev).
/proc显示内核信息
device model
   ├─ sysfs → /sys/block/sda
   └─ devtmpfs → /dev/sda、/dev/tty1、/dev/tty6
硬盘
  ├─ 数据接口 → /dev/sda(用来读写磁盘)
  └─ 信息面板 → /sys/block/sda(显示设备信息)
加载驱动、激活 LVM/RAID、挂载真正的根文件系统到 /mnt/root
switch_root 切换到真正的根分区（如 /mnt/root），并执行 /sbin/init（系统真正的第一个用户进程）
在启动过程中，PID 1 的"身份"会从 /init 传递给 /sbin/init
```

```
disk(组成RAID)变成PV(Physical Volume)组成VG(Volume Group)划分LV(Logical Volume)
LV：可动态调整大小的disk，属于块设备
mysql建立在文件系统上，oracle建立在disk/raid/lv/文件系统上
```

```
在 GRUB 启动菜单按 e 进入编辑模式，此时可修改内核启动参数。
系统启动时默认以 init 程序管理进程，但通过覆盖 init=/bin/sh 会强制内核直接启动一个无需密码的 Shell，绕过正常登录流程。
mount -o remount,rw /
将根文件系统重新挂载为 读写（rw） 模式，允许修改系统文件
直接运行 passwd 修改 root 密码，新密码会写入 /etc/shadow 文件
exec /sbin/init
手动执行系统的原始初始化进程，正常启动系统
```

# pcie

MMIO（Memory-Mapped I/O，内存映射 I/O）将外设的硬件寄存器地址映射到 CPU 的物理地址空间

每个 PCIe 设备都内置基地址寄存器（BAR, Base Address Register），属于设备的 PCIe 配置空间，BAR 会预设 “所需 MMIO 空间的类型（32/64 位）+ 大小”

```
PCIe 枚举流程：
根复合体扫描所有挂载的 PCIe 设备；
读取每个设备的 BAR 寄存器，获取其所需 MMIO 空间的大小和类型；
从 CPU 预留的 MMIO 地址池中，为设备分配一段连续且未被占用的物理地址段；
将分配的起始地址写入设备的 BAR 寄存器（设备从此知晓自己的 MMIO 基地址）；
系统将 “设备 ID ↔ MMIO 地址段” 的映射关系记录到系统表中（供驱动使用）。
```

```
当 CPU 执行iowrite32(val, mmio_addr)（写 MMIO 地址）时：
CPU 将 MMIO 虚拟地址转换为物理地址；
MMU 识别到该地址属于 MMIO 区域，将读写请求发送到根复合体；
根复合体根据地址映射关系，将请求封装为 PCIe 事务层数据包（TLP），通过 PCIe 总线发送到目标设备；
目标设备解析 TLP，将数据写入自身的对应寄存器（写操作），或读取寄存器值返回（读操作）；
响应数据经 PCIe 总线返回 CPU，完成一次 MMIO 交互。
```

```
CPU 通过 MMIO 配置 DMA（目标物理地址，传输长度）
设备 DMA 引擎启动
设备变成总线主设备，主动发 TLP：
根复合体直接路由到内存控制器
```

NVMe SSD控制器：OS 通过块设备层将 I/O 请求交给 NVMe 驱动，NVMe 驱动通过 MMIO 来提交命令和传输数据。

NVMe和PCIe的关系类似 HTTP和TCP

USB 控制器对 CPU 来说，就是一个 PCIe 设备

每个根复合体可以包含多个 PCIe Host Bridge，每个 PHB 可以对应一个独立的 PCIe 域，每个 PHB 可拆分为多个 Root Port，每一个 Root Port 都代表一个独立的 PCIe 控制器实例。它负责发起配置事务、管理中断以及处理数据包的路由。Root Port 存在于cpu和南桥内，cpu和南桥之间也是 PCIe 总线，南桥的 Root Port 的数量远多于 CPU 与南桥之间的物理连接组数，Root Port 通过 PCIe Switch 可以挂载多个设备。PCIe 通道是 Root Port 提供的带宽资源。

M.2 是通用插座，设计时就预留了多协议支持；PCIe 插槽是专用插座，只为 PCIe 协议服务。M.2 插槽的信号线既连到 SATA 控制器，也连到 PCIe 控制器。但不能同时启用（电气冲突）。开机时，BIOS 根据设置或自动检测，禁用其中一个控制器

# GPU

cpu（计算）-图形适配器（数据转换）-显示器

cpu -显卡（gpu+图形适配器）-显示器

cpu -显卡（视频解码+gpu+图形适配器）-显示器

渲染：

Vulkan/DX/OpenGL 是开放标准，NVIDIA 只是“实现者”之一，提供 专有 Vulkan 驱动（包含在 nvidia-driver 中），驱动将 Vulkan 调用翻译为 NVIDIA GPU 的底层指令

窗口系统接收 GPU 渲染结果（来自各个应用）；将多个窗口的内容渲染为一帧；GPU 显示控制器读取并输出

计算：

CUDA 是 NVIDIA 私有技术，只有 NVIDIA GPU 能原生运行

```
OpenGL / Vulkan / DX12 / CUDA 
驱动编译 Shader / Kernel
生成 GPU ISA 指令（机器码）
驱动循环用 MMIO write 把指令逐个写到 GPU 的 MMIO 区域
```

# 产品

线程数是核心数的两倍，这通常是因为处理器支持超线程/同步多线程技术，现代 CPU 核心内部有大量执行资源，单个线程往往无法完全占满所有资源，在同一个物理核心上，同时调度两个线程共享执行资源

| 品牌  | 产品线       | 子产品线     | 代   | 子子产品线 | 后缀 |
| ----- | ------------ | ------------ | ---- | ---------- | ---- |
| Intel | Xeon         | Gold         | 6    | 230        |      |
|       | Core         | i7           | 13   | 700        |      |
|       | Pentium      |              |      |            |      |
|       | Celeron      |              |      |            |      |
| AMD   | EPYC         |              |      |            |      |
|       | Threadripper |              |      |            |      |
|       | Ryzen        | 7            | 7    | 700        |      |
|       | Athlon       |              |      |            |      |
| 海思  | 鲲鹏         | ARM          |      |            |      |
| Apple | M            | soc(ARM cpu) |      |            |      |

soc(ARM cpu)

| 品牌   | 产品线 | 子产品线 | 后缀  | 代   |
| ------ | ------ | -------- | ----- | ---- |
| 高通   | 骁龙   | 4/6/7/8  | s/+/E | Gen5 |
| 联发科 | 天玑   | 8        |       | 200  |
| 海思   | 麒麟   |          |       |      |
| Apple  | A      |          |       |      |
| 三星   | 猎户座 |          |       |      |

```
正代顶级旗舰：888→8gen1→8+gen1→8gen2→8gen3→8elite→8eliteGen5
附代次旗舰：8sgen3→8sgen4→8(s)gen5
```

```
iphone12
vivo,iqoo
oppo,oneplus,realme
小米，红米
华为，荣耀
```

# 屏幕

100%sRGB高色域：色彩数量

2880*1800：像素点数量

120Hz：刷新率

400尼特：最高亮度

