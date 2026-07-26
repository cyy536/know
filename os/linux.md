Linux Kernel Development, 3rd Edition

# 历史

1969，Ken 用汇编语言开发了 Unix 的最初版本，设计了 B 语言，B 语言的创建是为了开发一个更高效、更灵活的系统编程语言，用于重写和改进早期 Unix 操作系统

1972，Ritchie 在 B 语言基础上，发明了 C 语言，与 Ken 一起用 C 重写了 Unix

1978，Intel 8086微处理器

1981，MS-DOS；1993，Windows NT 3.1。与 Unix 无任何继承关系

```
AT&T 贝尔实验室 Unix
├── BSD（Berkeley Software Distribution） → FreeBSD(对照 Unix 重写) → macOS
└── System V 等商业分支
```

POSIX（Portable Operating System Interface） 是 IEEE 制定的一套标准，定义了 Unix-like 系统应提供的 API。让软件可以在不同的类 Unix 系统上无需修改或只需少量修改即可编译和运行

自由软件基金会（FSF）创始人 Richard Stallman 发起了 GNU 项目。目标：创建一个完全自由的、类 Unix 的完整操作系统，以对抗当时 Unix 的专有化和封闭。GNU 提供了构建完整操作系统所需的几乎所有用户空间工具。

glibc（GNU C Library） 是 Linux 和许多类 Unix 系统中最核心的标准和系统库之一

1991，Linux 内核(独立实现) + GNU 工具 = 一个完整的、可运行的自由操作系统

Android：使用 Linux 内核，但不用 GNU 工具，而是用 Google 的 Bionic C 库，不完全符合 POSIX 标准

每个进程的地址空间被划分成两个部分：用户空间，内核空间

进程上下文，中断上下文：代码有进程身份，没有进程/线程身份，不能休眠

#  进程

```
struct task_struct{
pid;
state;
prev;
next;
parent;
children;
};
```

内核栈大小固定，入栈，栈顶指针减小，栈的最低地址存放thread_info

```
struct thread_info{
struct task_struct*;
};
```

POSIX 使用fork创建进程，fork会调用clone系统调用

clone可以创建进程或线程：

```
创建并复制父进程的task_struct和内核栈，分配pid

either duplicates or shares resources. These resources are typically shared between threads in a given process; otherwise they are unique and thus copied here.

拷贝或共享进程地址空间mm_struct，内核线程没有mm_struct，被调度时复用前一个进程的mm_struct，内核空间是相同的
拷贝地址空间采用写时复制，延迟拷贝，如果子进程在fork后调用exec载入可执行文件，地址空间是全新的，不需要拷贝
创建并复制父进程的PGD → PUD → PMD → PTE
将父子双方的 PTE 都标记为只读
当任一进程尝试写入某页时，会触发page fault，分配新物理页，复制原内容，更新PTE
相同用户空间，物理地址不同
```

pthread 是 POSIX 标准定义的一套线程 API，NPTL 是 glibc 提供的 pthread API 的具体实现

```
主线程的用户栈：由内核在 exec() 时设置
子线程的用户栈：由 NPTL 通过 mmap 动态分配
void pthread_create(...) {
    void *stack = mmap(...); 
    clone(
        thread_func,         
        stack + STACK_SIZE,
```

exit:

```
减少资源引用，减为0则释放
向父进程发信号，给子进程设置养父（其他线程或init进程）
僵死状态
schedule()
```

wait：挂起进程直到有子进程退出

```
父进程获得信号后释放子进程task_struct，thread_info，内核栈
init会例行调用wait
```

# 进程调度

IO密集：调度频繁，每次运行时间短

计算密集：调度次数少，每次运行时间长

调度策略需要考虑响应时间和吞吐量

linux 有多个调度类，每一类负责一些进程，有不同的优先级。只要高优先级调度类有可运行任务，它就一定会抢占当前正在运行的低优先级任务。

CFS负责普通进程：

```
超参数 targeted latency，可运行的时间片总量，按权重分配给每个可执行进程。targeted latency 越小交互性越好切换越多。
Each process then runs for a “timeslice” proportional to its weight divided by the total weight of all runnable threads.
在nice相同的情况下，时间片相同，IO密集很快挂起，使用时间片少，可以抢占计算密集，计算密集会占用CPU大部分时间
超参数 minimum granularity，进程最小时间片，当进程数量很多时，所有进程时间片相同。1ms
这些超参怎么用的?

update_curr计算当前进程的执行时间，时钟中断处理程序的一部分
__update_curr(delta_exec)根据可运行进程数量和nice对delta_exec进行加权，更新vruntime
在必要时设置need_resched

vruntime 是 CFS 判断“谁该下一个运行”的唯一依据：总是选择 vruntime 最小的可运行进程。
将所有可运行进程存储在红黑树（key=vruntime）
```

休眠：从红黑树移出，加入等待队列，调用schedule

two states are associated with sleeping, TASK_INTERRUPTIBLE and TASK_UNINTERRUPTIBLE. They differ only in that tasks in the TASK_UNINTERRUPTIBLE state ignore signals, whereas tasks in the TASK_INTERRUPTIBLE state wake up prematurely and respond to a signal if one is issued. 

```
DEFINE_WAIT(wait);//等待队列
add_wait_queue(q, &wait); 
while (!condition) {    /* condition is the event that we are waiting for */
    prepare_to_wait(&q, &wait, TASK_INTERRUPTIBLE); //被唤醒后可能再次加入等待队列
    if (signal_pending(current))
		break;//当返回用户空间时，内核检查到 pending 信号 → 跳转到用户注册的信号 handler
	schedule();//返回时，state==TASK_RUNNING
}
finish_wait(&q, &wait);
```

唤醒：将等待队列的所有进程移入红黑树，如果被唤醒的进程优先级比当前进程高，设置need_resched

```
int send_signal(int sig, struct task_struct *t) {
    add_signal_to_pending_queue(t, sig);
    if (t->state == TASK_INTERRUPTIBLE) {
        wake_up_process(t);//task state is set to TASK_RUNNING
    }
}
```

即使是 SIGKILL，也只对 TASK_RUNNING 或 TASK_INTERRUPTIBLE 有效。从 Linux 2.6.25 起，内核引入了 TASK_KILLABLE 状态：TASK_UNINTERRUPTIBLE（不响应普通信号）；但可被 SIGKILL 唤醒！

schedule的时机：

内核返回用户空间，need_resched==1

内核代码被中断打断后返回，need_resched==1且preempt_count(使用锁的数量)==0

# 中断

OS通过设备驱动程序和外设交互，读写外设端口

中断处理程序 ISR 是设备驱动程序的一部分，一个设备可能产生不同的中断，相应有多个中断处理程序，多个设备可以共享同一个中断处理程序，但必须独立注册

网卡上半部读数据包到内存，下半部处理数据包，上半部借用被中断进程的内核栈

在早期 PC（如 ISA 总线时代），外设通过专用的中断请求线（IRQ 线） 连接到中断控制器

在现代系统（PCI/PCIe 时代），外设不再使用专用 IRQ 线，而是通过 PCIe 数据通道发送一个特殊的“中断消息”（本质是一次内存写事务），中断控制器（如 APIC）监听特定内存地址，收到写操作即视为中断。

每个核心有一个 Local APIC，系统中还有一个 I/O APIC（或多个），用于将外部设备中断分发给各个核心的 Local APIC。

```
request_irq(IRQ,intr_handler,flags,dev)
//注册中断处理程序
flags：多个中断处理程序可以共享中断号
intr_handler(IRQ，dev)
```

一个 IRQ 会被路由到某一个特定的 CPU，同一 IRQ 不会同时在多个 CPU 上被处理，在处理一个 IRQ 时，该 IRQ 在所有 CPU 上都会被暂时屏蔽，对于共享 IRQ，内核会串行调用所有注册的处理程序（在同一个 CPU 上），因此这些 ISR 不会并行执行，ISR 必须能够判断设备是否产生中断，设备要提供状态寄存器供 ISR 查询

自 Linux 2.6.35 移除 IRQF_DISABLED 后，所有 ISR 默认在中断使能状态下运行，高优先级中断可以抢占当前 ISR

# 下半部

软中断只有32个，在编译期间注册，相同软中断可以在所有处理器上同时执行

tasklet是基于软中断实现的，可以动态注册，相同tasklet不能同时执行

工作队列会创建内核线程（每个处理器一个），不断处理提交的任务，无任务时休眠，系统有一个默认工作队列，可以创建新的工作队列

上面三种都可以响应中断

中断处理程序在返回前会标记软中断，使其在稍后被执行（中断处理程序返回或内核线程）

```
open_softirq(索引号，处理函数)//注册软中断
raise_softirq(索引号)//标记软中断
do_softirq()//处理所有被标记的软中断
```

软中断执行时可以重新触发自己以便再次得到执行，当大量软中断出现时，内核会唤醒一组内核线程来处理，在最低优先级运行，每个处理器都有一个，会不停的调用do_softirq

网络子系统、定时器使用软中断，大部分下半部通过tasklet实现

```
tasklet_schedule(&my_takslet)
//关中断，把需要调度的tasklet加到处理器的tasklet链表，唤起软中断，恢复中断
tasklet_action()//软中断处理函数：遍历tasklet链表，处理
```

# 同步

原子操作，自旋锁

信号量可以睡眠，可以在不同上下文增减

互斥锁是一种简化信号量，只能在同一个上下文里加锁和解锁

Linux/Unix 核心 IPC 分为 6 大类：管道、消息队列、共享内存、信号量、套接字、信号；

创建一个匿名管道，并返回了两个描述符。匿名管道是特殊的文件，只存在于内存，不存于文件系统中。使用 fork 创建子进程，创建的子进程会复制父进程的文件描述符（shell管道就是复制shell进程的描述符到子进程）

如果进程没有注册自己的信号处理函数，那么操作系统会执行默认的信号处理程序，但如果注册了，则会执行自己的信号处理函数

# 内存管理

```
struct page;//物理页
struct zone;
在x86-32上，0-16MB是zone_dma，16-896MB是zone_normal，>896MB是zone_highmem
32位CPU进程地址空间总共是4GB，内核空间3GB-4GB，地址数量较少，不能映射全部物理内存，部分地址要动态映射
0xC0000000 + 0 ~ 0xC0000000 + 896 MB，映射zone_dma+zone_normal
剩下128MB，kmap分配一个地址给zone_highmem里的page
x86-64没有zone_highmem，内核空间范围64-128TB，所有物理内存都能放入直接映射区
alloc_pages/__get_free_pages分配物理页，返回的是page*/虚拟地址。所谓的分配和释放实际上只是标记zone等数据结构的使用情况
kmalloc分配的物理空间连续，vmalloc分配的物理空间无需连续，kmalloc可禁止睡眠，vmalloc可能睡眠
slab缓存特定数据结构（task_struct），降低分配和释放开销，由一个或多个物理页组成
```

# 虚拟文件系统

各种文件系统的通用接口，具体文件系统必须提供这些接口的实现

```
write(fd,buf,len);
//write系统调用找到fd所在的文件系统实现的write，再执行文件系统的write

fd = open("a.txt");
//创建 file 结构，在当前进程fd_table里找个空位，fd_table[3] → file
//两次open打开相同的文件，file是不同的，指向相同的dentry
struct file *fd_table[];
fork() 时父子进程会复制fd_table，共享file，父子线程共享fd_table

int saved = dup(1); //备份原 stdout
//fd_table[saved] = fd_table[1];
dup2(fd, 1);   // 重定向 stdout
//fd_table[1] = fd_table[fd];
dup2(saved, 1);   // 恢复 stdout
```

目录就是存放文件名和 inode 对应关系的特殊文件

硬链接 = 在某个目录文件里增加一个新的目录项，指向已有 inode。

软链接增加一个新的目录项，创建了一个新的 inode，内容是“路径字符串”

```
/ → home → user → a.txt
读取/的内容，找到home的inode，读取home的内容，找到user的inode，读取user的内容，找到a.txt的inode
形成4个目录项对象，目录项会被缓存
```

文件元数据（索引节点）和文件本身分开存储，文件系统元数据存储在超级块

```
超级快，索引节点区，数据区
文件系统在磁盘中的布局不一定如此，但必须提供相应的对象
```

超级块对象代表已挂载的文件系统，全局只有一个

文件系统需要挂载（vfsmount）才能使用，同一个 super_block 可以被多个 vfsmount 引用

每个进程的 struct task_struct 里，有一个指针指向它所属的 mnt_namespace，每个 mnt_namespace 维护了该 namespace 下所有 vfsmount

容器创建了一个新的 mount namespace，挂载新根文件系统

```
mount -t overlay overlay2 -o lowerdir=/dir1:/dir2,upperdir=/dir3,workdir=/dir4 /merged
lowerdir：镜像层（只读）
upperdir：容器修改层（可写）
merged：合并后给容器看到的 rootfs
```

文件系统类型对象：内核维护一个全局链表，保存所有已注册的文件系统类型

```
mkdir /tmp
# tmpfs：必须指定 -t tmpfs，且第一个参数是任意标签（通常也写 tmpfs）
mount -t tmpfs tmpfs /tmp
# 磁盘文件系统：通常省略 -t（自动探测），第一个参数是设备
# mount /dev/sdb1 /mnt/data
内核在链表中找到 name == "tmpfs" 的 file_system_type
调用其 .mount 函数（shmem_mount）
shmem_mount 创建 super_block（如果没有），初始化该 super_block 的根 inode 和根 dentry
VFS 将此挂载点封装为 vfsmount，插入当前进程的 mount namespace
查找路径时：
内核会检查 dentry 是否是一个挂载点，通过查询当前 mount namespace 中的挂载表
如果是，不再使用 dentry 对应的 inode（即原 /tmp 目录的内容），而是切换到被挂载文件系统的根 dentry，继续在新文件系统中查找剩余路径
```

```
/usr/bin/bash
当 /usr/bin 是挂载点时：
dentry cache 中会生成两条关键缓存项：
(vfsmount_root, usr/bin) → 指向 /usr/bin 挂载点的 “挂载标识”（而非原目录 inode）；
(vfsmount_bin, bash) → 指向 bash 的 inode；
下次访问 /usr/bin/bash 时：
内核先通过第一条缓存找到 /usr/bin 是挂载点，切换 vfsmount 后，再通过第二条缓存找到 bash，全程无需重新解析磁盘。
```

Network Attached Storage: 网络文件服务器
```
应用
   │
POSIX接口
   │
NFS/SMB
   │
NAS

mount -t nfs 192.168.1.10:/data /mnt
```

S3、OSS
```
{oid,metadata,data}
http：put, get, delete
网盘、音视频、图片、网页
```

# namespace

```
ps/top → 读 /proc
/proc 是内核把自己的内存数据结构，以文件形式暴露给用户态。你看到的所有内容，都是内核实时生成的。
以数字命名的目录：进程信息，ps/top 只生成当前 PID namespace 可见的进程目录。
子 ns → 看不见父 ns、兄弟 ns
父 ns → 能看见所有后代 ns
同 ns → 互相可见
以英文命名的文件 / 目录：全局系统信息
```

```
System V IPC: 早期 Linux/Unix 进程通信方案
struct ipc_namespace {
    struct ipc_ids  shm_ids;  // 共享内存资源列表
    struct ipc_ids  msg_ids;  // 消息队列资源列表
    struct ipc_ids  sem_ids;  // 信号量资源列表
    // 其他属性：IPC 资源限制、引用计数等
};
POSIX IPC: 标准化、现代化的 IPC 方案
POSIX 共享内存（/dev/shm）、POSIX 消息队列（/dev/mqueue）本质是文件系统，由 Mount namespace 隔离
如果设置 --ipc=host：
容器使用宿主机的 IPC namespace → 能访问宿主机的 System V IPC；
但 /dev/shm 仍为容器独立挂载 → POSIX IPC 依然隔离（除非手动绑定宿主机的 /dev/shm）
```

```
struct user_namespace {
    struct uid_gid_map uid_map;  // UID 映射表（容器 UID → 宿主机 UID）
    struct uid_gid_map gid_map;  // GID 映射表（容器 GID → 宿主机 GID）
    struct user_namespace *parent;  // 父 User namespace（层级结构）
};
内核权限检查只认映射后的外部 ID，容器内 root 无宿主机 root 权限；
```

```
/sbin/init → 实际是 systemd 的符号链接
systemd：
挂载 tmpfs 到 /run系统和守护进程运行状态 /tmp通用临时文件 /dev/shm进程间共享内存；
挂载 devpts 到 /dev/pts（伪终端目录，SSH / 终端都靠它）；
挂载 cgroup2 到 /sys/fs/cgroup
启动 dbus-daemon ，创建 System Bus，创建 Unix Socket，等待客户端连接，维护服务注册表
启动ssh、NetworkManager
启动登录界面
```

```
Cgroup 最核心的设计是：通过伪文件系统（cgroupfs）暴露接口，用户态修改文件 → 内核生效资源规则。
/sys/fs/cgroup/
├── cgroup.controllers       # 支持的子系统
├── cgroup.subtree_control   # 启用子系统控制
├── cgroup.procs             # 根组进程
├── containerA/              # 容器 A 的控制组
│   ├── cgroup.procs         # 容器 A 的进程 PID
│   ├── cpu.max              # 容器 A CPU 限制
│   ├── memory.max           # 容器 A 内存限制
│   └── io.max               # 容器 A IO 限制
└── containerB/              # 容器 B 的控制组
    ├── cgroup.procs
    ├── cpu.weight
    └── pids.max
```

```
systemctl → D-Bus → systemd → sshd
GUI / nmcli / VPN / settings
   ↓ org.freedesktop.NetworkManager.AddConnection()
D-Bus
   ↓
NetworkManager
```

```
service ssh start
service是一个封装脚本（通常位于 /usr/sbin/service），它会自动检测系统的初始化方式：
如果 systemd 存在 → 调用 systemctl。
如果 systemd 不存在 → 调用传统的 SysVinit 系统的 /etc/init.d/脚本。
/etc/init.d/ssh: Shell 脚本定义了 start、stop 等操作
/usr/sbin/sshd 是 真正的 SSH 服务二进制文件
```

# 块IO层

```
扇区(设备)<=块(文件系统)<=页
```

块存放在物理内存缓冲区，一个缓冲区对应一块，缓冲区头buffer_head结构体记录每一个磁盘块和物理内存的映射关系

```
每个块IO操作由一个结构体表示
struct bio{
struct bio_vec*;//链表
bi_idx;//bio_vec当前索引
};
struct bio_vec{
struct page*;
bv_len;
bv_offset;
};
```

块IO请求可以包含多个bio，文件系统创建块IO请求，挂到特定的块设备的请求队列上

IO调度程序将请求队列的请求进行合并和按扇区排序（电梯算法），调用驱动程序处理请求

```
disk-buffer_head-page
disk-bio-pages
```

# 进程地址空间

```
struct mm_struct{
struct vm_area_struct *mmap;//vm_area链表
struct rb_root;//vm_area红黑树
struct list_head;//mm_struct组成的链表
};
struct vm_area_struct{
vm_start;
vm_end;
vm_page_prot;//访问控制权限：读、写、执行、所映射内容可多进程共享
struct vm_area_struct *vm_next;
struct rb_node;
struct file *vm_file;// 若为文件映射
};
```

vm_area代表C中的代码段、数据段等，动态链接库也有自己的代码段、数据段等，也是独立的vm_area

do_mmap()创建或扩展vm_area

```
do_mmap(flie*,addr,offset);//文件映射，直接读写page cache 
//addr搜索空闲空间的起始位置，offset文件偏移
//file==NULL，匿名映射
```

虚拟地址-页表、TLB-物理地址，搜索页表由硬件完成，设置页表由操作系统完成

malloc() 分配的是虚拟内存

malloc 申请内存的时候，会有两种方式向操作系统申请堆内存。通过 brk() 系统调用从堆分配内存。通过 mmap() 系统调用在文件映射区域分配内存。默认分配大块内存 (128KB) 才使用 mmap 分配内存空间。

malloc() 在分配内存的时候，并不是老老实实按用户预期申请的字节数来分配内存空间大小，而是会预分配更大的空间作为内存池。free()后，通过 brk() 方式申请的内存不会归还os，通过 mmap 方式申请的内存会归还

维护未使用的内存块，按大小分类（如 bins），支持快速合并相邻空闲块

malloc 返回给用户态的内存起始地址比进程的堆空间起始地址多了 16 字节，多出来的 16 字节就是保存了该内存块的描述信息

# 页高速缓存

用空闲物理内存作磁盘块的缓存，可以收缩以缓解内存压力

read会检查磁盘数据是否在缓存中，失败才访问磁盘，读来的数据会缓存起来

write-through同时写缓存和磁盘

write back由回写进程周期性将脏页写回磁盘

缓存不足，需要驱逐：LRU

一个文件只能有一个address_space，可以有多个vm_area_struct

```
struct address_space{
struct inode;
struct prio_tree_root i_mmap;//优先搜索树
struct address_space_operations *;//文件系统指定，从磁盘读数据到缓存页，从缓存页写到磁盘
};
```

read

```
page=find_get_page(mapping,index);//从address_space找文件[index]对应的缓存页
error=add_to_page_cache_lru(page,mapping,index);
//没命中的话，分配新page，映射加入address_space
error=mapping->a_ops->readpage(file,page);
//从磁盘读数据到page
```

write：查找，(分配page)，用户数据先copy到page，SetPageDirty

文件映射：（缺页），CPU 执行写操作时硬件自动设置页表项的 dirty bit

```
inode-address_space-pages
```

回写时机：内存不足，驻留时间，sync系统调用

一个磁盘对应一个回写线程

# 设备

块设备通过文件访问，字符设备通过字符设备节点文件访问，网络设备通过socket访问

模块是一些代码，可动态加入内核

```
统一设备模型被嵌入到其他结构体中，如字符设备
struct kobject{
struct kobject *parent;//拓扑连接
struct sysfs_dirent *sd;//表示当前kobject的inode
struct kobj_type *ktype;
struct kset *kset;//kobject形成的链表，表示全部块设备等，形成sysfs的独立目录
};
struct kobj_type{
struct sysfs_ops *;//读写文件操作
struct attribute **default_attrs;//转换为文件
};
struct attribute{
const char *name;//转换成文件名
};
```

sysfs是内存文件系统，将kobject和目录项连接

```
block块设备
dev设备
devices设备拓扑
kernel内核
module已加载模块
```

```
GPU 通常通过“字符设备（character device）”暴露给用户空间
ioctl(fd, NVGPU_SUBMIT_KERNEL, &cmd);向设备发送控制命令

CUDA 和 OpenGL 等用户态 GPU API 通过 libnvidia-*.so 提供实现。
这些库通过 ioctl 等系统调用访问 /dev/nvidia* 设备节点，与内核驱动 nvidia.ko 通信，
由驱动提交命令到 GPU，最终由 GPU 执行计算或图形任务。

宿主机必须先加载 nvidia.ko ，内核才会创建 /dev/nvidia* 设备文件；
容器内没有 nvidia.ko，容器内的 /dev/nvidia* 和 libnvidia-*.so 是 “映射” 而非 “拷贝”：
容器内的 libcudart.so 是 CUDA 运行时，依赖宿主机的 libnvidia-cuda.so（驱动库）才能工作 —— 前者可安装，后者只能映射。

Docker CLI（docker run --gpus all）
    ↓ 解析 --gpus 参数，调用 Docker 守护进程
Docker Daemon（dockerd）
    ↓ 识别到 GPU 请求，调用指定的运行时
nvidia-container-runtime（GPU 专用运行时）
    ↓ 内部调用 runc，并注入 GPU 相关配置
runc（容器底层运行时，创建容器Namespace/Cgroup）
    ↓ 启动容器，容器内可见 GPU 设备/库
```

# socket

创建socket会创建inode和file

Linux 内核支持的 4 种 socket 类型

RAW：原始套接字，直接读写 IP 层数据包

UNIX：只在本机内通信，不经过网卡、不进网络协议栈，以文件路径作为地址

自己发给自己：走 loopback 设备 lo，纯软件虚拟设备

服务端 accept 返回用于传输的 socket 的文件描述符，监听的 socket 和真正用来传送数据的 socket，是「两个」 socket

Netfilter在数据包流经协议栈的关键位置埋5个钩子点，（ip-eth，ip-tcp）

```
ip->Netfilter:LOCAL_OUT->路由表->Netfilter:POST_ROUTING->填好以太网头->网卡驱动

驱动把 skb 内存 映射成 DMA 地址(虚拟地址->物理地址，网卡只认识物理内存地址)
把tx_desc放到 Tx Ring 队列里
通知网卡：有包要发
struct tx_desc {
    dma_addr_t buf_addr;  // 数据包的物理地址
    int len;              // 长度
    int flags;            // 标识：是否最后一个描述符
    int status;           // 网卡写完会置位：发完了
};

网卡收到通知后：
从 Tx Ring 里取下一个描述符
拿到里面的 buf_addr（物理地址）
启动 DMA 读操作：直接从内存地址把数据包读到网卡内部
网卡补全物理层帧
从网口发出

ring_desc 
硬件中断
驱动把收到的帧封装成 sk_buff
(网桥->网卡驱动)
ip
Netfilter：PRE_ROUTING
路由决定：走 LOCAL_IN（本机） 还是 FORWARD（转发）
Netfilter：LOCAL_IN->TCP/UDP
Netfilter：FORWARD->Netfilter：POST_ROUTING

IP 是本机 → 本机处理
IP 不是本机
	ip_forward=1 → 转发
	ip_forward=0 → 丢
```

iptables 是配置 Netfilter 的命令行工具

POST_ROUTING：iptables 配置的 SNAT 规则生效，替换源IP

```
Destination     Gateway         Genmask         Flags Metric Iface
0.0.0.0         192.168.1.1     0.0.0.0         UG    0      eth0
192.168.1.0     0.0.0.0         255.255.255.0   U     0      eth0
127.0.0.0       0.0.0.0         255.0.0.0       U     0      lo

网卡 IP 的作用：发包时当源 IP，收包时判断是不是本机
目标 IP 按位 & 子网掩码 = 路由表里的 Destination
就算匹配上。如果多条都匹配，选掩码最长的（最精确的）
如果是直连网段，Gateway 是 0.0.0.0

网卡 MAC 的作用：发包时当源 MAC，收包时判断是不是本机
掩码在链路层的作用：判断接收方是否和发送方在一个网络，不在，要经过网关，目标 MAC 为网关地址；在，目标 MAC 为接收方链路层地址
arp：根据ip获取链路层地址，在链路层广播询问
```

```
网卡只接收目的 MAC 地址是自己，或者是广播、组播的数据包，其余包一律在硬件层面丢弃
混杂模式：允许网卡接收任何 MAC 地址的数据帧。tcpdump 或 Wireshark 通过特殊的内核接口直接在二层克隆一份数据流
交换机本质上就是一根多端口的智能网线，只记别人的 MAC，自己不需要mac，也不需要ip
网桥：网卡把包吸进来后，网桥决定这个包是转发给某个端口，还是丢给宿主机自己的协议栈

ip link add br0 type bridge
会生成虚拟网卡 + 虚拟交换机
ip link set eth0 master br0
br0的MAC来自某一张物理网卡，物理网卡不再需要mac，变成交换机端口了，由br0代为物理网卡
将网卡和路由器的单线连接扩展接口

pc1-eth1   ──┐
pc2-eth2   ──┼── br0(pc0)
路由器-eth3 ──┘
```

```
标准路由器每个口必须要有IP和mac，作为网关
家庭路由器：标准路由器+交换机
pc1-lan1 ──┐
pc2-lan2 ──┼── br0(pc0) - forward - wan
pc3-lan3 ──┘
LAN口是交换机端口
网关是路由器内部一个虚拟网卡，虚拟网卡也连接交换机端口
交换机根据目的MAC，交给虚拟网卡
ip不是本机，进入路由 → NAT → 从 WAN 口出去
软路由：用主机实现标准路由器/家庭路由器
```

# tty

用户读键盘，写显示，需要两个队列

```
通路1：键盘写输入队列
回显：tty把输入队列自动复制到输出队列
通路2：用户 read 输入队列
通路3：用户 write 输出队列
```

```
规范模式（Cannonical Mode / Cooked Mode）
这是 TTY 默认模式，也是最常用的模式（比如执行 ls/echo 时）
输入预处理：行缓冲、控制键解析、回显

原始模式（Raw Mode）
vim/nvim/nano/top 等全屏程序的核心模式
内核把预处理的责任全甩给用户
```

用户读写：

```
当你执行 echo "hello" > /dev/tty1 时，流程是：
解析路径 /dev/tty1 → 找到 /dev 是 devtmpfs 挂载点；
devtmpfs 根据文件名 tty1，找到自己管理的inode（比如 inode 号 123）；
从这个 inode 里读取：“字符设备，主 4 次 1”；
内核根据主设备号 4，找到 TTY 子系统的 tty_fops，tty_fops封装共性
内核为每种终端硬件定义了 tty_driver 结构体，通用入口根据次设备号（比如 tty1 是次 1，ttyS0 是次 64）找到对应的硬件驱动；最终执行硬件特有的底层函数

struct tty_driver console_driver = {
    .name = "tty",
    .major = 4,          // 主设备号 4
    .open = con_open,    // 虚拟终端特有的 open
    .write = con_write,  // 虚拟终端特有的 write
};
```

键盘输入：

```
键盘驱动解析扫描码，调用 input_event() 上报，此时事件同时被 kbd 驱动 和 evdev 驱动 接收
kbd 驱动直接操作 tty 结构体内部的队列。
kbd 负责切换 VT
```

显示输出：

```
VT的显示驱动 vgacon、终端模拟器，内置了 ANSI 解析逻辑
基础逻辑是「在当前光标位置显示字符，然后光标后移」；而原始模式下，程序能通过清屏 + 精准控制光标位置，实现全屏任意位置的绘制。

VGACON：直接操作显卡显存，把字符 / ANSI 指令转换成屏幕像素（比如光标位置对应显存地址，字符对应字库点阵）；
终端模拟器：在图形界面里模拟 “字符屏幕”，调用图形 API 绘制字符和光标；

本质上，nvim 和 vim 都只能输出字符（含 ANSI 控制指令），没有任何图形绘制能力，二者的核心差异是对扩展字符 / ANSI 指令的支持程度
vim 不管 “画字符”，但必须算准字符宽度，才能给终端发精准的光标 / 排版指令；
老版本 Vim 因 ASCII 时代的老旧架构，对 Unicode 全角 / 半角宽度计算错误，导致终端显示错位；
Neovim 重构了底层计算逻辑，严格遵循 Unicode 宽度标准，从根源上解决了这个问题；

早期终端（VT100、老式 Linux 控制台）硬件性能有限，只能支持少量预设颜色
真彩色：可指定任意 RGB 值

图片（JPG/PNG）：nvim 仅输出 “图片指令 + 路径”（本质还是字符），调用终端的图片协议，由终端解析后显示
```

# login

```
Session：一组进程的集合
Process Group：会话内的子集，可整体接收信号
Foreground PG：当前与tty交互的进程组
```

init 会为每个虚拟控制台启动一个 agetty 进程

```
int fd = open("/dev/tty1", O_RDWR);
dup2(fd, STDIN_FILENO);
dup2(fd, STDOUT_FILENO);
dup2(fd, STDERR_FILENO);
//将自己的标准输入、输出、错误重定向到该 TTY
char *username = read_username();
execl("/bin/login", "login", username, (char *)NULL);
// 此时 agetty 进程消失，PID 不变，login 接管终端
login 进程继承了 agetty 打开的 TTY 文件描述符
验证密码
systemd --user → dbus session bus
调用 setsid() 创建新会话（SID = login 的 PID）
ioctl()，内核将 /dev/tty1 绑定为该会话的控制终端
execl("/bin/bash", ...)
bash：
pid_t pid = fork();
if (pid == 0) {
    // 子进程
    setpgid(0, 0);  // 创建新进程组
    exec("vim");
}
设置vim为Foreground PG，vim 退出后，切回bash
当 bash 退出，内核自动向整个会话发送 SIGHUP，所有子进程被通知清理资源，内核自动解除 TTY 与会话的绑定
```

桌面环境在 Session Bus 上注册桌面通知服务

# pty

一套 PTY = 2 个设备（master + slave），每个都有输入队列 + 输出队列

master = 虚拟键盘 + 虚拟显示器

```
用户 read slave 输入队列
用户 write slave 输出队列
回显：写 slave 输入 → 自动复制到 slave 输出
模拟器 write master 输出队列 → 转给 slave 输入
模拟器 read master 输入队列（来自 slave 输出）
```

```
// sshd 分配 PTY 的简化流程
int master_fd = open("/dev/ptmx", O_RDWR);        // 打开 PTY master
grantpt(master_fd);                               // 设置权限
unlockpt(master_fd);
const char *slave_name = ptsname(master_fd);      // 获取 slave 路径（如 /dev/pts/3）

// fork 子进程
if (fork() == 0) {
    // 子进程：成为 shell
    int slave_fd = open(slave_name, O_RDWR);
    dup2(slave_fd, 0);  // stdin → PTY slave
    dup2(slave_fd, 1);  // stdout → PTY slave
    dup2(slave_fd, 2);  // stderr → PTY slave
    setsid();           // 创建新会话
    ioctl(slave_fd, TIOCSCTTY, 1);  // 设置控制终端
    exec("/bin/bash");  // 启动 shell
}

// 父进程（sshd）：通过 master_fd 与 shell 通信
char buf[4096];
while (1) {
    n = read(master_fd, buf, sizeof(buf));  // 读取 shell 输出
    send_to_client(buf, n);                 // 通过 SSH 协议发送给客户端
    n = recv_from_client(buf, sizeof(buf)); // 接收客户端输入
    write(master_fd, buf, n);               // 写入 PTY master → 触发 slave 的 TTY 行规程
}
```

```
vt-vim
vt-ssh-sshd-master-slave-vim
SSH 将本地 TTY 设为原始模式，Ctrl+C 在 slave 处理
tty-tmux_client-tmux_server-master1-slave1
					       -master2-slave2
```

# compositor

evdev → 用户空间合成器 → 应用

```
Wayland compositor 职责：
通过 libinput 监听 /dev/input/eventX
直接决定事件路由
通过 Wayland protocol (wl_keyboard) 发送 key 事件给客户端
```

```
应用声明"需要输入法"
按键事件先发送给输入法
输入法合成预编辑文本（如拼音 → 汉字候选）
用户确认后，输入法向应用提交最终文本
输入法的候选框主要由应用自身渲染
```

# 显示

VT 可以用 vgacon、fbcon 或 drm_con。

```
        ├─ vgacon      （老式 VGA 文本模式）
        ├─ fbcon       （基于 fbdev 帧缓冲）
        └─ drm_con     （现代 GPU DRM/KMS）
```

fbdev:

```
int fd = open("/dev/fb0", O_RDWR);
void *fb = mmap(NULL, screensize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
memset(fb, 0xff, screensize);  // 全屏白色
```

compositor:

```
打开Chrome，Xorg 给 Chrome 分配一个 1200×800 的窗口缓冲区（显存地址 A）；
Chrome 调用 OpenGL 画网页内容 → 通过 libGL.so → 渲染到显存地址 A；
你又打开 VS Code，Xorg 给它分配显存地址 B（1000×700）；
Xorg 把地址 A（Chrome）和地址 B（VS Code）按层级合成到 “全屏帧缓冲区”（显存地址 C）；
Xorg 通过标准的 DRM API 调用 libnvidia-modeset.so，告诉 nvidia.ko：“显示引擎从地址 C 读取数据输出到 HDMI”；
GPU 显示引擎按 60Hz 扫描地址 C → 屏幕显示 Chrome + VS Code 的叠加窗口。
```

# I/O

一次 read 操作通常包含两个阶段：等待内核中的数据准备好，将数据从内核空间拷贝到用户空间

非阻塞 I/O 的特点是：如果内核数据还没准备好，read 会立即返回，不会阻塞线程。应用程序需要反复轮询，检查数据是否准备好。一旦数据准备好，read 会把数据从内核空间拷贝到用户空间。这个拷贝过程仍然是同步的，调用线程需要等待

I/O 多路复用用于在一个线程中处理多个 I/O 请求：应用程序把多个 socket 交给内核监听。当某个 socket 的数据准备好时，内核通过事件通知应用程序。应用程序再调用 read 读取数据

无论是阻塞 I/O、非阻塞 I/O，还是 I/O 的多路复用都是同步 I/O，数据拷贝阶段由用户线程等待完成。异步 I/O 是内核自动将数据从内核空间拷贝到用户空间。

# I/O 多路复用

Reactor 是一种非阻塞同步网络模型。它的核心思想是：Reactor 负责监听事件和分发事件，真正的业务处理交给 Handler 完成。

单 Reactor 单线程

Reactor 监听连接事件和读写事件。
有新连接时，调用 accept 建立连接。
有读写事件时，交给对应 Handler 处理。
Handler 完成 read -> 业务逻辑 -> send。

单 Reactor 多线程：把业务处理分发出去

Handler 通过 read 读取数据。
Handler 将数据交给子线程 Processor。Processor 处理业务逻辑，将结果交回 Handler。
Handler 调用 send 返回响应。

多 Reactor 多线程：把连接事件和读写事件分离

主线程中的 MainReactor 监听新连接，有新连接到来时，接收连接，将连接分配给某个子线程。
子线程中的 SubReactor 继续监听该连接的读写事件。
有读写事件发生时，SubReactor 分发给对应 Handler。
Handler 完成读取、业务处理和发送响应。

Nginx 采用的是多 Reactor 多进程方案。

# 零拷贝

传统 read + write 的过程
```
用户进程调用 read()，从用户态切换到内核态
磁盘通过 DMA 将文件数据拷贝到内核缓冲区
CPU 将数据从内核缓冲区拷贝到用户缓冲区
read() 返回，从内核态切换回用户态
用户进程调用 write()，再次从用户态切换到内核态
CPU 将用户缓冲区的数据拷贝到 socket 缓冲区
网卡通过 DMA 将 socket 缓冲区的数据拷贝到网卡
write() 返回，从内核态切换回用户态
```

```
buf = mmap(file);
write(socket, buf, len);
```
相比 read + write，它减少了一次 CPU 拷贝：不再需要把数据从内核缓冲区拷贝到用户缓冲区。应用程序可以直接访问映射后的内核缓冲区数据。调用 write() 时，内核再把数据从内核缓冲区拷贝到 socket 缓冲区

```
sendfile(socket_fd, file_fd, ...);
```

专门发送文件的系统调用函数 sendfile() 只有 2 次上下文切换，和 3 次数据拷贝（减少两次上下文切换）

从 Linux 内核 2.4 版本开始起，如果网卡支持 SG-DMA（和普通的 DMA 有所不同），调用sendfile()，网卡的 SG-DMA 控制器就可以直接将内核缓存中的数据拷贝到网卡的缓冲区里（减少1次cpu拷贝）

这就是所谓的零拷贝（Zero-copy）技术（只有dma拷贝）。零拷贝技术可以把文件传输的性能提高至少一倍以上

# ptrace

```
当你执行 gdb ./a.out 时，GDB 会先 fork 一个子进程，然后通过 execve 加载目标程序，但不会立即运行；
如果调试已运行的程序（gdb -p pid），GDB 会调用 ptrace(PTRACE_ATTACH, pid, ...)，内核会暂停目标进程，并将其控制权交给 GDB；
GDB 会成为目标进程的 “父进程”，目标进程的所有状态变更（如信号、异常）都会通知 GDB。
```

```
断点：
GDB 先读取目标地址处的原始机器指令（如 1 字节的 nop 或其他指令）并保存；
将该地址的指令替换为 int 3（x86 架构）—— 这是一个 1 字节的中断指令，执行到此处会触发 SIGTRAP 信号；
当目标程序运行到 int 3 时，内核会暂停程序，并通知 GDB；
GDB 收到信号后，会将 int 3 还原为原始指令，方便后续单步执行或查看变量。
```

```
当你执行 step 时，GDB 会通过 ptrace 告诉内核：让目标进程仅执行一条指令就暂停：
调用 ptrace(PTRACE_SINGLESTEP, pid, ...)；
内核执行目标进程的一条指令后，立即发送 SIGTRAP 信号，GDB 接管控制权；
```

```
查看变量：GDB 先通过符号表（编译时 -g 选项生成）找到变量对应的内存地址 / 寄存器，再调用 ptrace(PTRACE_PEEKDATA, pid, 地址, ...) 读取数据，解析后展示给用户；
修改变量：调用 ptrace(PTRACE_POKEDATA, pid, 地址, 新值, ...) 直接修改目标进程的内存 / 寄存器数据
```

# kvm

KVM 管 CPU / 内存；QEMU 管 I/O（磁盘 / 网卡 / USB）；

依赖 Intel VT-x / AMD-V 硬件虚拟化

```
Guest 应用（Non-root 模式 Ring3）→ 系统调用 → Guest 内核（Non-root 模式 Ring0）；
Guest 内核执行 in/out I/O 指令 → 触发 VM Exit（因为 Non-root 模式的 Ring0 没真权限）；
切回 Root 模式的 KVM（真 Ring0）→ KVM 把 I/O 请求转给 QEMU；
QEMU 模拟硬盘操作 → 结果通过 VM Entry 回传给 Guest 内核；
Guest 内核把结果返回给 Guest 应用。

宿主机的硬件中断（比如网卡、时钟）只会发给 Root 模式的 KVM
KVM 会把需要转发的中断（比如 Guest 的时钟中断）“模拟” 后，通过 VMEntry 注入到 vCPU 中，保证 Guest 只能收到自己的中断，宿主机中断和其他 Guest 的中断不会干扰它。
```

QEMU 首先打开 /dev/kvm 设备文件（KVM 暴露给用户态的接口），通过这个接口向 KVM 发送指令，比如 “创建一个新的 VM 实例”。

CPU / 内存：QEMU 告诉 KVM“需要 2 个 vCPU、4G 内存”，KVM 负责初始化 vCPU 线程（每个 vCPU 对应一个 Linux 线程）、设置内存的 EPT/NPT 地址映射；

```
它虚拟化的是 x86 CPU，只要能在物理电脑上运行的 x86 OS，KVM 基本都能跑。
```

```
Guest 虚拟地址（GVA）→ Guest 物理地址（GPA）→ 宿主机物理地址（HPA）
```

I/O 设备：QEMU 自己模拟磁盘、网卡、显卡等，比如把主机的 ubuntu.img 文件映射成 VM 的 “硬盘”，把虚拟网卡桥接到主机的物理网卡。

QEMU 调用 KVM 的 KVM_RUN 接口，触发 vCPU 线程进入 “非根模式”（VM 运行模式），VM 开始执行指令（Guest OS）；同时 QEMU 接管 VM 的输入输出（比如键盘、鼠标、显示器），你可以通过 QEMU 的窗口操作 VM。

```
半虚拟化 I/O —— Virtio
Guest 装了 virtio 驱动，知道自己是虚拟机。
Guest 把一堆数据包放进共享内存
只触发 一次 VM Exit 通知 QEMU
QEMU 直接去共享内存一次性取走所有数据
处理完再写回共享内存，通知 Guest
```

```
设备直通 —— PCI Passthrough
直接把一个 PCI 设备（网卡、GPU、NVMe）给虚拟机独占使用。
```

```
Type-2 型：Hypervisor 以内核模块的形式加载到系统内核中，利用操作系统的硬件驱动和资源管理能力。
KVM、VirtualBox、VMware Workstation
整个宿主 OS 内核都运行在根模式下
Type-1 型：Hypervisor 直接安装在物理硬件上，它本身就充当了“微型操作系统”的角色，直接管理 CPU、内存和硬件
Hyper-V
只有 Hypervisor 运行在根模式
学术界和工业界更倾向于将 KVM 归类为 Type-1。因为 kvm.ko 模块会将 Linux 内核从“通用 OS”提升为“Hypervisor”。KVM 是目前全球云厂商使用最广泛的技术。
```

```
启用 Hyper-V 后，它本身就是一个裸金属管理程序，宿主机 Windows 本身也会运行在 Hyper-V 之上的一个特殊虚拟机里。
当用户在 Windows 中启动 WSL 2 时，系统实际上是在后台静默启动了一个基于 Hyper-V 架构的虚拟机。所有 WSL 2 实例都共享同一个 Linux 内核。每个在 VM 内部实际上是通过 Linux Namespaces 进行隔离的。
当你安装 Docker Desktop 时，它会创建一个名为 docker-desktop 的特殊 WSL 2 实例。你运行的所有 Docker 容器，其实都是在这个 WSL 2 实例的内核里跑。
```

创建虚拟机时，在虚拟机和主机都创建了网卡

```
二层转发
桥接模式：创建网桥，把虚拟机主机侧网卡和一个主机物理网卡变成网桥端口，虚拟机网关是路由器
子网通信：虚拟机-虚拟交换机-br0
跨网通信：虚拟机-虚拟交换机-物理网卡-路由器
虚拟机网关mac不是主机，数据包从物理网卡出去
```

```
三层转发
NAT 模式：创建网桥，把虚拟机主机侧网卡变成网桥端口，虚拟机网关是br0
子网通信：虚拟机-虚拟交换机-br0
跨网通信：虚拟机-虚拟交换机-FORWARD-物理网卡
```

仅主机模式：NAT 模式不FORWARD

```
Docker 的 Bridge 模式 = 虚拟机的 NAT 模式
当你启动容器而不指定net时，在容器和主机都创建了网卡，把容器主机侧网卡变成网桥端口，网关是网桥（默认是docker0）
当你使用 -p 8080:80 启动容器时，Docker 会在宿主机的 iptables 里设置规则

Host 模式：The container shares the host’s network interfaces, IP addresses, and ports (no port mapping is needed).
```

# 包管理

apt update从配置的软件仓库下载最新的包索引（package index），描述仓库中所有可用包的元数据

Ubuntu LTS：版本冻结于发布时，5 年内仅小版本更新

Ubuntu 非 LTS：版本较新，但生命周期短（9 个月）

Arch：始终接近上游最新版（延迟 1~4 周）

Arch 仓库的隐含契约："今天仓库中的所有包，彼此兼容"，但不承诺："今天的包与昨天的包兼容"

Do not use pacman -Sy package without -u. This will synchronize the package list and install the specified package without upgrading the rest of the system. This can lead to dependency issues and break the system.

```
# 用户执行
$ sudo pacman -Sy firefox
# 内部发生：
1. 同步仓库：发现 firefox 126 依赖 libvpx 1.14
2. 但系统当前为 libvpx 1.13（未升级）
3. pacman 强制安装 firefox 126 + libvpx 1.14
4. 结果：
   - firefox 126 运行正常
   - 但 chromium 125 仍链接 libvpx 1.13 → 崩溃
   - 系统进入"半残废"状态（部分软件工作，部分崩溃）
```

滚挂

```
部分更新导致依赖断裂
新版软件要求新配置格式，但 pacman 保留用户修改的旧配置，需用户手动合并
仓库缺陷，需等待 Arch 团队负责修复
闭源驱动不兼容，需等待驱动更新
硬件特定问题，需内核参数调整
```

PPA 是 Ubuntu 官方源的 “补充仓库”，由社区开发者 / 团队维护，用来提供官方源里没有的软件包；

apt: 下载 python3.11-dev.deb，解决依赖

dpkg: 解包，放文件到系统，注册安装信息

apt 装的软件在 /usr/bin（系统级），脚本 / 编译装的软件在 /usr/local/bin（本地自定义级）

# ...

## awk

```
awk -F 'loss: ' '{print $2}' file
按'loss: '分割，取第2列
```

## cgroup

```
cat /sys/fs/cgroup/memory/memory.limit_in_bytes
```

## chown

```
id
uid=1009(cyy) gid=1009(cyy) groups=1009(cyy),100(users),988(docker)
chown -R 1009:1009 .
读（r=4）、写（w=2）、执行（x=1）
chmod -R xxx .
```

## diff

```
diff -r -x "node_modules" -x ".git" -x "*.log" dir1 dir2
-x==--exclude
换行符不同也算不同
```

## dpkg

```
-l list
-L pack 一个包安装了哪些文件
-S /usr/bin/python3.11 某个文件属于哪个包
```
## du

```
du -sh ./*
df -h
```

## find

```
find / -name "*.ttf" -o -name "*.otf" 2>/dev/null | grep -Ei "wqy|cjk|noto|simhei"
-o==--or

find / -type f -name "*trace_system*" 2>/dev/null

find /opt/conda -samefile /opt/conda/lib/libcudart.so.12.1.105
检查硬链接（共享相同的 inode 和数据块）
```

## font

```
apt update
apt install fontconfig
fc-list :lang=zh
apt install fonts-noto-cjk
rm /root/.cache/matplotlib/ -rf 
```

## grep

```
grep -rni --exclude="*.log" "kk" /usr/
grep -rni --exclude-dir=build "distributed_strategy.proto" .
grep -rni --include="*.h" "ShardingMergeForTensors" Paddle/
grep -A 10 "jsd"
grep -B 10 "jsd"
grep -C 10 "jsd"
grep -v -e "tom" -e "lisi" names.txt 
既不包含"tom"也不包含"lisi"的行
```

## ip

```
ip addr show
ip addr show eno1
```

## ln

```
ln -s local/observation observation
```

## nc

nc 监听tcp端口，发起tcp连接，发送udp

curl 非交互式
```
curl -I https://www.google.com
curl -X POST "$API_URL" \
      -H "Content-Type:application/json" \
      -d "{ \"message\": { \"body\": [ { \"type\": \"TEXT\", \"content\": \"${MESSAGE}\" } ] } }"
curl -L -o 文件名 URL
-L：跟随重定向
```

wget 专注下载
```
python -m http.server 8001
ifconfig | grep "inet 10"
unset http_proxy && unset https_proxy
wget http://10.95.237.97:5675/xx
```

```
scp 是基于 SSH 的文件传输工具
scp -r -P 2222 ./local_file.txt user@remote_host:/remote/path/

SFTP 是运行在 SSH 之上的文件传输协议，列目录、上传、下载、删除、重命名
sftp user@example.com
```

## nohup

```
DEBUG=1 MAX_JOBS=16 nohup python -m pip install --no-build-isolation -v -e . > cyyinstall.log 2>&1 &
```

## pkill

```
ps -aux
pkill -9 -f ".*run_pretrain_auto.py"
-f==--full
```

## split

```
split -b 50M -d big big_p
-d数字后缀
cat big_p* > big
```
## ss

socket statistics

```
ss -tulnp | grep sshd
-t  显示 TCP socket
-u  显示 UDP socket
-l  只显示监听状态的 socket
-n  不解析服务名，直接显示数字端口
-p  显示使用该 socket 的进程信息

Netid State  Local Address:Port  Peer Address:Port  Process
tcp   LISTEN 0.0.0.0:22          0.0.0.0:*          users:(("sshd",pid=1234,fd=3))
udp   UNCONN 0.0.0.0:53          0.0.0.0:*          users:(("dnsmasq",pid=567,fd=5))
```

## tar

只打包不压缩：只是把多个文件合并成一个文件

```
tar -cf xx.tar xx
tar -xf xx.tar -C out/
```

.gz只压缩，不打包，只能压缩单个文件

```
gzip -dk file.gz
-d：decompress，解压
-k：keep，保留输入文件
```

.tar.gz/.tgz/.zip（= tar + gzip）

## watch

```
watch -n 3 'ps -aux | grep -E "run|python"'

watch -n 2 'python -c "import sys; print(int(sys.argv[1])/1024/1024/1024,\"GiB\")" `cat /sys/fs/cgroup/memory/memory.usage_in_bytes`'

alias wm='watch -n 2 '\''python -c "import sys; print(int(sys.argv[1])/1024/1024/1024,\"GiB\")" `cat /sys/fs/cgroup/memory/memory.usage_in_bytes`'\'
```