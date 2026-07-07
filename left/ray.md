Task（无状态任务）

```
# 定义时要求资源
@ray.remote(num_cpus=2, num_gpus=0.5)
def rollout():
    return
    
# 异步提交多个 rollout
futures = [rollout.remote("CartPole-v1", seed=i) for i in range(4)]
results = ray.get(futures)  # 阻塞等待所有结果
```

Actor（有状态对象）

```
# 创建 2 个 worker
workers = [RolloutWorker.remote("CartPole-v1") for _ in range(2)]

# 并行采样
trajs = ray.get([w.sample_trajectory.remote() for w in workers])
stats = ray.get([w.get_stats.remote() for w in workers])
```

```
Global Control Store (GCS)：这是 Ray 的“大脑”，存储了整个集群的状态（如节点元数据、任务状态等）。
Raylet：运行在每个工作节点上的守护进程。它包含两个核心模块：
    Node Manager：负责节点内的资源管理和任务调度。
    Object Store (Plasma)：这是一个分布式共享内存对象存储。它允许同一节点上的不同进程通过内存映射直接访问数据，无需拷贝。
    
Driver 是运行你编写的“主程序”的进程。
当你执行 ray.init() 时，当前的 Python 脚本就变成了一个 Driver。它负责定义 Task 和 Actor，并调用 .remote() 发起计算任务。它负责把任务发给集群并拿回 ObjectRef。它通过 ray.get() 从 Object Store 拿到结果。

Driver 向本地 Raylet 请求资源租约（Lease）。
如果本地没资源，本地 Raylet 会根据 GCS 中的负载信息，告诉发起者：“去 Node B 试试”。
发起者会直接连接到 Node B 的 Raylet。

Worker 是集群中专门负责执行被 @ray.remote 装饰的任务（Task）或类（Actor）的进程。
它的工作：它是一个被动角色。它坐在那里等 Raylet 分配任务。一旦拿到任务（函数名、参数、依赖），它就执行 Python 代码，并将结果存入本地的 Object Store。
```

```
Head 节点本质上也是一个 Ray 节点，它像普通节点一样运行着 Raylet、Plasma Store 和多个 Worker。
除了普通功能，它还额外“背负”了一些全局管理进程，其中最重要的就是 GCS，以及 Ray Dashboard（可视化监控）和 Ray Client Server（处理远程连接）。
Ray Client Server 在集群内部充当一个“影子 Driver”，替网络请求向本地 Raylet 申请资源、提交任务。
```

Ray 的一个节点对应的就是一台物理机或一个虚拟机或一个 Pod