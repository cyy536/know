# docker

```
dockerd 镜像构建、网络配置
containerd 下载镜像、启动 / 停止 / 删除容器
(nvidia-container-runtime)
runc 根据配置创建容器，启动完容器进程就退出
```

```
docker客户端->HTTP + JSON->dockerd 
RESTful是约定：用 URL 表示东西，用 HTTP 方法表示干什么
POST /containers
数据: { "image": "nginx" }
```

```
dockerd 要发消息
用 gRPC 定义 CreateContainer 服务
把参数用 Protobuf 打成二进制
装进 HTTP/2 数据包
通过 Unix Socket 发给 containerd
```

## ...

```
/etc/systemd/system/docker.service.d/http-proxy.conf
[Service]
Environment="HTTP_PROXY=http://172.18.217.148:10809/"
Environment="HTTPS_PROXY=http://172.18.217.148:10809/"
Environment="NO_PROXY=localhost,127.0.0.1,.local"

sudo systemctl daemon-reload
sudo systemctl restart docker
sudo systemctl show docker --property Environment
```

```
docker save -o image.tar 镜像名:标签
docker load -i image.tar
```

```
--privileged：开启所有权限
--cap-add=SYS_ADMIN：容器内的进程就被允许执行许多原本只有宿主机 Root 才能做的操作
```

```
docker exec，ssh容器，vscode ssh的环境变量不同
```

# k8s

k8s 集群由两种角色组成：Master Node，Worker Node

```
kube-apiserver： 其他所有组件都不直接相互通信，而是通过与 API Server 交互来协同工作。
etcd： 一个高可用的键值数据库，持久化存储整个集群的所有配置和数据（如Pod、Service、Node等信息）。
kube-scheduler： 调度器，负责决定将新创建的 Pod 放在哪个 Node 上运行。
kube-controller-manager： 控制器管理器，运行着各种控制器，这些控制器不断地监控集群状态，确保其与期望状态一致（例如，确保Pod副本数始终符合要求）。
```

```
kubelet：它是跑在每个 Node 上的代理，确保容器按照 PodSpec（说明书）运行。如果容器挂了，kubelet 负责把它重启。
kube-proxy：它负责维护节点上的网络规则，实现 Service 的负载均衡。
Container Runtime：containerd
```

```
你通过 kubectl 给 apiserver 发指令。
apiserver 把需求记在 etcd 里。
scheduler 发现有新活，选了个空闲的 Node，告诉 apiserver。
目标 Node 上的 kubelet 收到 apiserver 的通知，找 Container Runtime 拉镜像启动。
kube-proxy 配置网络，让别人能访问到这个 Pod。
```

```
常规 Pod (如 Deployment 启动的)：通常被设计为“永不停止”。比如推理服务
Job 管理的 Pod：被设计为“完成任务就下班”。比如训练任务
```

# 云计算

云服务器提供计算资源

SaaS: 软件即服务，office365，chatgpt

IaaS: 基础设施即服务

amazon电子商务公司有闲置服务器，推出amazon web service

ec2可扩缩容虚拟机，s3对象存储

gpushare

PaaS: 平台即服务

数据库，程序运行时，api网关，版本控制工具

AWS Lambda 是一项 AWS 提供的无服务器 (Serverless)、事件驱动的计算服务。它允许开发者上传代码，无需预置或管理服务器，即可运行后端应用。

API 网关就是架在客户端与后端服务之间的那道“门”。

BMaaS：裸金属即服务

# gpushare

```
curl -L -o /usr/local/bin/oss https://gpucloud-static-public-prod.gpushare.com/installation/oss/oss_linux_x86_64
chmod u+x /usr/local/bin/oss
oss login
oss mkdir oss://datasets
oss cp xx oss://datasets/
oss ls -s 
oss rm xx(必须是准确名字)
```

