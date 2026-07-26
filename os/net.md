# http

三部分组成：url，http预定义的dict，body

GET 请求的参数位置一般是写在 URL 中，URL 规定只能支持 ASCII。URL可以通过一种叫做百分号编码的方式来包含中文或其他非ASCII字符。

```
headers = {
  "Content-Type": "application/json",
  "Authorization": "Basic <base64(username:password)>",
}
headers = {
  "Content-Type": "text/html",
  "Authorization": "Bearer sk-xxxx",
  #OAuth2、LLM API Key
}
```

HTTP body 从协议层看是二进制字节；从应用层看，可以是 JSON、HTML、图片、文件、表单、Protobuf 等任何格式。

## tls

加密：公钥上锁，传输数据，私钥解锁。先非对称再对称。

数字签名：对数据进行哈希，私钥对哈希值进行处理，确保哈希值的正确性，防止中间人替换。

数字证书：公钥正确性。
警察局用他们自己的私钥对你父亲的公钥做了个数字签名，然后把你爸爸的「个人信息 + 公钥 + 签名公钥」打包成一个数字证书。

多级证书，从下向上请求证书，最上级是可以被本地证书的公钥验证的，从上到下验证证书

TLS: 消息被分割成多个较短的片段，然后分别对每个片段进行压缩。经过压缩的片段再加上消息认证码(哈希)会一起通过对称密码进行加密。

中间人服务器与客户端在 TLS 握手过程中，实际上发送了自己伪造的证书给浏览器，而这个伪造的证书是能被客户端识别出是非法的，于是就会提醒用户该证书存在问题。如果你的电脑中毒了，被恶意导入了中间人的根证书，浏览器是不会弹出证书存在问题的风险提醒的。使用抓包工具进行 HTTPS 抓包的时候，需要在客户端安装 Fiddler 的根证书。

## HTTP/1.1

开启了 HTTP Keep-Alive 机制后，TCP 连接就不会中断，而是保持连接。当客户端发送另一个请求时，它会使用同一个连接，一直持续到客户端或服务器端提出断开连接

HTTP/1.1 默认支持长连接，使得请求管道化成为可能。管道化允许客户端在同一条 TCP 连接上连续发送多个请求，而不必等待前一个请求的响应返回。服务器必须按照接收请求的顺序返回响应。由于存在队头阻塞、代理兼容性问题等，HTTP/1.1 管道化通常不是默认开启，主流浏览器后来基本没有实际启用，实际并发更多依赖建立多条 TCP 连接。HTTP/2 则通过多路复用更好地解决了这个问题。

## HTTP/2

HTTP/2 采用二进制分帧格式传输数据，HTTP 消息会被拆分为 HEADERS Frame、DATA Frame 等多个 Frame。

HTTP/2 使用 Stream 表示一次请求/响应交换。同一个请求和响应属于同一个 Stream；多个 Stream 可以复用在同一条 TCP 连接上，不同 Stream 的 Frame 可以交错发送。

HTTP/2 使用 HPACK 进行头部压缩，通过静态表、动态表和编码机制减少重复 Header 的传输开销。

HTTP/2 支持 Server Push，服务器可以在客户端请求 HTML 时主动推送 CSS、JS 等资源。不过该机制在现代浏览器和实际生产中已经基本不推荐使用。

HTTP/2 解决了 HTTP/1.1 层面的应用层队头阻塞，但由于它仍然基于 TCP，仍存在 TCP 层队头阻塞：当前面的 TCP 数据丢失时，后续已经到达的数据也不能越过它交付给应用层。

## HTTP/3

HTTP/3 将传输层从 HTTP/2 使用的 TCP+TLS，改为基于 UDP 的 QUIC。UDP 本身不可靠，但 QUIC 在其上实现了可靠传输、拥塞控制、丢包重传、流量控制等能力。

QUIC 也有 Stream 和多路复用的概念。不同 Stream 可以在同一个 QUIC 连接中并发传输；当某个 Stream 发生丢包时，通常只会阻塞该 Stream，不会像 HTTP/2 over TCP 那样阻塞整条连接上的所有 Stream。

QUIC 不像 TCP 那样主要依赖四元组来标识连接，而是使用 Connection ID 标记连接。移动设备从 Wi-Fi 切换到蜂窝网络后，即使 IP 地址或端口发生变化，只要保留连接上下文，例如 Connection ID、TLS 密钥等，就可以进行连接迁移，继续复用原连接。

## OAuth2

OAuth2 是一种授权协议/框架，用来让第三方应用在不拿到用户密码的情况下，访问用户在某个服务上的资源。

```
你用某个网站点击“使用 Google 登录”；
第三方应用把你带到 Google 授权页面；
你登录 Google 并同意授权；
Google 发给第三方应用一个 Access Token；
第三方应用拿这个 token 去访问允许范围内的资源。
```
## WebSocket

传统 HTTP/1.1 是请求-响应模型：客户端先发请求，服务端再响应。只靠普通 HTTP，服务端不能随时主动向浏览器推送消息。

为了实现类似服务器推送的效果，常见方式有轮询、长轮询和 WebSocket。

轮询：前端定时向服务器发送 HTTP 请求，服务器立即返回当前状态。缺点是请求多，实时性取决于轮询间隔。

长轮询：客户端发起 HTTP 请求后，服务器先挂起不立即返回。如果在超时时间内发生事件，例如扫码成功或登录确认，就立即响应；如果超时仍无事件，则返回空结果，客户端再马上发起下一次请求。

WebSocket 是一个基于 TCP 的应用层协议，适合客户端和服务端频繁双向通信。它通常先通过 HTTP/1.1 发起握手，携带 Upgrade 相关 Header；服务端返回 101 Switching Protocols 后，连接升级为 WebSocket。之后双方可以在同一条长连接上互相主动发送消息。

普通 WebSocket 使用 `ws://`，默认端口 80；安全 WebSocket 使用 `wss://`，即 WebSocket over TLS，默认端口 443。

# ssh

Telnet 和 SSH 都是远程登录协议，但 Telnet 明文裸奔，SSH 加密隧道

```
ssh user@hostname 

ssh -T -i ~/.ssh/id_rsa_qua git@github.com
-T: 选项，表示 "Disable pseudo-terminal allocation"

ssh user@hostname "command" < input_file.txt > output_file.txt
#执行命令：command < inut_file.txt > output_file.txt

git push origin main
ssh git@github.com \
  "git-receive-pack 'username/repo.git'" \
  < /tmp/git-pack-data
```

SSH配置里空白符是无效的，SSH配置解析器的工作流程：

```
初始化全局默认值（读取文件开头的无host配置）
遇到host指令时：
    创建一个新的配置上下文（Context）
    只继承全局配置，不继承其他host块的配置
优先使用当前host块的参数，没有则回退到全局默认值
```

```
StrictHostKeyChecking no
//known_hosts
Port=38276

Host 3090_flashattn
HostName 172.18.218.232
Port 58922
User root
RemoteForward 7897 127.0.0.1:7897

Host github.com
Hostname ssh.github.com
Port 443
IdentityFile ~/.ssh/id_rsa_qua
//ssh.github.com是 GitHub 专门提供的443端口
```

ssh -o 的 -o 是 option（选项） 的意思，用于在命令行中临时指定 SSH 配置项。它的作用和写在 ~/.ssh/config 或 /etc/ssh/ssh_config 中的配置完全一样，只不过只对当前这一次连接生效。

ssh有两种认证方式：密码，密钥

A免密ssh B: A的公钥存放在B的~/.ssh/authorized_keys

```
ssh-keygen -t rsa -b 4096 -C "your_email@example.com" -f ~/.ssh/id_rsa_qua
```

首次连接服务器时，SSH 会提示你确认服务器指纹（公钥的哈希值），确认后将该公钥存入 known_hosts。

再次连接时，SSH 会比对服务器的公钥与本地存储的是否一致。若不一致会发出警告（可能遭遇攻击或服务器配置变更）。

容器里面只能看到一个sshd，但可以连上两个，22端口应该是宿主机的sshd

```
ss -tulnp | grep sshd

telnet 127.0.0.1 38276
Trying 127.0.0.1...
Connected to 127.0.0.1.
Escape character is '^]'.
SSH-2.0-OpenSSH_9.6p1 Ubuntu-3ubuntu13.11

telnet 127.0.0.1 22
Trying 127.0.0.1...
Connected to 127.0.0.1.
Escape character is '^]'.
SSH-2.0-OpenSSH_7.4
```

# email

SMTP: 客户端 → 邮件服务器，或服务器之间

POP3: 从邮件服务器 → 客户端

IMAP: 在服务器上管理邮件，下载

网页版邮箱通过 HTTPS 与邮箱服务商的 Web 应用服务器交互，再由后端服务调用标准邮件协议完成实际的邮件收发。

# BitTorrent

```
.torrent
├── announce              # Tracker 地址，通常在 info 外面
├── announce-list         # 多 Tracker 列表，可选
├── creation date         # 创建时间，可选
├── comment               # 注释，可选
├── created by            # 创建软件，可选
└── info                  # 核心资源信息
    ├── name              # 文件/目录名
    ├── piece length      # 分片大小
    ├── pieces            # 每个分片的 SHA-1 哈希拼接
    ├── length            # 单文件模式：文件大小
    └── files             # 多文件模式：文件列表与大小
```

InfoHash 是对 .torrent 文件中的 info 字段进行 SHA-1 哈希运算得到的结果。

Tracker 是一台协调服务器，但它不保存文件内容，只负责帮 Peer 互相发现。

```
1. 你的 BT 客户端连接 Tracker
2. 告诉 Tracker：我想下载这个 InfoHash，我的 IP 和端口是这些，我当前进度是多少
3. Tracker 返回一批也在下载或做种这个资源的 Peer 地址
4. 你的客户端再去连接这些 Peer，真正的数据交换发生在 Peer 之间
```

一旦你拿到了 Peer 列表，你的客户端就会主动发起多个 TCP 连接。如果你只下载不上传，别的 Peer 就会“扼杀”发给你的数据流。

现在很多 BT 下载不完全依赖 Tracker，而是使用 DHT。DHT 可以理解为一个分布式的 Peer 发现网络。


```
1. 你想下载 InfoHash 为 H 的资源
2. 客户端先询问自己已知的 DHT 节点：谁知道 H 对应的 Peer？
3. 对方如果不知道，就返回一些在逻辑距离上更接近 H 的节点
4. 客户端继续迭代询问更近的节点
5. 最终找到负责保存 H 附近信息的一批 DHT 节点
6. 这些节点返回正在下载或做种该资源的 Peer 地址
7. 你的客户端再连接这些 Peer 下载数据
```

磁力链接的核心通常就是 InfoHash，客户端拿到磁力链接后，会通过 DHT、Tracker 查找资源对应的 Peer，再从 Peer 那里下载 torrent 元数据，最后根据元数据下载真正的文件分片。

# tcp

IPv4 + TCP:
1500 - 20(IP头) - 20(TCP头) = 1460 字节 MSS

listen 0.0.0.0，那么它表示本机上的所有IPV4地址。

每一个socket执行listen时，内核都会自动创建一个半连接队列和全连接队列。半连接队列，也称 SYN 队列；全连接队列，也称 Accept 队列。当服务端接收到客户端的 SYN 报文时，会创建一个半连接的对象，将其加入到内核的「SYN 队列」，服务端接收到 ACK 报文后，从「SYN 队列」取出一个半连接对象，然后创建一个新的连接对象放入到「Accept 队列」；应用通过调用 accpet 接口，从「Accept 队列」取出连接对象。

服务端接收到了 FIN 报文，TCP 协议栈会为 FIN 包插入一个文件结束符 EOF 到接收缓冲区中，应用程序可以通过 read 调用来感知这个 FIN 包。调用 close 关闭它的套接字，发出一个 FIN 包。

当服务端的进程崩溃或在 kill 掉进程后，挥手过程也都是在内核完成

在 TCP 头部「选项」字段里加一个 SACK 的东西，它可以将已收到的数据的信息发送给「发送方」，只重传丢失的数据。接收方不需要按顺序接收。如果不启用 SACK，就必须重传丢失包之后的每个数据包。

flow control and congestion control，window size受两部分影响

拥塞控制：加性增加，乘性减少

慢启动：当发送方每收到一个 ACK，拥塞窗口 cwnd 的大小就会加 1。

拥塞避免：每当收到一个 ACK 时，cwnd 增加 1/cwnd。

超时重传：ssthresh 设为 cwnd/2，cwnd 重置为 1 

快速重传不以时间为驱动，而是以数据驱动重传，收到三个相同的 ACK 报文

# ip

## nat

都在nat里面，都连接到外部服务器，让服务器当中间人传递数据

都在nat里面，互相通过外部服务器获取nat ip和port，再直接通信

## dhcp

获取 ip、mask、gateway_ip、dns_ip

封装在udp包，ip和链路层都是广播

ip广播不能跨子网，网关被配置了“中继功能”。它看到是 DHCP 请求，不会丢弃它，而是把这个广播包重新封装成一个正常的单播 IP 包。网关在单播包的源地址填上自己的接口 IP，目的地址填上远在核心机房的 DHCP 服务器真实 IP。服务器处理完后，把分配的 IP 发回给网关，网关再通过本地广播发给你的电脑。

## dns

客户端（应用）对 本地 DNS 服务器（LDNS） 发起的就是递归查询。

LDNS 去询问 根域名服务器 或 权威域名服务器 时，通常采用迭代查询。

CNAME（Canonical Name，规范名称记录）是一种DNS记录类型，用于将一个域名（别名）映射到另一个域名（规范名称）。LDNS 收到 CNAME 后，会自动启动一轮新的查询。它会把原来的域名放下，转而去查那个“真名”

CDN是一种通过在各地部署节点服务器，将网站的静态资源（如图片、视频、JS/CSS文件）缓存到用户身边的技术。阿里云在电信、联通、移动的各个省份机房里，租用机位并放了一堆“缓存服务器”，它在网络逻辑上通常拥有电信的 IP 地址。

```
当你输入 www.taobao.com 时，DNS 解析并不是直接给你一个固定的 IP，而是经历了一个动态分配的过程：
淘宝的域名在 DNS 后台其实指向了一个 CNAME，比如 taobao.com.danuoyi.tbcache.com。
LDNS 问调度系统（特殊的权威 DNS）：taobao.com.danuoyi.tbcache.com 的 IP 是什么？
调度系统会看你的请求是从哪来的。调度系统发现杭州电信机房里刚好有几台缓存服务器（CDN 节点），而且负载很低。于是，它就把这台电信机房服务器的 IP 给了你。
真实的源站 IP 躲在 CDN 后面，只有 CDN 节点知道它在哪里。
```

个人使用CDN服务

```
当用户访问你的域名时，他先问根服务器。根服务器说：“去问 .com 的管家（顶级域名服务器 TLD）。”
当你修改了原域名商的 NS 后，原域名商会向 TLD 服务器发送一个通知：“以后 example.com 的事儿不归我管了，去问 cloudflare.com。”
TLD 服务器会更新它内存里的 NS 记录。
修改前：记录的是 dns1.hichina.com (阿里云)。
修改后：记录的是 ashley.ns.cloudflare.com (Cloudflare)。
```

```
glibc.getaddrinfo()
根据/etc/nsswitch.conf
hosts: files resolve [!UNAVAIL=return] dns
files: 先看 /etc/hosts
resolve: 这代表调用 nss-resolve 模块。这个模块会通过 D-Bus 直接去问 systemd-resolved 服务。
dns: 如果 systemd-resolved 没开，再根据 /etc/resolv.conf 中指定的 DNS 服务器地址发送查询请求。

/etc/hosts：1.1.1.1 www.taobao.com
systemd-resolved 接收到请求后，会进行缓存、并发查询或加密解析 (DoT/DoH)。
操作系统内核在 DHCP 租约达成后，会将 DNS 信息直接写入 /etc/resolv.conf 或交给 systemd-resolved

DoH：无视 DHCP 提供的 DNS 服务器，建立一个加密连接（HTTPS）到特定的服务器（如 Cloudflare 的 1.1.1.1 或 Google 的 8.8.8.8）。

dig 直接与 DNS 服务器通信。（需要读取系统的 DNS 设置）

浏览器缓存：先看自己有没有存过这个域名的 IP。
```

## 路由协议(很复杂，可忽视)

AS (自治系统)：是一个管理层面的概念。一个 AS 内部通常包含成千上万个子网，拥有成百上千台路由器。它是一组由同一个管理机构（如运营商、阿里云、大学）控制的 IP 网络和路由器的集合。

as内路由协议：rip、ospf

边界路由器 (ASBR)：这些设备驻守在 AS 的边境上，运行 BGP 协议

互联网是由数万个 AS 组成的。BGP 协议告诉全世界，某个 IP 段现在在哪一个 AS 手里，以及经过哪些 AS 可以到达那里。

中国电信在跨越太平洋时，走的是自己参与投资、自己拥有使用权的光纤。

```
			 [ 互联网金字塔顶端 ]
                  Tier 1 ISP 
            (万能中转：Lumen, NTT)
             /                  \
            /                     \  (仅当无直连时走这里)
           /                        \
          /                          \
   [ 中国电信国内 ] <====物理连接====> [ 中国电信美国路由器 ]
     (AS4134/4809)                 (电信在美国的PoP点)
        /      \                       |
       /        \                      |
    [个人]    [杭州阿里云] <----------> [ 美国阿里云 ] 
    (你)      (AS45102)               (AS45102)
```

```
当一个位于日本的用户想要访问阿里云（AS45102）时，日本运营商（如 NTT）的路由器会看到两条路：
去往上海/杭州方向的 AS45102（物理距离近，BGP 跳数少）。
去往美国西海岸方向的 AS45102（物理距离远，BGP 跳数可能多一跳）。
路由器会选择“逻辑跳数（AS_PATH）最短”的那个出口。

阿里云的 AS45102 并不是一个连续的物理实体，而是分布在全球各地的“基地”。阿里云在杭州和美国边界之间建立了一个逻辑隧道。数据包虽然物理上穿过了别家的路由器，但因为被“信封”（封装协议）包裹着，中间的路由器只管搬运，不看内层 IP。在逻辑上，数据包从未离开过 AS45102。
```

在物理上，IXP 通常是一个放置了大量高性能交换机的机房。阿里巴巴、中国电信等，都会把自己的光纤拉到这个机房，插在 IXP 的交换机上。

## igmp

数据源并不关心有多少个接收者。它只需将数据包封装好，并在 IP 头的目的地址处填写一个 D 类多播地址

路由器之间通过多播路由协议构建出一条从源到所有接收者的路径。

接收者通过 IGMP 协议通知本地路由器，表明自己希望加入或退出某个多播组。路由器只将多播流量转发到有成员存在的端口

IGMP 封装在 IP 数据包

目标MAC：前 24 位01-00-5E，第 25 位固定为 0，取出 IP 地址的后 23 位，直接填充到 MAC 地址的后 23 位中。

传统 IPTV 使用组播，互联网直播使用的是单播，配合 CDN。

运营商在自己的内网中构建了完整的多播树。组播在跨运营商、跨地域的公网上很难实现。因为公网上的路由器属于不同公司，很多路由器为了安全或性能考虑，默认是关闭组播转发功能的。

## icmp

ICMP 报文是封装在 IP 包里面，一类是用于诊断的查询消息，另一类是通知出错原因的错误消息

ping是一个利用 ICMP 协议来测试网络连通性的应用程序。

ping 本机IP，相关的网络数据，都是走的 lo0

linux 有多个路由表，按优先级查：

```
0:      from all lookup local
32766:  from all lookup main
32767:  from all lookup default
```

当你 ping 192.168.1.20，如果 192.168.1.20 是本机地址，会匹配到：

```
local 192.168.1.20 dev eth0 scope host
```

traceroute 故意设置特殊的 TTL，来追踪去往目的地时沿途经过的路由器，当 TTL 减到 0 时，路由器会丢弃这个包，并向发送者回发一个 ICMP 超时报文。

traceroute 在发送 UDP 包时，会填入一个不可能的端口号值作为 UDP 目标端口号，目标主机会向你回发一个 ICMP “目的端口不可达”。

traceroute 还有一个作用是故意设置不分片，从而确定路径的 MTU。

# ipv6

IPv6 没有广播；需要一对多通信时主要用组播。

任播地址会配置在多个节点/接口上，发送到这个地址的数据包会被路由到“拓扑意义上最近”的那个节点。

# eth

CSMA/CD 是以太网最初用来解决 “多人抢一根线” 的核心机制，是早期半双工以太网的标配规则。

监听，空闲才发，检测碰撞，冲突就停，随机等待再重发

最晚检测到碰撞的时间t=2l/c，这时候不应该发完，否则不能判断是否碰撞，这规定了最短帧长=发送速度*t

当增大网线速度时，t要减少，所以网线长度要减少

交换机是全双工，CSMA/CD 被关闭

交换机之间互连容易形成环路（loop），必须跑 STP（生成树协议）

选举根桥，每台交换机计算到根桥的最优路径，阻塞多余端口，只保留一棵树状结构，只有树上的端口才能转发数据包，被阻塞的端口只收 STP 报文
```
0x0800 = IPv4
0x0806 = ARP
0x86DD = IPv6
```
## vlan

VLAN 是一种逻辑切分技术。它能把一台物理上的交换机，变成多台独立的虚拟交换机。

当一个数据帧（Frame）进入支持 VLAN 的交换机时，交换机会给它贴上一个“隐形标签”——VLAN ID（根据 IEEE 802.1Q 标准）。

隔离规则： 交换机在转发时会检查标签。只有标签一致（同一个 VLAN ID）的端口，数据才能互通。

Trunk 是一条能够同时承载多个 VLAN 流量的“超级通道”。

假设你有两台交换机（Switch A 和 Switch B），每台交换机上都有 VLAN 10（研发部） 和 VLAN 20（市场部）。你只需要一根物理网线。通过在两端的接口上配置 Trunk 模式，这根线就能同时传输所有 VLAN 的数据。

```
如何让 VLAN 10 访问 VLAN 20？这就必须经过三层设备
单臂路由 (Router-on-a-Stick)
交换机和路由器之间只接一根物理线。这根线配置为 Trunk 模式。
路由器在那根线上开出多个“虚拟子接口”（如 eth0.10 和 eth0.20），每个子接口分别作为 VLAN 10 和 VLAN 20 的默认网关。
VLAN 10 的包发给网关（路由器子接口 1）。
路由器拆掉 VLAN 10 的标签，查看路由表。
路由器贴上 VLAN 20 的标签，从原路（同一根线）发回给交换机。
交换机根据标签把包传给 VLAN 20。

使用具备路由功能的交换机。它内部可以创建虚拟的路由器接口
```

# wifi

csma/ca：监听，空闲等待一段随机时间发，发之前先发 RTS，收到 CTS 才真正发数据，发完等 ACK 确认

Wi‑Fi 4，支持 2.4G+5G；Wi‑Fi 5，只跑 5G

2.4G Wi-Fi = 2401MHz ~ 2484MHz 这一小段

Wi‑Fi 把 2.4G / 5G 频段切成了一段一段固定的小频段，每一段就叫一个信道（Channel）。

2.4G 切出了 1~14 号信道，每个信道占 22MHz 宽度，所以会互相重叠。2.4G 只有 1、6、11 三个信道完全不打架。

5G 频段宽、信道多

AP是一个​物理硬件设备​​，是Wi-Fi技术的具体实现者。它的核心作用是​将有线网络信号转换为无线信号

AP会周期性广播SSID（Wi‑Fi 的名字），开多个 Wi‑Fi，每个 Wi‑Fi 打上不同 VLAN 标签

多个 AP 独立发射信号、共同覆盖一片区域。所有 AP 都直接连交换机，不互相转发数据包。相邻 AP 之间要错开信道，互不干扰。

当网络规模很大，需要部署几十、上百个AP时，逐个去配置和管理每个AP将非常繁琐。AC就是一个​集中化管理平台​（可以是软件或硬件设备）

# 5g

# bluetooth

全栈

# rdma

```
InfiniBand      原生支持 RDMA 的专用网络
RoCE            RDMA over UDP/IP
iWARP           RDMA over TCP/IP
```

虽然 RoCE v2 封装在 UDP/IP 中，但它的处理完全由支持 RoCE 的智能网卡和用户态驱动（如 libibverbs）完成，内核的 TCP/IP 协议栈并不解析或处理这些 UDP 包

Mellanox ConnectX 系列网卡是双模网卡，既支持标准以太网（通过`xgbe0`），也支持 RDMA（通过`mlx5_0`）。

RDMA 要求应用在使用一段内存前显式注册内存区域，也就是 Memory Region, MR。
注册 MR 时，内核驱动通常会做这些事：
```
锁定相关物理页，防止被换出到磁盘
建立虚拟地址到物理地址/DMA 地址的映射
把映射关系提供给 RDMA 网卡使用
生成访问权限和 key，比如 lkey、rkey
```

QP, Queue Pair 是 RDMA 通信中的核心对象，可以理解为一个通信端点，作用有点像 socket。发送方把 Work Request, WR 提交到 Send Queue。接收方通常要提前在 Receive Queue 里投递接收缓冲区，用来接收对端发送来的数据。WR 被放入队列后，通常会变成网卡可识别的 WQE, Work Queue Element。网卡异步执行这些请求。执行完成后，会在 CQ, Completion Queue 中生成 CQE, Completion Queue Entry。应用程序可以轮询 CQ 来获取完成通知。

```
1. 应用注册本地内存，得到 lkey
2. 应用从对端获得远端地址和 rkey
3. 应用创建并连接 QP
4. 应用提交 RDMA Write WR 到 Send Queue
5. 网卡根据 WR 描述从本地内存 DMA 读取数据
6. 网卡生成 RDMA/底层网络协议头
7. 网卡把协议头和数据封装成网络包发出
8. 对端网卡收到后校验权限，并直接 DMA 写入远端 MR
9. 本端网卡在 CQ 中生成 CQE，通知写操作完成
```

建立 RDMA 通信前，双方需要通过某种控制通道交换元数据；这个控制通道可以是 TCP 等

# vpn

vpn会添加路由信息

发送：OS->IP路由到虚拟网卡->虚拟网卡驱动.send->vpn处理->IP路由到物理网卡

http>tcp>ip(私有网络)>tls>tcp>ip(公有网关)

接收：物理网卡接收->OS->vpn处理->虚拟网卡驱动.recv->注入内核网络协议栈->OS

# proxy

## http

```
export https_proxy=agent.baidu.com:8188 && export http_proxy=agent.baidu.com:8188 && export no_proxy=baidu.com,baidubce.com,localhost,127.0.0.1,bj.bcebos.com
```

HTTP 代理 和 HTTPS 代理（即代理 HTTPS 流量）在“连接代理的方式”上，通常都是使用 HTTP 协议（明文）；
它们的本质区别在于：代理与目标服务器之间的通信内容和方式不同

http:

```
建立 TCP 到 proxy:8080
发送：
GET http://example.com/page HTTP/1.1
Host: example.com
代理解析 URL，建立 TCP 到 example.com:80
发送：
GET /page HTTP/1.1
Host: example.com
服务器返回响应，代理转发响应
```

https:

```
建立 tcp_A 到 proxy:8080
发送：
CONNECT secure.com:443 HTTP/1.1
Host: secure.com:443
代理建立 tcp_B 连接到 secure.com
收到：HTTP/1.1 200 Connection Established
直接发送 TLS ClientHello（原始字节）
代理从 tcp_A 读到这些字节，原样写入 tcp_B
用户和服务器建立 TLS 连接
```

## socket5

```
TCP 连接 proxy:1080
发送目标地址请求
代理 TCP 连接 example.com:80
双向透传字节

TCP 连接 proxy:1080
代理分配一个 UDP 中继地址
客户端向该地址发送封装后的 UDP 包
代理解析目标，将 DATA 转发到真实目标
返回的 UDP 包也经由代理回传
```

## shadowsocks

上面涉及3个实体，下面涉及4个，把代理拆分成client-server

```
[Client] ←→ [Shadowsocks Local] ←→ [Shadowsocks Server] ←→ [Target]
   ↑               ↑         (加密隧道)        ↑                ↑
应用层         本地 SOCKS5 代理             远程服务器         真实目标网站

TCP 连接 Shadowsocks Local
Shadowsocks Local TCP 连接 Shadowsocks Server
Shadowsocks Server TCP 连接 example.com:80
SS Local 先“拆开”你的 TCP 包，把里面的Payload（比如 HTTP 请求）拿出来，加密发出
```

## ssh

本地和代理服务器之间建立ssh隧道，将应用层数据作为ssh数据包内容，代理服务器与服务器建立TCP连接，双向转发

前两个是静态端口，target固定

本地端口转发：本地ssh监听5555端口，本地发起连接，代理

```
ssh -L 5555:127.0.0.1:3333 user@jump_host
远程侧：远程服务器上的 VS Code Server 监听 127.0.0.1:3333（仅允许本机访问），等待接收请求；
本地侧：本地启动 SSH 客户端进程，监听 localhost:5555，同时通过 22 端口和远程 sshd 建立加密 SSH 隧道；
数据传递：
本地 VS Code 向 localhost:5555 发送请求（比如「读取远程文件」）；
本地 SSH 客户端捕获 5555 端口的流量，加密后通过 SSH 隧道（22 端口）发给远程 sshd；
远程 sshd 解密流量，转发到本机的 127.0.0.1:3333（VS Code Server）；
VS Code Server 处理请求后返回响应，按原路（3333 → sshd 加密 → 隧道 → 本地 SSH 客户端解密 → 5555）回到本地 VS Code。
```

远程端口转发：代理服务器ssh监听3333端口，远程发起连接，反向代理

```
ssh -R 3333:localhost:5555 user@jump_host
3333 → ssh隧道 → 5555
```

动态端口转发：服务器不固定，需要SOCKS5协议指明服务器，ssh客户端作为SOCKS5服务器，接收请求目标，只支持封装tcp

```
ssh -D 1080 user@jump_host
你的浏览器（SOCKS5 客户端）连接到你电脑本机的 127.0.0.1:1080（SOCKS5服务器）
数据包通过已经建立好的 22 端口隧道传到代理服务器的 SSH 服务端
SSH 服务端向 github.com:443 发起一个真实的 TCP 连接。
浏览器和github握手
```

跳板机转发：经过中间人登录

```
ssh -J anhelor@172.18.218.215:2222 username@127.0.0.1 -p 2333
你的本地 SSH 客户端与代理建立第 1 个 SSH 连接。
你的本地客户端通过第 1 个连接发送指令，要求代理建立一个通往目标机端口（如 127.0.0.1:2333） TCP 连接。
你的本地客户端通过这条 TCP 通道，直接与目标机进行 SSH 握手，建立第 2 个 SSH 连接。

你想在容器里执行 ls。
你的 SSH 客户端用目标机的公钥对 ls 进行加密。
客户端把这个加密后的包，外面再套上一层用代理公钥加密的包，告诉代理：“把里面这个包裹送到目标机”。
代理拆掉最外层包装，把内层的包裹原封不动传过去。
目标机收到后，用自己的私钥解密，终于看到了 ls。
```

## frp
反向代理

FRP 通过“内网主动连接公网”建立长连接隧道，
公网服务器将用户请求封装后，通过该隧道转发给内网客户端，
内网客户端再将请求代理给本地服务，实现安全、透明的内网穿透。 

## v2ray

V2Ray (Project V) 引入了 VMess 协议，它本身会加密一次数据，如果你又开启了 TLS，就是二次加密

V2Ray 的原作者在网络上突然消失，由于开发组的其他人没有完整的仓库权限，项目更名 V2Fly 后继续进行开发

VLESS 是 VMess 的减负版，它本身不包含任何加密和校验，它把这些任务全部交给了底层的 TLS 层

V2Fly 社区内部，开发者们对于 XTLS 的许可协议产生了严重分歧。一部分开发者带着 XTLS 技术离开了 V2Fly 社区，另起炉灶创建了 Project X，其核心成果就是 Xray

v2rayN 和 v2rayNG 使用 Xray 内核

```
  "inbounds": [
    {
      "tag": "socks",
      "port": 10808,
      "listen": "127.0.0.1",
      "protocol": "mixed",
```

```
./xray run --config config.json
```

## clash

mihomo (Clash Meta) 自动选最快节点，主要做客户端代理内核

```
curl -L https://github.com/MetaCubeX/meta-rules-dat/releases/download/latest/geoip.metadb -o geoip.metadb
curl -L https://github.com/MetaCubeX/meta-rules-dat/releases/download/latest/geosite.dat -o geosite.dat
./mihomo-linux-amd64-v2-v1.19.24 -f config.yaml -d .
```

clash-verge-rev, ClashMetaForAndroid

订阅下载下来的配置文件，下次更新订阅会被覆盖。

```
port: 7890 http/https
socks-port: 7891 socks5
redir-port: 7892
mixed-port: 7893 http/https/socks5
dns:
  default-nameserver:
    - 180.184.1.1: 解析后面那些 https:// 格式的 DNS 服务器域名
  enhanced-mode: fake-ip
  fake-ip-range: 198.18.0.1/16
  nameserver-policy:
    'geosite:cn,private':
      - 119.29.29.29
      - 223.5.5.5
      - system
  nameserver:
    - 'https://223.6.6.6/dns-query'
    - 'https://doh.pub/dns-query'
  proxy-server-nameserver: 专门用来解析节点域名的 DNS
proxies: 节点
proxy-groups:
  - name: 
    type: select
    proxies:
  - name: '新加坡-自动选择'
    type: 'url-test'
    url: 'http://www.google-analytics.com/generate_204'
    interval: 300
    tolerance: 50
    proxies:
rules: 根据规则匹配流量，并指派给某个 proxy-groups
profile:
  store-selected: true: 记忆你选中的节点
```

select: 手动选择模式。在 UI 界面上你可以点击切换节点。
url-test: 自动选择延迟最低的节点（测速排队）。
fallback: 故障转移。如果第一个节点挂了，自动切换到下一个。
load-balance: 负载均衡
在 proxy-groups 下的 proxies 字段，既可以是具体的节点，也可以是其他的组

规则执行严格遵循 “自上而下，匹配即停”
具体域名 (DOMAIN)：最精确，放最上面。
域名后缀 (DOMAIN-SUFFIX)：中等范围。
地理位置/IP 段 (GEOIP/IP-CIDR)：范围大，可能触发 DNS 解析，放中下游。
最后匹配 (MATCH)：兜底规则，必须放最后。

```
浏览器发起 DNS 请求询问 google.com、baidu.com 的 IP。
Clash 立即返回一个 Fake-IP（如 198.18.0.x）。
浏览器发起 TCP 连接。
Clash 知道 198.18.0.x 实际上代表 xxx.com。
匹配规则：域名遇到域名直接进入代理组，直连触发dns；域名遇到 IP，触发dns解析。
baidu 符合 nameserver-policy，Clash 调用 119.29.29.29 查到真实 IP。
google 用 nameserver 解析。即使 Clash 已经在本地为了匹配 IP 规则而解析出了 IP 地址，它在建立代理隧道时，默认逻辑依然是优先传输域名。
Clash 将 google.com 这个原始域名打包进代理协议（如 VMess/Trojan），直接发往你的代理服务器。
https://ipleak.net
```

编辑规则：新增 rules，加入到运行配置里
编辑节点：新增节点
编辑代理组：新增代理组

扩展覆写配置：有相同的key，覆盖运行配置；没有，新增

```
dns:
  nameserver-policy:
    "+.sysu.edu.cn":
      - 119.29.29.29
      - 223.5.5.5
      - system
```

扩展脚本：用 JavaScript 脚本动态修改运行配置

## 入口

export变量和系统代理在本质上都是一种“建议”，需要应用自行发起代理。tun模式会在系统里创建一个虚拟网卡，修改路由表

网关共享：同一局域网下，A设置网关为B，B开启ip转发至tun

路由共享：B创建局域网供A接入，B在局域网的网卡转发tun

## 出口

直连：用户-代理

不同地方的不同运营商到不同代理的速度差异大

国内中转：用户-中转1-代理1
             -中转2-代理2

在国内先选好运营商

专线：中转到国外不走公网

中转和专线被拔线了

国外中转：中转在国外

# mpi

mpirun在远程节点上启动MPI进程

```
#!/bin/bash
/usr/local/openmpi-4.1.5/bin/orterun --allow-run-as-root -bind-to none -tag-output \
                                    -timestamp-output --hostfile /root/paddlejob/workspace/hostfile  \
                                    -x PATH -x LD_LIBRARY_PATH $*

-tag-output 给每行输出增加进程标识
-x PATH 把当前机器的 PATH 复制到所有 MPI 进程环境中
```

orterun 需要借助一个远程启动机制（通常是 SSH），先在远程机器启动 OpenMPI 的 daemon（orted），再由 daemon 启动真正的 MPI 进程，daemon 会随着整个 MPI Job 的结束而自动退出。每次 orterun 启动的作业都会被分配一个唯一的 Job ID，该作业中的所有 Rank 属于同一个 MPI_COMM_WORLD，不同 Job 之间天然隔离。

hostfile
```
10.54.107.140 slots=1
10.54.107.139 slots=1

slots用于指定某台主机（节点）上可以运行的最大 MPI 进程数量，OpenMPI 默认 slots=1
--oversubscribe会忽略限制，不指定-np会启动最大数量
```

-H优先级更高，可以绕过--hostfile
```
mpirun -H node1:1,node2:1 sh overlap.sh
```
mpicxx 是 C++ 编译器包装器，会自动加上mpi的头文件和库

```
mpicxx --showme
g++ -I/usr/local/openmpi-4.1.5/include -L/usr/local/openmpi-4.1.5/lib -Wl,-rpath -Wl,/usr/local/openmpi-4.1.5/lib -Wl,--enable-new-dtags -lmpi
```

数据传输基于TCP，Shared Memory，RDMA，...

