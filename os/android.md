每个应用默认有自己的 Linux 用户 ID (UID) 和独立进程。当应用启动时，系统会在该进程中启动 ART 虚拟机实例 来运行 Java/Kotlin 代码。

App 的私有目录属于这个 UID，其他 App（不同 UID）无法访问

Android 权限（相机、麦克风、网络）最终都会落实到 Linux UID

系统分身：Android 原生多用户（User）机制

```
主用户 (User 0)
  微信 appId=10001
  → Linux UID = 0 * 100000 + 10001 = 10001

分身用户 (User 10)
  微信 appId=10001
  → Linux UID = 10 * 100000 + 10001 = 1010001
```

应用分身：系统 / 厂商在“单用户内”人为制造的第二个 App（分配 新的 UID）

```
User 0:
  微信(主) → UID = 10001
  微信(分身) → UID = 10002
```

安卓手机要运行Google Play，必须依赖 GMS（Google Mobile Services）核心套件，打开系统自带谷歌服务开关
