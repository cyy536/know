﻿﻿﻿﻿[Online FlowChart & Diagrams Editor - Mermaid Live Editor](https://mermaid.live/edit#pako:eNqVkE1rg0AQhv_KMmdN_YhalxIwGughp_RQaMxhWccPoq6sK20a0t9ePxJC0ly6LOwM87zvzOwRuEgQKKSl-OQ5k4qsN3FN-hNs34XcF3VGokIiV0IedkTXSVYowpKkDxdkuX1TLBuYQCLbnYUXiufI96JTJI71_o6KYGKWF0Zii4q8roKIvKRFiYuHFBdV1T9DKdyuBWcl2WAj2mKcaoLDW8upZyvSyf3HnAa-Y5uuzcdCtN1gJRT-9Y2ubFk-paj4pAjvtm0VO3uthl9p87N-dQs0onm44j_lS9Agk0UCVMkONahQVmxI4TjoYlA5VhgD7cMEU9aVKoa4PvWyhtUfQlQXpRRdlgNNWdn2WdckTGFUsEyyK4J1gjIUXa2AOqMD0CN8AdUta-bNHc_ybcMyDW_u2hocgFqmOfNt1_ZNw_Yc3zxp8D32NGbu87PrzG3DdhzDswzr9Avdmb39)

# WD

本地文件系统的实际文件

文件状态通过 `git status` 显示为 Untracked（未跟踪）或 Modified（已修改）

# LR

键值数据库，键（Key）是对象内容的 SHA-1 哈希值，而值（Value）是对象本身（Blob、Tree、Commit 或 Tag）。

存储SA的内容

```
Blob 对象
存储文件内容（不包含文件名、权限等元数据）。

Tree 对象
100644 blob e69de29...    README.md
040000 tree 1a2b3c...    src/

Commit 对象
tree 1a2b3c...
parent 4d5e6f...
author Alice <alice@example.com> 1625097600 +0800
committer Bob <bob@example.com> 1625097600 +0800
Commit message
合并提交（Merge Commit）有多个父 Commit

Tag 对象指向一个 Commit 对象

分支是一个指向 Commit 的可变指针， HEAD 指针指向当前检出的分支

Fast-Forward Merge
A---B---C 你的分支（feature）
    /
   D 目标分支（develop）
git checkout develop
git merge feature
A---B---C（develop 直接指向 C）
        ｜
        D
        
True Merge
      A---B---C 你的分支（feature）
     /
D---E---F---G 目标分支（develop）
git checkout develop
git merge feature
      A---B---C
     /         \
D---E---F---G---H  （H 是新的合并提交）
基于 共同祖先（Base）、当前分支（Ours）、目标分支（Theirs）计算合并结果。若同一文件在 Base 后均被修改，则触发冲突。
合并提交有两个父提交

      A---B---C 你的分支（feature）
     /
D---E---F---G 目标分支（develop）
git checkout feature
git rebase develop
              A'--B'--C' 你的分支（feature）
             /
D---E---F---G 目标分支（develop）
每个提交重放时都可能冲突
git rebase -i HEAD~14
修改当前分支最近 14 个提交的历史（包括重排、合并、编辑提交信息等）
squash    s    将该提交合并到前一个提交

git cherry-pick：将指定 Commit 的变更 重新应用到当前分支，生成新 Commit（保留原提交信息）。
A---B---C---D  main
     \
      E---F---G  feature
git cherry-pick F
A---B---C---D---F'  main
     \
      E---F---G     feature
每个提交cherry-pick都可能冲突
```

# Stash

默认不会存储未跟踪文件！(从未add的文件，即新创建文件）

```
git stash push -m "新的描述名称"
git stash drop 1
git stash list
git stash apply stash@{0}
```

# .git

```
.git/index
.git/objects/
.git/refs/tags/v1.0    # 指向tag
.git/refs/heads/main   # 本地分支
.git/refs/remotes/origin/main  # 记录远程 main 分支的最新状态
.git/config
```

# script

```
git diff (--staged) (<commit1>)
diff A B: A=WD if --staged=null else SA, B=SA if <commit1>=null else <commit1>

git diff <commit1> <commit2>
git diff a1b2c3d -- README.md  # 比较 README.md 在 a1b2c3d 提交和当前版本的差异

git diff --name-only <commit-hash>^ <commit-hash>
git archive -o changes.zip 2102511777 $(git diff --name-only 2102511777^ 2102511777)
```

```
git log --oneline --graph --all

git fetch xing FLAGS_enable_inplace_master_grad
git fetch origin 4c26fdf5d08bce3e368ba3daef4da211fc73f4a7
git merge feature
git pull --force origin master(:develop)
拉取远程
git push <远程仓库名称> <本地分支名>:<远程分支名>
push本地

git branch <local-branch-name> <remote-name>/<remote-branch-name>
git brance new_branch
git checkout develop/label
git branch -a
git branch -d update_nccl_init
git branch -d -r origin/update_nccl_init
git push origin --delete sp_async_reduce_scatter
git branch -m sp_async_reduce_scatter sp_async_reduce_scatter_in_dygraph_auto_mode
```

```
git remote -v
git remote add upstream https://github.com/PaddlePaddle/Paddle.git
git remote set-url origin git@github.com:username/repo.git
git config --global user.name "chenyuyan"
git config --global user.email "2903319587@qq.com"
使用ssh或https取决于地址的写法，ssh需要密钥，https需要每次输入用户名和token

git clone --depth=1 https://gitee.com/mirrors/oneDNN.git third_party/onednn
只下载最新一次提交

git submodule sync
sync 命令会先读取 .gitmodules 里记录的子模块远程地址
[submodule "third_party/onednn"]
    path = third_party/onednn
    url = https://gitee.com/mirrors/oneDNN.git
    branch = main
然后把这个地址同步到本地运行时配置（.git/config 或 .git/modules/）

git submodule update --init --recursive third_party/onednn
首先读取本地运行时配置（.git/config 或 .git/modules/[子模块名]/config）；
如果本地运行时配置缺失 / 不一致，会 fallback 到 .gitmodules 的默认地址

git config --global http.proxy http://127.0.0.1:7890
git config --global https.proxy http://127.0.0.1:7890
```

```
__pycache__/
*.egg-info/
*.sqlite
*.so
*.swp
*.jsonl
*.log
*.pdf
*output_gptq.json
*output_smooth.json
*output_lut.json
ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json
*TwoWeeksJan2021.txt
*TwoWeeksJan2021.txt*
```

git config credential.helper 'cache --timeout=3600'