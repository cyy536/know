# 语法

[【数据库】SQL 3小时快速入门 #数据库教程 #SQL教程 #MySQL教程 #database#Python连接数据库_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1PT4y1e7UU/?spm_id_from=333.337.search-card.all.click&vd_source=7c406f161c56ca0055b900d0df22d9bc)

foreign key对应一个表的primary key，可以与其他表格建立关联，也可以是自己的表

```sql
create database `xx`;
show databases;
drop database `xx`;
use `xx`;
```

```
dtype
blob(binary large object)
```

```sql
create table `xx`(
    `xx` dtype auto_increment,
    `xx` dtype not null,
    `xx` dtype unique,
    `xx` dtype default xx,
    primary key(`xx`,`xx`),
    foreign key(`xx`) references `xx`(`xx`) on delete set null
    foreign key(`xx`) references `xx`(`xx`) on delete cascade
);
describe `xx`;
drop `xx`;
alter table `xx` add `xx` dtype;

alter table `xx` 
add foreign key(`xx`) 
references `xx`(`xx`) on delete set null;

alter table `xx` drop column `xx`;
```

```sql
insert into `xx` values(xx);
insert into `xx`(`xx`) values(xx);

update `xx`
set `xx`=xx
where `xx`=xx;

delete from `xx`
where `xx`=xx;

select `xx`,`xx` from `xx` 
where `xx`=xx or `xx` in(xx,xx)
order by `xx`,`xx` desc
limit 2;

select distinct `xx` from `xx`;
select count(`xx`) from `xx`;
select `xx` from `xx` where `xx` like '23%';

压缩行
SELECT department_id, job_title, AVG(salary) AS avg_salary
FROM employees
GROUP BY department_id, job_title;

扩展行
select `xx` as `xx` from `xx` 
union
select `xx` from `xx`;

连接（扩展列）
select `xx`.`xx` 
from `xx` (inner) join `xx`
on `xx`.xx`=`xx`.`xx`;
相同列相同值
for a in A
    for b in B
        if a.x==b.x
            (a,b-x)

left join:左边全部返回，如果左表的行在右表中没有匹配，那么这一行右表中对应数据用NULL代替。

递归（条件）
select `xx`
from `xx`
where `xx`=(
    select `xx` from `xx` where `xx`=xx
);
```

```
connection=connect(host,port,user,passwd)
connection.cursor.execute()
connection.cursor.fetchall()
connection.cursor.close()
connection.close()
```

可以对视图进行增删改查操作

# 基础

MySQL 的架构共分为两层：Server 层和存储引擎层

Server 层负责建立连接、缓存、分析和执行 SQL

MySQL 是基于 TCP 协议进行传输的。MySQL 的连接也跟 HTTP 一样，有短连接和长连接的概念

查询缓存是以 key-value 形式保存在内存中的，key 为 SQL 查询语句，value 为 SQL 语句查询的结果。只要一个表有更新操作，那么这个表的查询缓存就会被清空。MySQL 8.0 版本移除的是 server 层的查询缓存。

词法分析、语法分析

预处理阶段：检查 SQL 查询语句中的表或者字段是否存在。将 select * 中的 * 符号，扩展为表上的所有列

优化器主要负责将 SQL 查询语句的执行方案确定下来，减少查询成本

执行器从存储引擎读到完整记录后，接着判断记录是否符合查询条件，如果符合则发送给客户端，如果不符合则跳过该记录。一直重复上述过程，直到存储引擎把表中的所有记录读完

Server 层每从存储引擎读到一条记录就会发送给客户端，客户端是等查询语句查询完成后，才会显示出所有的记录

每创建一个 database（数据库） 都会在 /var/lib/mysql/ 目录里面创建一个以 database 为名的目录，然后保存表结构和表数据的文件都会存放在这个目录里。建立一张表都会生成一个.frm 文件。每一张表的数据都存放在一个独立的 .ibd 文件。

```
/var/lib/mysql/
    database
        A.frm
        A.ibd
        B.frm
        B.ibd
            索引段
            数据段
            回滚段    
```

记录是按照行来存储的，InnoDB 的数据是按「页」为单位来读写的，按照区（extent）为单位分配。

表空间是由各个段（segment）组成的，段是由多个区（extent）组成的。

如果有主键，默认会使用主键作为聚簇索引的索引键；

如果没有主键，就选择第一个不包含 NULL 值的唯一列作为聚簇索引的索引键

在上面两个都没有的情况下，InnoDB 将自动生成一个隐式自增 id 列（row_id）作为聚簇索引的索引键

其它索引都属于辅助索引，创建的主键索引和二级索引默认使用的是 B+Tree 索引。

b+: 叶子节点才存放数据，非叶子节点只存放索引，每一层父节点的索引值都会出现在下层子节点的索引值中，每一个叶子节点都有两个指针，分别指向下一个叶子节点和上一个叶子节点，适合 MySQL 中常见的基于范围的顺序查找

索引段：存放 B + 树的非叶子节点的区的集合；数据段：存放 B + 树的叶子节点的区的集合；

把读取一个非节点当作一次磁盘 I/O 操作，每个非节点都是一个页。

```
变长字段长度列表（存在变长类型）、NULL 值列表（存在null类型）、记录头信息、row_id、trx_id、roll_pointer、...
```

变长字段长度列表、NULL 值列表：逆序存放，提高 CPU Cache 的命中率。

记录头信息：标识此条数据是否被删除，下一条记录的位置。

事务id，表示这个数据是由哪个事务生成的

这条记录上一个版本的指针

一行记录除了 TEXT、BLOBs 类型的列，限制最大为 65535 字节

如果一个数据页存不了一条记录，InnoDB 存储引擎会自动将溢出的数据存放到「溢出页」中。在记录的真实数据处只会保存该列的一部分数据，用 20 字节存储指向溢出页的地址

# 索引

[索引常见面试题 | 小林coding](https://xiaolincoding.com/mysql/index/index_interview.html#什么是索引)

二级索引的 B+Tree 的叶子节点存放的是主键值，而不是实际数据。会先检二级索引中的 B+Tree 的索引值，找到对应的叶子节点，然后获取主键值，然后再通过主键索引中的 B+Tree 树查询到对应的叶子节点，然后获取整行数据。

建立在多列上的索引称为联合索引；联合索引查询的 B+Tree 是先按 product_no 进行排序，然后再 product_no 相同的情况再按 name 字段排序。（联合索引也可以是主键）

利用索引的前提是索引里的 key 是有序的，可以重复。范围查询可能不是所有字段都可以用到联合索引（a > 1 and b = 2）

索引下推可以在联合索引遍历过程中，对联合索引中包含的字段先做判断（b是否等于2），直接过滤掉不满足条件的记录，能够减少二级索引在查询时的回表操作（查找主键）

在符合 a >= 1 条件的二级索引记录的范围里，b 字段的值是无序的。但是对于符合 a = 1 的二级索引记录的范围里，b 字段的值是「有序」的。可以从符合 a = 1 and b = 2 条件的第一条记录开始扫描，而不需要从第一个 a 字段值为 1 的记录开始扫描。

覆盖索引是指 SQL 中 query 的所有字段，在索引 B+Tree 的叶子节点上都能找得到的那些索引。如果索引中存在这些数据，查询将不会再次检索主键索引

如果我们使用自增主键，插入一条新记录，都是追加操作，不需要重新移动数据。否则，插入到现有数据页中间的某个位置，这将不得不移动其它数据来满足新数据的插入

查询优化器发现某个值出现在表的数据行中的百分比（惯用的百分比界线是"30%"）很高的时候，它一般会忽略索引，进行全表扫描

当查询语句中包含 group by 操作，而且无法利用索引完成排序操作的时候， 这时不得不选择相应的排序算法进行，使用临时表保存中间结果（排好序才能分组？）

[从数据页的角度看 B+ 树 | 小林coding](https://xiaolincoding.com/mysql/index/page.html#innodb-是如何存储数据的)

```
文件头
用户记录
空闲空间
页目录
```

在 File Header 中有两个指针，分别指向上一个数据页和下一个数据页，连接起来的页相当于一个双向的链表

数据页中的记录按照「主键」顺序组成单向链表。数据页中有一个页目录，起到记录的索引作用

将所有的记录划分成几个组，页目录用来存储每组最后一条记录的地址偏移量，可以使用二分法快速定位要查询的记录在哪个组

MySQL 为了提高性能，会将表的索引装载到内存中

索引结构不会影响单表最大行数，2000W 也只是推荐值，超过了这个值可能会导致 B + 树层级更高，影响查询性能。

MySQL 在遇到字符串和数字比较的时候，会自动把字符串转为数字，然后再进行比较。

如果数据库表中的字段都是索引的话，即使查询过程中，没有遵循最左匹配原则，也是走全扫描二级索引树

server 层会循环向 InnoDB 读取一条记录，如果 count 函数指定的参数不为 NULL，那么就会将变量 count 加 1，直到符合查询的全部记录被读完。最后将 count 变量的值发送给客户端。

count(*) 执行过程跟 count(1) 执行过程基本一样的。如果有多个二级索引的时候，优化器会使用key_len 最小的二级索引进行扫描。只有当没有二级索引的时候，才会采用主键索引来进行统计。不会读取记录，开销小。

count(1) 相比 count(主键字段) 少一个步骤，就是不需要读取记录中的字段值。

不要使用 count(字段) 来统计记录个数，因为它的效率是最差的，会采用全表扫描的方式来统计

# 事务

[事务隔离级别是怎么实现的？ | 小林coding](https://xiaolincoding.com/mysql/transaction/mvcc.html#事务的隔离级别有哪些)

ACID

持久性是通过 redo log （重做日志）来保证的 ; 掉电故障恢复

原子性是通过 undo log（回滚日志） 来保证的；mvcc

隔离性是通过 MVCC（多版本并发控制） 或锁机制来保证的；

一致性则是通过持久性+原子性+隔离性来保证；指事务操作前和操作后，数据满足完整性约束，数据库保持一致性状态

事务可以嵌套执行

## 隔离

在同时处理多个事务的时候，就可能出现脏读（dirty read）、不可重复读（non-repeatable read）、幻读（phantom read）的问题。

脏读：读到其他事务未提交的数据；

不可重复读：一个事务内，前后读取的数据不一致 ;

幻读：前后读取的记录数量不一致。

读未提交（最新），读提交，可重复读，串行化（读写锁）

对于「读提交」和「可重复读」隔离级别的事务来说，它们是通过 Read View 来实现的。

```
creator_trx_id,m_idxs,min_trx_id,max_trx_id
```

m_ids ：指的是在创建 Read View 时，当前数据库中「活跃事务」的事务 id 列表

当一个事务对某条聚簇索引记录进行改动时，就会把该事务的事务 id 记录在 trx_id 隐藏列里；

每次对某条聚簇索引记录进行改动时，都会把旧版本的记录写入到 undo 日志中，然后这个隐藏列是个指针，指向每一个旧版本记录。读取第一个可见版本

如果记录的 trx_id 值小于 Read View 中的 min_trx_id 值，一定提交了，该版本可见

如果记录的 trx_id 值大于等于 Read View 中的 max_trx_id 值，不知道有没有提交，该版本不可见

之间：如果记录的 trx_id 在 m_ids 列表中，一定没提交，不可见；不在，一定提交了，可见

「读提交」隔离级别是在「每个读前」都会重新生成一个 Read View，「可重复读」隔离级别是「启动事务时」生成一个 Read View

MySQL InnoDB 引擎的默认隔离级别虽然是「可重复读」，但是它可以避免幻读现象

针对快照读（普通 select 语句），是通过 MVCC 方式解决了幻读；针对当前读（select ... for update 等语句），记录锁+间隙锁

update、insert、delete、当前读，这些语句执行前都会查询最新版本的数据

# 锁

## 全局锁

整个数据库就处于只读状态了，主要应用于做全库逻辑备份

不用锁：如果数据库的引擎支持的事务支持可重复读的隔离级别，那么在备份数据库之前先开启事务，备份期间业务依然可以对数据进行更新操作。

## 表级锁

### 表锁

读锁，写锁

表锁和行锁是满足读读共享、读写互斥、写写互斥的。

### 元数据锁

不需要显示的使用 MDL，会自动给这个表加上 MDL

对一张表进行 CRUD 操作时，加的是 MDL 读锁；对一张表做结构变更操作的时候，加的是 MDL 写锁；

MDL 是在事务提交后才会释放

申请 MDL 锁的操作会形成一个队列，队列中写锁获取优先级高于读锁，一旦出现 MDL 写锁等待，会阻塞后续该表的所有 CRUD 操作

读-写-读，后面的读会被阻塞

### 意向锁（标记）

当执行插入、更新、删除、当前读操作，需要先对表加上「意向独占锁」，然后对该记录加独占锁。普通的 select 是不会加行级锁的

在给记录加读（写）锁前，需要先在表上加意向读（写）锁。

意向锁只会和表锁冲突

意向锁的目的是为了快速判断表里是否有记录被加锁。如果没有「意向锁」，那么加「独占表锁」时，就需要遍历表里所有记录

### auto-inc锁

AUTO_INCREMENT：数据库会自动给主键赋值递增的值，这主要是通过 AUTO-INC 锁实现的。在插入数据时，会加一个表级别的 AUTO-INC 锁，等插入语句执行完成后，才会把 AUTO-INC 锁释放掉。其他事务的如果要向该表插入语句都会被阻塞

轻量级锁，给该字段赋值一个自增的值，就把这个轻量级锁释放了，而不需要等待整个插入语句执行完后才释放锁

## 行级锁

当事务执行 commit 后，事务过程中生成的锁才会被释放

### Record Lock

记录（读写）锁，也就是仅仅把一条记录锁上

记录锁无法防止插入，需要间隙锁。

### Gap Lock（标记）

间隙锁，锁定一个范围（3，5），但是不包含记录本身。两个事务可以同时持有包含共同间隙范围的间隙锁

### Next-Key Lock

Record Lock + Gap Lock 的组合，锁定一个范围（3，5]，并且锁定记录5。相同范围的互斥关系和record lock相同

对于这种范围为 (1006, +∞] 的 next-key lock，两个事务是可以同时持有的

### 插入意向锁

一个事务在插入一条记录的时候，需要判断插入位置是否已被其他事务加了间隙锁，如果有的话，插入操作就会发生阻塞，直到拥有间隙锁的那个事务提交为止，在此期间会生成一个插入意向锁等待。和间隙锁互斥

### 如何加

[MySQL 是怎么加锁的？ | 小林coding](https://xiaolincoding.com/mysql/lock/how_to_lock.html#唯一索引范围查询)

插入，update 和 delete，当前读操作都会加锁。在扫描过程中加锁

加锁的对象是索引或记录，加锁的基本单位是 next-key lock，在能使用记录锁或者间隙锁就能避免幻读现象的场景下， next-key lock 就会退化成记录锁或间隙锁。

二级索引加锁，查询到的记录的主键索引项上加「记录锁」

#### 唯一索引等值

存在，记录锁；不存在，间隙锁

#### 唯一索引范围

会对每一个扫描到的索引加 next-key 锁。

针对「大于等于」的范围查询，如果等值查询的记录是存在，该记录的索引中的 next-key 锁会退化成记录锁，在特殊记录（ supremum pseudo-record）的主键索引上，加了范围为 (20, +∞] 的 next-key 锁

针对「小于或者小于等于」的范围查询，当条件值的记录不在表中，终止范围查询的记录（第一个大于条件值）的索引的 next-key 锁会退化成间隙锁

针对「小于」的范围查询，当条件值的记录在表中，终止范围查询的记录时，该记录的索引的 next-key 锁会退化成间隙锁

#### 非唯一索引等值

对扫描到的二级索引记录加的是 next-key 锁，而对于第一个不符合条件的二级索引记录退化成间隙锁。防止等值插入。

插入间隙锁边界值有些可以有些不可以？先定位到该记录在 B+树 的位置，如果插入的位置的下一条记录的索引上有间隙锁，才会发生阻塞

#### 非唯一索引范围

唯一索引范围不退化

#### 非索引

全表加next-key，除了 select ... from语句，其他语句都会被锁住不能执行

## 死锁

事务 A 和事务 B 在执行完后 select ... for update 语句后都持有范围为(1006,+∞]的next-key 锁，而接下来的插入操作为了获取到插入意向锁，都在等待对方事务的间隙锁释放

设置事务等待锁的超时时间，开启主动死锁检测

当事务需要加锁的时，如果这个锁不可能发生冲突，InnoDB会跳过加锁环节，这种机制称为隐式锁

隐式锁就是在 Insert 过程中不加锁，只有在特殊情况下，才会将隐式锁转换为显示锁：

如果记录之间加有间隙锁，此时会生成一个插入意向锁

# 日志

## undolog

执行一条语句是否自动提交事务，是由 autocommit 参数决定的，默认是开启。

在事务没提交之前，MySQL 会先记录更新前的数据到 undo log 日志文件里面

不同的操作，需要记录的内容也是不同的

更新操作产生的 undo log 格式都有一个 roll_pointer 指针和一个 trx_id 事务id：通过 trx_id 可以知道该记录是被哪个事务修改的; 通过 roll_pointer 指针可以将这些 undo log 串成一个链表

undo log 和数据页的刷盘策略是一样的，都需要通过 redo log 保证持久化

```
innodb buffer pool：数据页、索引页、undo页、锁、。。。
```

## redolog

当有一条记录需要更新的时候，InnoDB 引擎就会先更新内存（同时标记为脏页），然后将本次对这个页的修改以 redo log 的形式记录下来。先写日志，然后在合适的时间再写到磁盘上。

对 XXX 表空间中的 YYY 数据页 ZZZ 偏移量的地方做了AAA 更新

写入 redo log 的方式使用了追加操作，写入磁盘的开销更小。redo log 也有自己的缓存—— redo log buffer。

「重做日志文件组」由有 2 个 redo log 文件组成。重做日志文件组是以循环写的方式工作的。脏页刷新到了磁盘中，那么 redo log 对应的记录也就没用了。redo log 文件满了，MySQL 会被阻塞。

## binlog

MySQL 在完成一条更新操作后，Server 层还会生成一条 binlog，事务提交的时候，会将该事物执行过程中产生的所有 binlog 统一写 入 binlog 文件。主要用于数据备份和主从复制

记录了所有数据库表结构变更和表数据修改的日志

binlog 有 3 种格式类型

STATEMENT：记录sql语句

ROW：记录最终结果

binlog 是追加写，写满一个文件，就创建一个新的文件继续写

如果不小心整个数据库的数据被删除了，只能使用 binlog 文件恢复。

在写数据时只写主库，在读数据时只读从库

提交事务后，复制binlog到所有从库

异步复制（默认模型）：MySQL 主库提交事务的线程并不会等待 binlog 同步到各从库，就返回客户端结果。

半同步复制：只要数据成功复制到任意一个从库上，主库的事务线程就可以返回给客户端。

事务执行过程中，先把日志写到 binlog cache，事务提交的时候，再把 binlog cache 写到 binlog 文件中。给每个线程分配了一片内存用于缓冲 binlog ，该内存叫 binlog cache。虽然每个线程有自己 binlog cache，但是最终都写到同一个 binlog 文件(page cache)

```
undo log,redo log
do,redo log
binlog
```

## 两阶段提交

因为 redo log 影响主库的数据，binlog 影响从库的数据。redo log 和 binlog 必须保持一致才能保证主从数据一致。

把单个事务的提交拆分成了 2 个阶段，分别是「准备（Prepare）阶段」和「提交（Commit）阶段」

将 redo log 的写入拆成了两个步骤：prepare 和 commit，中间再穿插写入binlog

对于处于 prepare 阶段的 redo log，即可以提交事务，也可以回滚事务，这取决于是否能在 binlog 中查找到与 redo log 相同的 XID

两阶段提交是以 binlog 写成功为事务提交成功的标识

缺点：磁盘 I/O 次数高，锁竞争激烈（保证多事务的情况下，两个日志的提交顺序一致）

当有多个事务提交的时候，会将多个 binlog 刷盘操作合并成一个，从而减少磁盘 I/O 的次数

# 内存

Innodb 存储引擎设计了一个缓冲池，Buffer Pool 是在 MySQL 启动的时候，向操作系统申请的一片连续的内存空间

InnoDB 为每一个缓存页都创建了一个控制块，控制块也是占有内存空间的，它是放在 Buffer Pool 的最前面

为了能够快速找到空闲的缓存页，使用Free 链表；了能快速知道哪些缓存页是脏的，于是就设计出 Flush 链表

脏页同时存在于 LRU 链表和 Flush 链表。

将 LRU 划分了 2 个区域：old 区域 和 young 区域。预读的页就只需要加入到 old 区域的头部

Buffer Pool 污染？扫描了大量的数据，导致大量热数据被淘汰了。进入到 young 区域条件增加了一个停留在 old 区域的时间判断。只有同时满足「被访问」与「在 old 区域停留时间超过 1 秒」两个条件，才会被插入到 young 区域头部。

# mongodb

把mysql的行换成字典，不同字典的key可以不相同，节省存储

