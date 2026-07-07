# introduction

https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2265/user-guide/docs/index.html

- single-threaded control of all GPUs

- multi-threaded, for example, using one thread per GPU

- multi-process, for example, MPI

Multiple calls to ncclSend() and ncclRecv() targeting different peers can be fused together with ncclGroupStart() andncclGroupEnd() to form more complex communication patterns such as one-to-all (scatter), all-to-one (gather), all-to-all. Aggregating communication operations to improve performance.

Point-to-point calls within a group will be blocking until that group of calls completes, but calls within a group can be seen as progressing independently.

阅读材料：

https://blog.csdn.net/kidgin7439/category_11998768.html（nccl-v2.7.8）

# 脚本

## build
在comm.mk开debug flag
```
make -j src.build NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"
```
## nccl_mpi_bcast.cpp

一个进程一个GPU

```
#include <nccl.h>
#include <mpi.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <assert.h>
#include <unistd.h> 

#define DATA_SIZE 1024*1024*100  // 测试数据长度

// 检查 CUDA 调用是否成功
void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        printf("CUDA Error (%s): %s\n", msg, cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

// 检查 NCCL 调用是否成功
void checkNccl(ncclResult_t res, const char* msg) {
    if (res != ncclSuccess) {
        printf("NCCL Error (%s): %s\n", msg, ncclGetErrorString(res));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

int main(int argc, char* argv[]) {
    sleep(20);

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    checkCuda(cudaSetDevice(rank%8), "cudaSetDevice");
    int current_gpu;
    checkCuda(cudaGetDevice(&current_gpu), "cudaGetDevice");
    printf("Rank %d is using GPU %d\n", rank, current_gpu);

    ncclComm_t comm;
    cudaStream_t stream;
    checkCuda(cudaStreamCreate(&stream), "cudaStreamCreate");

    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "cudaEventCreate");
    checkCuda(cudaEventCreate(&stop), "cudaEventCreate");
    float time_ms;
    int co;

    MPI_Barrier(MPI_COMM_WORLD);
    ncclUniqueId id;
    checkCuda(cudaEventRecord(start), "cudaEventRecord");
    if (rank == 0) checkNccl(ncclGetUniqueId(&id), "ncclGetUniqueId");
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    checkNccl(ncclCommInitRank(&comm, size, id, rank), "ncclCommInitRank");
    checkCuda(cudaEventRecord(stop), "cudaEventRecord");
    checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize");
    checkCuda(cudaEventElapsedTime(&time_ms, start, stop), "cudaEventElapsedTime");
    printf("Rank %d: ncclCommInitRank time = %.3f ms\n", rank, time_ms);

    float *d_data;
    checkCuda(cudaMalloc(&d_data, DATA_SIZE * sizeof(float)), "cudaMalloc");
    co=2;
    while(co--){
        MPI_Barrier(MPI_COMM_WORLD);
        checkCuda(cudaEventRecord(start, stream), "cudaEventRecord");
        checkNccl(ncclBroadcast(d_data, d_data, DATA_SIZE, ncclFloat, 0, comm, stream), "ncclBroadcast");
        checkCuda(cudaEventRecord(stop, stream), "cudaEventRecord");
        if(co==0) {
            checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize");
            checkCuda(cudaEventElapsedTime(&time_ms, start, stop), "cudaEventElapsedTime");
            printf("Rank %d: ncclBroadcast time = %.3f ms\n", rank, time_ms);
        }
    }

    float *d_reduce_result;
    if (rank == 0) {
        checkCuda(cudaMalloc(&d_reduce_result, DATA_SIZE * sizeof(float)), "cudaMalloc");
        checkCuda(cudaMemset(d_reduce_result, 0, DATA_SIZE * sizeof(float)), "cudaMemset");
    } else {
        d_reduce_result = NULL; 
    }
    co=2;
    while(co--){
    MPI_Barrier(MPI_COMM_WORLD);
    checkCuda(cudaEventRecord(start, stream), "cudaEventRecord");
    checkNccl(ncclReduce(d_data, d_reduce_result, DATA_SIZE, ncclFloat, ncclSum, 0, comm, stream), "ncclReduce");
    checkCuda(cudaEventRecord(stop, stream), "cudaEventRecord");
    checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize");
    checkCuda(cudaEventElapsedTime(&time_ms, start, stop), "cudaEventElapsedTime");
    if(co==0) printf("Rank %d: ncclReduce time = %.3f ms\n", rank, time_ms);
    }

    float *d_allgather_result;
    checkCuda(cudaMalloc(&d_allgather_result, DATA_SIZE * sizeof(float)), "cudaMalloc");
    checkCuda(cudaMemset(d_allgather_result, 0, DATA_SIZE * sizeof(float)), "cudaMemset");
    co=2;
    while(co--){
    MPI_Barrier(MPI_COMM_WORLD);
    checkCuda(cudaEventRecord(start, stream), "cudaEventRecord");
    checkNccl(ncclAllGather(d_data, d_allgather_result, DATA_SIZE / size, ncclFloat, comm, stream), "ncclAllGather");
    checkCuda(cudaEventRecord(stop, stream), "cudaEventRecord");
    checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize");
    checkCuda(cudaEventElapsedTime(&time_ms, start, stop), "cudaEventElapsedTime");
    if(co==0) printf("Rank %d: ncclAllGather time = %.3f ms\n", rank, time_ms);
    }

    float *d_reducescatter_result;
    checkCuda(cudaMalloc(&d_reducescatter_result, DATA_SIZE / size * sizeof(float)), "cudaMalloc");
    checkCuda(cudaMemset(d_reducescatter_result, 0, DATA_SIZE / size * sizeof(float)), "cudaMemset");
    co=2;
    while(co--){
    MPI_Barrier(MPI_COMM_WORLD);
    checkCuda(cudaEventRecord(start, stream), "cudaEventRecord");
    checkNccl(ncclReduceScatter(d_data, d_reducescatter_result, DATA_SIZE / size, ncclFloat, ncclSum, comm, stream), "ncclReduceScatter");
    checkCuda(cudaEventRecord(stop, stream), "cudaEventRecord");
    checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize");
    checkCuda(cudaEventElapsedTime(&time_ms, start, stop), "cudaEventElapsedTime");
    if(co==0) printf("Rank %d: ncclReduceScatter time = %.3f ms\n", rank, time_ms);
    }

    float *d_allreduce_result;
    checkCuda(cudaMalloc(&d_allreduce_result, DATA_SIZE * sizeof(float)), "cudaMalloc");
    checkCuda(cudaMemcpy(d_allreduce_result, d_data, DATA_SIZE * sizeof(float), cudaMemcpyDeviceToDevice), "cudaMemcpy");
    co=2;
    while(co--){
    MPI_Barrier(MPI_COMM_WORLD);
    checkCuda(cudaEventRecord(start, stream), "cudaEventRecord");
    checkNccl(ncclAllReduce(d_allreduce_result, d_allreduce_result, DATA_SIZE, ncclFloat, ncclSum, comm, stream), "ncclAllReduce");
    checkCuda(cudaEventRecord(stop, stream), "cudaEventRecord");
    checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize");
    checkCuda(cudaEventElapsedTime(&time_ms, start, stop), "cudaEventElapsedTime");
    if(co==0) printf("Rank %d: ncclAllReduce time = %.3f ms\n", rank, time_ms);
    }

    checkCuda(cudaFree(d_data), "cudaFree");
    if (rank == 0) checkCuda(cudaFree(d_reduce_result), "cudaFree");
    checkCuda(cudaFree(d_allgather_result), "cudaFree");
    checkCuda(cudaFree(d_reducescatter_result), "cudaFree");
    checkCuda(cudaFree(d_allreduce_result), "cudaFree");
    ncclCommDestroy(comm);
    MPI_Finalize();
    return 0;
}
```

编译命令

```
mpicxx -g -O0 \
    -I/usr/local/cuda/include \
    -I/root/paddlejob/workspace/env_run/output/chenyuyan/Nccl/nccl/build/include \
    nccl_mpi_bcast.cpp -o nccl_mpi_bcast \
    -L/usr/local/cuda/lib64 \
    -L/root/paddlejob/workspace/env_run/output/chenyuyan/Nccl/nccl/build/lib \
    -lcudart -lnccl 
```

启动命令

```
export NCCL_RUNTIME_CONNECT=0 
root_path=/root/paddlejob/workspace/env_run/output/chenyuyan/Nccl
task_name=nccl_test
# export NCCL_TOPO_DUMP_FILE=$root_path/topo.xml
export NCCL_DEBUG_FILE=$root_path/nccl_log/$task_name/%h.%p.log
if [ -d $root_path/nccl_log/$task_name/ ]; then
    rm -rf $root_path/nccl_log/$task_name/
fi
mkdir -p $root_path/nccl_log/$task_name/
export NCCL_DEBUG_SUBSYS=INIT,COLL,TUNING,ALLOC,ENV,GRAPH
export LD_LIBRARY_PATH=$root_path/nccl/build/lib:$LD_LIBRARY_PATH 
# mpiexec --allow-run-as-root -np 8 ./nccl_mpi_bcast
./nccl_mpi_bcast
```

mpirun --oversubscribe -np 16 sh bcast.sh

## NCCl通信组特征分析工具

```
import re
import os
import argparse
import subprocess
import pandas as pd
import concurrent.futures
from tqdm import tqdm
from collections import defaultdict

######################################## re pattern #################################################

# 支持3种初始化格式
# NCCL INFO ncclCommInitRank comm 0x12026320 rank 0 nranks 2 cudaDev 0 nvmlDev 0 busId 19000 commId 0xc266e719e89eb1ab - Init START
# NCCL INFO ncclCommInitRankConfigMemOpt comm 0x55a7a339db50 rank 0 nranks 8 cudaDev 0 nvmlDev 0 busId 4b000 commId 0x62a8e54c3929a7df - Init START    
# NCCL INFO ncclCommInitRankConfig
initRank_start_pattern = re.compile(
    r'.*?ncclCommInitRank(?:Config)?(?:MemOpt)? comm ([0-9a-fx]+) rank 0 nranks (\d+) cudaDev (\d+) .* commId ([0-9a-fx]+) - Init START',
    re.IGNORECASE
)

# NCCL INFO comm 0x55a7a339db50 rank 0 nRanks 8 nNodes 1 localRanks 8 localRank 0 MNNVL 0
nNodes_pattern = re.compile(
    r'.*?NCCL INFO comm ([0-9a-fx]+) rank 0 nRanks (\d+) nNodes (\d+) localRanks (\d)+ localRank \d+ MNNVL \d+',
    re.IGNORECASE
)

# NCCL INFO comm 0x555e0a3036b0 rank 0 nranks 8 cudaDev 0 setBuffSizesMemOpt commName paddleComm set LL BuffSize to 524288
commName_pattern = re.compile(
    r'NCCL INFO comm ([0-9a-fx]+) rank 0 nranks \d+ cudaDev \d+ setBuffSizesMemOpt commName (\w+) set LL BuffSize to (\d+)',
    re.IGNORECASE
)

# NCCL INFO comm 0x55a7a339db50, 16 coll channels, 16 collnet channels, 0 nvls channels, 16 p2p channels, 2 p2p channels per peer
nChannels_pattern = re.compile(
    r'.*?NCCL INFO (?:comm ([0-9a-fx]+), )?(\d+) coll channels, (\d+) collnet channels, (\d+) nvls channels, (\d+) p2p channels, (\d+) p2p channels per peer',
    re.IGNORECASE
)

# NCCL INFO Init timings - ncclCommInitRankMemOpt: rank 0 nranks 8 total 0.48 (kernels 0.13, alloc 0.19, bootstrap 0.00, allgathers 0.01, topo 0.11, graphs 0.00, connections 0.04, rest 0.00)


# INFO(NCCL_TUNING, "%s: %ld Bytes -> Algo %s proto %s channel{Lo..Hi}={%d..%d} comm %p"
tuning_pattern = re.compile(
   r'.*?NCCL INFO (AllReduce|Broadcast|Reduce|AllGather|ReduceScatter|Send|Recv): (\d+) Bytes -> Algo (\w+) proto (\w+) channel\{Lo\.\.Hi\}=\{(\d+)\.\.(\d+)\} comm ([0-9a-fx]+)(?: progressState ([0-9a-fx]+))?'
)

# NCCL INFO progressState 0x55aff2e011c8 taskID 8 recv at 71820.535022 ms
rdmaStart_pattern = re.compile(
   r'.*?NCCL INFO progressState ([0-9a-fx]+) taskID (\d+) (?:recv|send) at ([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?) ms'
)

# NCCL INFO progressState 0x55aff2e011c8 taskID 8 end at 71837.969907 ms
rdmaEnd_pattern = re.compile(
   r'.*?NCCL INFO progressState ([0-9a-fx]+) taskID (\d+) end at ([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?) ms'
)

def new_comm_group():
    return {
        'id': 0,
        'name': '',
        'commId': '',
        'rank0_comm': '',
        'nranks': '',
        'nNodes': 0,
        'cudaDev': '',
        'localRanks': 0,
        'nchannels': 0,
        'initTime': 0,
        # colls[coll][msgsize] = [ {algo, proto, nc_used, count}, ... ]
        'colls': defaultdict(lambda: defaultdict(list))
    }

def parse_nccl_log(log_content,latency):
    status = defaultdict(new_comm_group)
    commid_to_rank0comm = {}  # commId -> rank0_comm
    progressState_to_rank0comm = {}

    lines = log_content.strip().split("\n")
    for line in tqdm(lines, desc="log parsing", ncols=80):
        line = line.strip()

        # 1.匹配通信组初始化行
        init_match = initRank_start_pattern.search(line)
        if init_match:
            rank0_comm = init_match.group(1)
            nranks = int(init_match.group(2))
            cudaDev = init_match.group(3)
            comm_id = init_match.group(4)
            commid_to_rank0comm[comm_id] = rank0_comm

            group = status[comm_id]
            group['id'] = len(status)
            group['name'] = '-'
            group['commId'] = comm_id
            group['rank0_comm'] = rank0_comm
            group['nranks'] = nranks
            group['cudaDev'] = cudaDev #此处仅为rank 0的cudaDev,后续需要补全，这个通信组的所有rank的cudaDev
            continue

        # 2.匹配 nNodes 行
        nNodes_match = nNodes_pattern.search(line)
        if nNodes_match:
            rank0_comm = nNodes_match.group(1)
            nranks = int(nNodes_match.group(2))
            nNodes = int(nNodes_match.group(3))
            localRanks = int(nNodes_match.group(4))
            # 查找 comm_id， 找不到comm_id，跳过
            comm_id = next((cid for cid, r0c in commid_to_rank0comm.items() if r0c == rank0_comm), None)
            if comm_id is None:
                comm_id = list(status.keys())[0]

            group = status[comm_id]
            group['nNodes'] = nNodes
            group['localRanks'] = localRanks
            # print(f"[nNodes_match] id={group['id']},rank0_comm={rank0_comm}, nNodes={nNodes}, localRanks={localRanks}, comm_id={comm_id}")
            continue

        # 3.匹配 commName 行
        commName_match = commName_pattern.search(line)
        if commName_match:
            rank0_comm = commName_match.group(1)
            commNameStr = commName_match.group(2)
            comm_id = next((cid for cid, r0c in commid_to_rank0comm.items() if r0c == rank0_comm), None)
            if comm_id is None:
                continue

            group = status[comm_id]
            group['name'] = commNameStr
            continue

        # 4.匹配 nChannels 行
        nc_match= nChannels_pattern.search(line)
        if(nc_match):
            rank0_comm = nc_match.group(1)
            nc = int(nc_match.group(2))
            comm_id = next((cid for cid, r0c in commid_to_rank0comm.items() if r0c == rank0_comm), None)
            if comm_id is None:
                comm_id = list(commid_to_rank0comm.keys())[0]

            group = status[comm_id]
            group['nchannels'] = nc
            # print(f"[nChannels_match] id={group['id']},rank0_comm={rank0_comm}, nc={nc}, comm_id={comm_id}")
            continue

        # 5.匹配tuning行，获取算法/协议/通道等信息
        tuning_match = tuning_pattern.search(line)
        if tuning_match:
            coll=tuning_match.group(1)
            msgsize = int(tuning_match.group(2))
            algo = tuning_match.group(3)
            proto = tuning_match.group(4)
            channel_lo = int(tuning_match.group(5))
            channel_hi = int(tuning_match.group(6))
            rank0_comm = tuning_match.group(7)

            nc_used = channel_hi - channel_lo + 1
            if algo == 'Unknown':
                algo = '-'

            comm_id = next((cid for cid, r0c in commid_to_rank0comm.items() if r0c == rank0_comm), None)
            if comm_id is None:
                comm_id = list(commid_to_rank0comm.keys())[0]

            group = status[comm_id]

            # 查找是否已有该msgsize+ algo+proto+nc_used的统计项
            entry_list = group['colls'][coll][msgsize]
            entry = next((e for e in entry_list if e['algo'] == algo and e['proto'] == proto and e['nc_used'] == nc_used), None)
            if entry and latency is False:
                entry['count'] += 1
            else:
                entry_list.append({'algo': algo, 'proto': proto, 'nc_used': nc_used, 'count': 1,})

    return status

def print_status(status, base_name=None, aggregate=False, latency=False):
    col_widths = {
        "no": 4, "name":14, "commId": 20, "rank0 comm": 20, "nranks": 8, "nNodes": 8, "Dev": 4,"localRanks": 10,
        "nchannels": 10, "coll": 14, "msgsize/B": 14, "min_size": 12, "max_size": 12,
        "algo": 6, "proto": 8, "nc_used": 8, "count": 6
    }
    def pad(val, width):
        return str(val).ljust(width)[:width]
    formatters = {col: (lambda w: (lambda x: pad(x, w)))(width) for col, width in col_widths.items()}

    # 生成表格数据
    rows = []
    for comm_id, group in status.items():
        first_group = True
        for coll, msgsize_dict in group['colls'].items():
            if aggregate:
                # 聚合: (algo, proto, nc_used) -> {'count':..., 'min':..., 'max':...}
                agg = {}
                for msgsize, entry_list in msgsize_dict.items():
                    for entry in entry_list:
                        key = (entry['algo'], entry['proto'], entry['nc_used'])
                        if key not in agg:
                            agg[key] = {
                                'count': 0,
                                'min_size': msgsize,
                                'max_size': msgsize
                            }
                        agg[key]['count'] += entry['count']
                        agg[key]['min_size'] = min(agg[key]['min_size'], msgsize)
                        agg[key]['max_size'] = max(agg[key]['max_size'], msgsize)
                for (algo, proto, nc_used), info in agg.items():
                    rows.append({
                        "no": group['id'] if first_group else "",
                        "name": group['name'] if first_group else "",
                        "commId": comm_id if first_group else "",
                        "rank0 comm": group['rank0_comm'] if first_group else "",
                        "nranks": group['nranks'] if first_group else "",
                        "nNodes": group['nNodes'] if first_group else "",
                        "Dev": group['cudaDev'] if first_group else "",
                        "localRanks": group['localRanks'] if first_group else "",
                        "nchannels": group['nchannels'] if first_group else "",
                        "coll": coll,
                        "min_size": info['min_size'],
                        "max_size": info['max_size'],
                        "algo": algo,
                        "proto": proto,
                        "nc_used": nc_used,
                        "count": info['count']
                    })
                    first_group = False
            else:
                for msgsize in sorted(msgsize_dict.keys()):# 对每个coll 下的 msgsize 进行升序排序
                    entry_list = msgsize_dict[msgsize]
                    for entry in entry_list:
                        rows.append({
                            "no": group['id'] if first_group else "",# 只输出1次通信组基础信息
                            "name": group['name'] if first_group else "",
                            "commId": comm_id if first_group else "",
                            "rank0 comm": group['rank0_comm'] if first_group else "",
                            "nranks": group['nranks'] if first_group else "",
                            "nNodes": group['nNodes'] if first_group else "",
                            "Dev": group['cudaDev'] if first_group else "",
                            "localRanks": group['localRanks'] if first_group else "",
                            "nchannels": group['nchannels'] if first_group else "",
                            "coll": coll,
                            "msgsize/B": msgsize,
                            "algo": entry['algo'],
                            "proto": entry['proto'],
                            "nc_used": entry['nc_used'],
                            "count": entry['count']
                        })
                        first_group = False
    df = pd.DataFrame(rows)

    header = " ".join([pad(col, col_widths[col]) for col in df.columns])
    lines = []
    lines.append(f"\nNCCL日志统计报告—logfile {base_name}\n")
    lines.append("=" * len(header))
    lines.append(header)
    lines.append("-" * len(header))
    # print(df.to_string(index=False, header=False, justify='left', formatters=formatters))

    last_comm_id = None
    for idx, row in df.iterrows():
        # 仅在通信组切换时打印分隔线（跳过第一行）
        if last_comm_id is not None and row['commId'] != '' and row['commId'] != last_comm_id:
            lines.append("- " * (len(header)//2))
        # 打印当前行
        lines.append(" ".join([pad(str(row[col]), col_widths[col]) for col in df.columns]))
        if row['commId'] != '':
            last_comm_id = row['commId']

    lines.append("=" * len(header))

    # if base_name:
    #     excel_file = base_name + ".xlsx"
    #     df.to_excel(excel_file, index=False)
    #     wb = load_workbook(excel_file)
    #     ws = wb.active
    #     for idx, col in enumerate(df.columns, 1):
    #         width = col_widths.get(col, 12)
    #         ws.column_dimensions[chr(64+idx)].width = width
    #     wb.save(excel_file)
    #     print(f"\n已导出到Excel: {excel_file}\n")

    return lines

def process_log_file(log_path, agg, latency):
    # 检查是否是rank 0
    result = subprocess.run(
        ["grep", "rank 0 nrank", log_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        encoding="utf-8"
    )
    if result.returncode != 0:
        print(f"\n[跳过] rank 0 not found - {log_path}\n")
        return

    cudaDev = None
    for line in result.stdout.splitlines():
        match = re.search(r'cudaDev (\d+)', line)
        if match:
            cudaDev = match.group(1)
            break

    if cudaDev is not None:
        print(f"\nrank 0 hit, cudaDev: {cudaDev} - {log_path}\n")
    else:
        print
        print(f"\nrank 0's cudaDev was not found - {log_path}\n")
        return 

    base_name = os.path.splitext(os.path.basename(log_path))[0]
    # 拼接cudaDev到文件名
    if cudaDev is not None and base_name:
        base_name = f"{base_name}-cudaDev-{cudaDev}"

    with open(log_path, 'r') as f:
        status = parse_nccl_log(f.read(),latency)
        lines=print_status(status, base_name, aggregate=agg, latency=latency)
        return base_name, lines

def main():
    parser = argparse.ArgumentParser(description="NCCL通信组特征分析工具")
    parser.add_argument("-i", "--input", help="logfile path")
    parser.add_argument("-d", "--dir", help="log目录,处理目录下所有.log文件")
    parser.add_argument("-o", "--output", nargs='?', const='', help="可选,若不指定则为result.txt")
    parser.add_argument("-a", "--agg", action="store_true", help="是否聚合msgsize区间（algo/proto/nc_used相同则合并）")
    parser.add_argument("-l", "--latency", action="store_true", help="每个task的时延")
    args = parser.parse_args()

    output_file = args.output
    if output_file == '':
        output_file = "result.txt"

    if args.latency:
        args.agg = False

    if args.dir: 
        log_tasks = []
        for fname in os.listdir(args.dir):
            if fname.endswith('.log'):
                log_path = os.path.join(args.dir, fname)
                log_tasks.append((log_path, args.agg, args.latency))
        # 多进程并行处理
        results = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_log_file, log_path, agg, latency) for log_path, agg, latency in log_tasks]
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        results.append(result)  # (base_name, lines)
                except Exception as exc:
                    print(f"日志处理出错: {exc}")

        # 按 base_name 排序
        results.sort(key=lambda x: x[0])

        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                for base_name, lines in results:
                    for line in lines:
                        # print(line)
                        f.write(line + "\n")
            print(f"\n所有日志已汇总到 {output_file}\n")
        else:
            for base_name, lines in results:
                for line in lines:
                    print(line)
    elif args.input:
        # 单文件处理
        base_name,lines= process_log_file(args.input, args.agg, args.latency)

        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                for line in lines:
                    f.write(line + "\n")
            print(f"\n已导出到: {output_file}\n")
        else:
            for line in lines:
                print(line)
    else:
        print("请指定 -i <logfile> 或 -d <logdir>")
        exit(1)


if __name__ == "__main__":
    main()
```

python get_comm_info.py -d ./nccl_log/nccl_test/ -a

# nccl-tests

```
make MPI=1 MPI_HOME=/usr/local/openmpi-4.1.5/ CUDA_HOME=/usr/local/cuda/ NCCL_HOME=/root/paddlejob/workspace/env_run/output/chenyuyan/Nccl/nccl/build/ LD_LIBRARY_PATH=/root/paddlejob/workspace/env_run/output/chenyuyan/Nccl/nccl/build/lib:/usr/local/openmpi-4.1.5/lib:$LD_LIBRARY_PATH 

./build/all_reduce_perf -b 8 -e 128M -f 2 -g 8
```

common.cu

```
int main(int argc, char* argv[]) {
    #ifdef MPI_SUPPORT
      MPI_Init(&argc, &argv);
    #endif
      TESTCHECK(run());   
        ncclTestEngine.getBuffSize(&sendBytes, &recvBytes, (size_t)maxBytes, (size_t)ncclProcs*nGpus*nThreads);
        ncclTestEngine.runTest

struct testColl allReduceTest = {
  "AllReduce",
  AllReduceGetCollByteCount,
  AllReduceInitData,
  AllReduceGetBw,
  AllReduceRunColl
};

AllReduceRunTest
    args->collTest = &allReduceTest;
    TimeTest(args,
        setupArgs(size, type, args);
            args->collTest->getCollByteCount
        BenchTime(args, type, op, root, 0));
            startColl
                args->collTest->runColl(
```

https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md

算法带宽=单卡数据量/统计时延

理论时延=数据移动总量/bus带宽总量

令统计时延==理论时延，得到算法带宽和bus带宽的关系

bus带宽是算法带宽去除rank数量的影响，应接近硬件的瓶颈，不是所有配置、数据都能达到

# ncclGetUniqueId

ncclInit

1. 读取配置，设置环境变量

2. bootstrapNetInit设置bootstrapNet网卡一张
   
   1. NCCL_COMM_ID
   
   2. NCCL_SOCKET_IFNAME 
   
   3. 先查找 ib 网卡，再查找以太网卡。

ncclUniqueId由随机数和bootstrapNetIfAddr组成

rank0启动bootstrapRoot线程，使用bootstrapNetIfAddr监听网络

```
static char bootstrapNetIfName[MAX_IF_NAME_SIZE+1];//xgbe0//eth0
static union ncclSocketAddress bootstrapNetIfAddr;//xx.xx.xx.xx

typedef struct { char internal[NCCL_UNIQUE_ID_BYTES]; } ncclUniqueId;

struct ncclBootstrapHandle {
  uint64_t magic;//随机数
  union ncclSocketAddress addr;//bootstrapNetIfAddr
};

ncclGetUniqueId
    ncclInit
    //设置环境变量，设置bootstrapNetIfName，设置bootstrapNetIfAddr
        initEnv();
        //从配置文件读取环境变量并设置到当前进程
            getenv("NCCL_CONF_FILE")
            setEnvFile(confFilePath);
                setenv(envVar, envValue, 0);
        initGdrCopy();//no
        bootstrapNetInit()
        //给bootstrapNet找网卡名和IP
            const char* env = ncclGetEnv("NCCL_COMM_ID");
            if (env) {
                ncclSocketGetAddrFromString
                ncclFindInterfaceMatchSubnet
            } else{
                int nIfs = ncclFindInterfaces(bootstrapNetIfName, &bootstrapNetIfAddr, MAX_IF_NAME_SIZE, 1);
                    const char* env = ncclGetEnv("NCCL_SOCKET_IFNAME");//xgbe0//eth0
                    if (env && strlen(env) > 1) {
                        nIfs = findInterfaces(env, ifNames, ifAddrs, sock_family, ifNameMaxSize, maxIfs);
                    } else {
                        nIfs = findInterfaces("ib", ifNames, ifAddrs, sock_family, ifNameMaxSize, maxIfs);
                        //满足ib开头的网卡
                            int nUserIfs = parseStringList(prefixList, userIfs, MAX_IFS);
                            //解析"eth0:80,eth1,lo:443,wlan0:8080"
                            //[ib:-1,]
                            getifaddrs(&interfaces);
                            //从操作系统获取网卡
                            //Create a linked list of `struct ifaddrs' structures, one for each network interface on the host machine.
                            matchIfList(interface->ifa_name, -1, userIfs, nUserIfs, searchExact)
                            //看机器和用户是否匹配
                            strncpy(names+found*maxIfNameSize, interface->ifa_name, maxIfNameSize);
                            // Store the interface name
                            memcpy(addrs+found, interface->ifa_addr, salen);
                            // Store the IP address
                        if (nIfs == 0) {
                          const char* commId = ncclGetEnv("NCCL_COMM_ID");
                          //Try to find interface that is in the same subnet as the IP in comm id
                          ncclSocketGetAddrFromString
                          ncclFindInterfaceMatchSubnet
                        }
                        if (nIfs == 0) nIfs = findInterfaces("^docker,lo", ifNames, ifAddrs, sock_family, ifNameMaxSize, maxIfs);
                        // Finally look for docker, then lo.
                        if (nIfs == 0) nIfs = findInterfaces("docker", ifNames, ifAddrs, sock_family, ifNameMaxSize, maxIfs);
                        if (nIfs == 0) nIfs = findInterfaces("lo", ifNames, ifAddrs, sock_family, ifNameMaxSize, maxIfs);
                    }
            }
    bootstrapGetUniqueId
        const char* env = ncclGetEnv("NCCL_COMM_ID");
        if (env) {
            ncclSocketGetAddrFromString(&handle->addr, env)
            handle->magic = NCCL_MAGIC;
            // 有NCCL_COMM_ID在后面bootstrapCreateRoot
        } else {
            NCCLCHECK(getRandomData(&handle->magic, sizeof(handle->magic)));
            memcpy(&handle->addr, &bootstrapNetIfAddr, sizeof(union ncclSocketAddress));
            bootstrapCreateRoot(handle, false));
                ncclSocketInit(listenSock, &handle->addr
                //fd=socket(addr)
                    sock->fd = -1;
                    sock->acceptFd = -1;
                    if (addr) {
                        memcpy(&sock->addr, addr,
                        socketResetFd(sock)
                            fd = socket(sock->addr.sa.sa_family, SOCK_STREAM, 0)
                            dup2(fd, sock->fd)
                     } else {
                        memset(&sock->addr, 0, sizeof(union ncclSocketAddress));
                     }
                ncclSocketListen(listenSock)
                //获取port，listen
                    bind(sock->fd, &sock->addr.sa, sock->salen)
                    // addr port should be 0 (Any port)
                    //系统会分配一个临时可用端口。
                    getsockname(sock->fd, &sock->addr.sa, &size)
                    //Get the assigned Port
                    listen(sock->fd, 16384)
                ncclSocketGetAddr(listenSock, &handle->addr)
                    memcpy(addr, &sock->addr
                    //设置port，原来是0
                args->listenSock = listenSock;
                pthread_create(&thread, NULL, bootstrapRoot, (void*)args)
                    listenSock = args->listenSock;
                    ncclSocketInit(&sock)
                    ncclSocketAccept(&sock, listenSock)
                    //创建socket用于连接
                    socketRecv(&sock, &info, sizeof(info))
                    ncclSocketClose(&sock)
                    ...
        }
```

# ncclNetInit

ncclNetInit选择一种网络作为ncclNet，设置ncclNet网卡s

1. IB: 一个用于元数据交换，其余传输数据

2. Socket...

```
struct ncclComm {
  ncclNet_t* ncclNet;

ncclCommInitRank
    ncclCudaLibraryInit
        cudaGetDevice(&cudaDev)//在main函数里面set
        cudaPfnFuncLoader()
            LOAD_SYM(cuGetErrorString, 0);
            ...
        ncclCuMemSupported = ncclIsCuMemSupported();//1
    ncclCommInitRankDev(newcomm, nranks, 1, &commId, myrank, cudaDev, &config,
        ncclInit
        parseCommConfig(comm, config), res, fail);
            comm->config.maxCTAs = internalConfigPtr->maxCTAs;
            //每个 SM (Streaming Multiprocessor) 上最少/最多的 CTA (Cooperative Thread Array，即线程块) 数量。
            envConfigOverride(comm)
                comm->config.maxCTAs = MAXCHANNELS;
        job->nId = nId;//1
        job->comm = comm;
        job->nranks = nranks;
        job->myrank = myrank;
        job->cudaDev = cudaDev;
        memcpy(job->commId, commId,
        commIdEnv = ncclGetEnv("NCCL_COMM_ID");
        if (commIdEnv && myrank == 0) {
            bootstrapCreateRoot
        }
        ncclAsyncLaunch((struct ncclAsyncJob*)job, ncclCommInitRankFunc, NULL, ncclCommInitJobFree, comm)
            ncclCommInitRankFunc(job)
                ncclInitKernelsForDevice
                    cudaFuncSetAttribute
                commAlloc(comm, NULL, job->nranks, job->myrank)
                    ncclNetPluginLoad(comm)//no
                    ncclNetInit(comm)
                        for (int i=0; i<3; i++) {
                        //获取ip，网卡名
                            netGetState(i, &state)
                                ncclNets[i]->init(ncclDebugLog, ncclProfilerCallback)
                            comm->ncclNet = ncclNets[i];
                            break
```

```
ncclNet_t* ncclNets[NCCL_NET_MAX_PLUGINS] = { nullptr, &ncclNetIb, &ncclNetSocket };
enum ncclNetState ncclNetStates[NCCL_NET_MAX_PLUGINS] = { ncclNetStateInit, ncclNetStateInit, ncclNetStateInit };

ncclNet_t ncclNetIb = {
  "IB",
  ncclIbInit,
  ...
  ncclIbMakeVDevice
};
struct ncclIbvSymbols ibvSymbols;//函数指针
static char ncclIbIfName[MAX_IF_NAME_SIZE+1];//元数据交换
static union ncclSocketAddress ncclIbIfAddr;
struct ncclIbDev ncclIbDevs[MAX_IB_DEVS];//数据传输
struct ncclIbMergedDev ncclIbMergedDevs[MAX_IB_VDEVS];

ncclResult_t ncclIbInit(ncclDebugLogger_t logFunction, ncclProfilerCallback_t profFunction) {
    wrap_ibv_symbols()
        buildIbvSymbols(&ibvSymbols)
        //将动态加载的 IBV 库函数符号（如 ibv_get_device_list）绑定到 NCCL 内部的结构体（ibvSymbols）中
            ASSIGN_SYM(ibvSymbols, ibv_get_device_list, ibv_internal_get_device_list);
    ncclNIbDevs = 0;
    ncclNMergedIbDevs = 0;
    ncclFindInterfaces(ncclIbIfName, &ncclIbIfAddr, MAX_IF_NAME_SIZE, 1) //xgbe0//eth0
    const char* userIbEnv = ncclGetEnv("NCCL_IB_HCA");
    //指定使用的 InfiniBand 适配器
    //mlx5_1,mlx5_8,mlx5_6,mlx5_4,mlx5_2,mlx5_9,mlx5_7,mlx5_5,mlx5_3
    //一共有10个，mlx5_0没用
    //mlx5_2,mlx5_9,mlx5_7,mlx5_5,mlx5_3,mlx5_1,mlx5_8,mlx5_6,mlx5_4
    int nUserIfs = parseStringList(userIbEnv, userIfs, MAX_IB_DEVS);
    wrap_ibv_get_device_list(&devices, &nIbDevs)
    for (int d=0; d<nIbDevs && ncclNIbDevs<MAX_IB_DEVS; d++) {
        wrap_ibv_open_device(&context, devices[d])
        wrap_ibv_query_device(context, &devAttr)
        for (int port_num = 1; port_num <= devAttr.phys_port_cnt; port_num++) {
            wrap_ibv_query_port(context, port_num, &portAttr)
            matchIfList(devices[d]->name, port_num, userIfs, nUserIfs, searchExact) ^ searchNot)
            ncclIbDevs[ncclNIbDevs].link = portAttr.link_layer;//2
            //1==IB,2==RoCE
            ncclIbDevs[ncclNIbDevs].speed = ncclIbSpeed(portAttr.active_speed) * ncclIbWidth(portAttr.active_width);
            ncclIbGetPciPath(ncclIbDevs[ncclNIbDevs].devName, &ncclIbDevs[ncclNIbDevs].pciPath, &ncclIbDevs[ncclNIbDevs].realPort)
            pthread_create(&ncclIbAsyncThread, NULL, ncclIbAsyncThreadMain, ncclIbDevs + ncclNIbDevs)
            //处理错误
            int vDev;
            ncclNetVDeviceProps_t vProps = {0};
            vProps.ndevs = 1;
            vProps.devs[0] = ncclNIbDevs;
            NCCLCHECK(ncclIbMakeVDeviceInternal(&vDev, &vProps));
            //将多个设备合一
                mDev = ncclIbMergedDevs + ncclNMergedIbDevs;
                for (int i = 0; i < props->ndevs; i++) {
                    ncclIbDev* dev = ncclIbDevs + props->devs[i];
                    mDev->vProps.devs[mDev->vProps.ndevs++] = props->devs[i];
                    mDev->speed += dev->speed;
                *d = ncclNMergedIbDevs++;
            ncclNIbDevs++;
            nPorts++;
    INFO(NCCL_INIT|NCCL_NET, "NET/IB : Using%s %s; OOB %s:%s", line, ncclIbRelaxedOrderingEnabled ? "[RO]" : "",
      ncclIbIfName, ncclSocketToString(&ncclIbIfAddr, addrline));
    //OOB（Out-of-Band）在此上下文中指带外通信（Out-of-Band Communication），是 NCCL 用于跨节点元数据交换的独立通信通道，与主数据传输（RDMA/IB）分离。
```

comm初始化：设置ncclNet，获取GPU的busid，cudaCompCap

```
struct ncclComm {
  ncclNet_t* ncclNet;

ncclCommInitRankDev
    ncclAsyncLaunch((struct ncclAsyncJob*)job, ncclCommInitRankFunc, NULL, ncclCommInitJobFree, comm)
        ncclCommInitRankFunc(job)
            commAlloc(comm, NULL, job->nranks, job->myrank)
                ncclNetPluginLoad(comm)//no
                ncclNetInit(comm)
                getBusId(comm->cudaDev, &comm->busId)
                ncclNvmlDeviceGetHandleByPciBusId
                ncclNvmlDeviceGetIndex(nvmlDev, (unsigned int*)&comm->nvmlDev)
                comm->compCap = ncclCudaCompCap();
                    cudaDeviceGetAttribute
                    return ccMajor*10+ccMinor;
                struct ncclSharedResources* sharedRes = NULL;
                sharedRes->owner = comm;
                sharedRes->tpNRanks = comm->nRanks;
                comm->sharedRes = sharedRes;
                if (comm->topParentRanks == NULL) {
                    NCCLCHECK(ncclCalloc(&comm->topParentRanks, comm->nRanks));
                    for (int i = 0; i < comm->nRanks; ++i)
                      comm->topParentRanks[i] = i;
                }
            comm->commHash = commIdHash = getHash(job->commId->internal, NCCL_UNIQUE_ID_BYTES);                
```

# bootstrapInit

ProxyService为数据传输准备资源，ncclProxyProgress实现rdma

bootstrapInit创建bootstrapNet(包括ProxyService addr)

1. 创建了两个listen socket(bootstrapNetIfAddr)，一个给root，一个给pre peer连接

2. 连接root(ncclUniqueId)把两个listen发给root

3. 接受root连接

4. 接收next peer addr

5. 连接next peer，接收pre peer，形成Ring

6. 创建listen sockets，RingAllGather state->peerP2pAddresses(addr for peers to contact me in P2P in bootstrapNet), state->peerProxyAddresses(addr for ProxyService), state->peerProxyAddressesUDS

```
struct ncclComm {
  ncclNet_t* ncclNet;
  void* bootstrap;
  struct ncclProxyState* proxyState;

struct bootstrapState {
  struct bootstrapRing_t ring;
  struct bootstrapListen_t listen;// socket for peers to contact me in P2P bootstrapNet
  ncclNet_t* net;//应该是无用的
  uint64_t* peerProxyAddressesUDS;//
  union ncclSocketAddress* peerProxyAddresses;//addr for ProxyService
  union ncclSocketAddress* peerP2pAddresses;// addr for peers to contact me in P2P in bootstrapNet
  struct unexConn* unexpectedConnections;
  int cudaDev;
  int rank;
  int nranks;
  uint64_t magic;
  volatile uint32_t* abortFlag;
};

struct bootstrapRing_t {
  union {
    struct {
      void *sendComm, *recvComm;
      ncclNetDeviceHandle_t *sendDevHandle, *recvDevHandle;
    } net;
    struct {
      struct ncclSocket recv;//接收pre peer连接
      struct ncclSocket send;//连接next peer
    } socket;
  };
};

struct bootstrapListen_t {
  struct ncclSocket peerSocket; // socket for peers to contact me in P2P bootstrapNet
  union {
    struct {
      int dev;
      void* comm;
      char handle[NCCL_NET_HANDLE_MAXSIZE];
    } net;
    struct ncclSocket socket; //socket for ring pre peer to contact me
  };
};

struct extInfo {
  int rank;                                
  int nranks;                              
  int iroot;                              
  int nroots;                             
  union ncclSocketAddress listenRootAddress; // address of my listenSocket for the root to contact
  union ringConnectInfo connectInfo;
};

union ringConnectInfo {
  union ncclSocketAddress addr;//addr for ring neightbor to contact me
  char handle[NCCL_NET_HANDLE_MAXSIZE];
};

struct ncclProxyState {
  struct ncclSocket* listenSock;//peerAddresses socket
  struct ncclIpcSocket ipcSock;//peerAddressesUDS socket
  union ncclSocketAddress* peerAddresses;//addr for ProxyService
  uint64_t *peerAddressesUDS; // 

ncclCommInitRankDev
    ncclAsyncLaunch((struct ncclAsyncJob*)job, ncclCommInitRankFunc, NULL, ncclCommInitJobFree, comm)
        ncclCommInitRankFunc(job)
            ncclInitKernelsForDevice
            commAlloc(comm, NULL, job->nranks, job->myrank)
            comm->commHash = commIdHash = getHash(job->commId->internal, NCCL_UNIQUE_ID_BYTES);
            bootstrapInit(job->nId, (struct ncclBootstrapHandle*)job->commId, comm)
                struct bootstrapState* state;
                struct extInfo info ;//发送给root两个listen addr
                union ringConnectInfo nextPeer;//接收ring下一个gpu的addr
                state->net = comm->ncclNet;
                comm->bootstrap = state;
                info.nranks = nranks;
                info.nroots = nHandles;
                //ncclParamBootstrapNetEnable())=0
                createListenSocket(comm, comm->magic, &STATE_LISTEN(state, socket), &info.connectInfo.addr,
                // create socket for ring neightbor to contact me
                    ncclSocketInit(socket, &bootstrapNetIfAddr,//xgbe0//eth0
                    //fd=socket(ip)
                    ncclSocketListen(socket)
                    //获取port，listen
                    ncclSocketGetAddr(listenSock, &handle->addr)
                createListenSocket(comm, BOOTSTRAP_HANDLE(handles, curr_root)->magic, &listenSockRoot, &info.listenRootAddress
                // Create socket for root to contact me
                info.rank = rank;
                info.iroot = curr_root;
                sendToRoot(BOOTSTRAP_HANDLE(handles, curr_root), comm, &info)
                //传输两个listen socket给root
                    ncclSocketInit(&sock, &handle->addr,
                    ncclSocketConnect(&sock)
                    //ip是目的ip，本地不需要绑定
                        sock->state = ncclSocketStateConnecting;
                        socketProgressState
                            socketStartConnect
                                connect(sock->fd, &sock->addr.sa, sock->salen)
                    socketSend(&sock, info,
                    // Additional socket based functions, first send the size, then send the message
                    ncclSocketClose(&sock)
                ncclSocketInit(&sock)
                // 000
                ncclSocketAccept(&sock, &listenSockRoot)
                    memcpy(sock, listenSock, sizeof(struct ncclSocket));
                    sock->acceptFd = listenSock->fd;
                    sock->state = ncclSocketStateAccepting;
                    sock->fd = accept(sock->acceptFd, (struct sockaddr*)&sock->addr, &socklen);
                    //记录客户端ip
                socketRecv(&sock, &nextPeer, sizeof(nextPeer))
                ncclSocketClose(&sock)
                ncclSocketClose(&listenSockRoot)
                socketRingConnect(&nextPeer.addr, &STATE_RING(state, socket.send), &STATE_LISTEN(state, socket), &STATE_RING(state, socket.recv)
                //形成ring
                    ncclSocketInit(sendSocket, addr
                    //next peer addr
                    ncclSocketConnect(sendSocket
                    ncclSocketInit(recvSocket)
                    //000
                    ncclSocketAccept(recvSocket, listenSock)
                    //接收pre peer连接
                createListenSocket(comm, comm->magic, proxySocket, state->peerProxyAddresses + rank,
                //创建listen socket，保存地址
                getUDS(state->peerProxyAddressesUDS + rank)
                //随机数？？？
                createListenSocket(comm, comm->magic, &STATE_LISTEN(state, peerSocket), &peerSocketAddress
                //create a socket for others to reach out (P2P)
                memcpy(state->peerP2pAddresses + rank, &peerSocketAddress,
                ringAllInfo(comm, state, state->peerP2pAddresses, state->peerProxyAddresses, state->peerProxyAddressesUDS, 
                    bootstrapAllGather(state, ringData, 
                        socketRingAllGather(&STATE_RING(state, socket.send), &STATE_RING(state, socket.recv), rank, nranks, (char*)allData, size), res, exit);
                ncclProxyInit(comm, proxySocket, state->peerProxyAddresses, state->peerProxyAddressesUDS)
                    comm->proxyState = comm->sharedRes->proxyState;
                    comm->proxyState->listenSock = sock;
                    comm->proxyState->peerAddresses = peerAddresses;
                    comm->proxyState->peerAddressesUDS = peerAddressesUDS;
                    NCCLCHECK(ncclIpcSocketInit(&comm->proxyState->ipcSock, comm->rank, peerAddressesUDS[comm->rank], comm->abortFlag));
                    // Create a Unix Domain Socket
                        fd = socket(AF_UNIX, SOCK_DGRAM, 0)
                        // Create unique name for the socket.
                        int len = snprintf(temp, NCCL_IPC_SOCKNAME_LEN, NCCL_IPC_SOCKNAME_STR, rank, hash);
                        strncpy(cliaddr.sun_path, temp, len);
                        bind(fd, (struct sockaddr *)&cliaddr, sizeof(cliaddr))
                        handle->fd = fd;
```

# xmlAddNode

h800物理连接

```
cpu-                   (qpi)-cpu
|  |  |    |     |     |     |     |     |     |
n0 n1 P    P     P     P     P     P     P     P
      | |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
     g0 n5 g1 n4 g2 n3 g3 n2 g4 n7 g5 n6 g6 n9 g7 n8
```
```
交换信息：cudaCompCap
ncclTopoFillGpu//创建GPU到cpu的xml
  xmlAddNode(pci)
  ncclTopoGetXmlFromSys
    从/sys获取max_link_speed,max_link_width
    if parent should be cpu: xmlAddNode(cpu)
    else: xmlAddNode(pci)
          ncclTopoGetXmlFromSys
  ncclTopoGetXmlFromGpu
    xmlAddNode(gpu)
    xmlAddNode(nvlink)
ncclTopoProcessNet//创建所有网卡的xml
  ncclTopoPopulateNics
    从ncclNet获取speed
    ncclTopoGetVNicParent
    ncclTopoFillNet
      xmlAddNode(nic) 
      xmlAddNode(net) 
  ...
allgather同一机器的xml
ncclTopoFuseXml
```
system->cpu(->pci)->pci->gpu->nvlink

system->cpu(->pci)->pci->nic->net
```
<system version="1">
  <cpu host_hash="0xa9832a2e9e166623" numaid="0" affinity="00000000,0000ffff,ffffffff,00000000,0000ffff,ffffffff" arch="x86_64" vendor="GenuineIntel" familyid="6" modelid="143">
    <pci busid="0000:16:00.0" class="0x060400" vendor="0x1000" device="0xc030" subsystem_vendor="0x1000" subsystem_device="0x100b" link_speed="32.0 GT/s PCIe" link_width="16">
      <pci busid="0000:18:00.0" class="0x030200" vendor="0x10de" device="0x5663" subsystem_vendor="0x10de" subsystem_device="0x16c1" link_speed="32.0 GT/s PCIe" link_width="16">
        <gpu dev="0" sm="90" rank="0" gdr="1">
          <nvlink target="0000:04:00.0" count="5" tclass="0x068000"/>
          <nvlink target="0000:06:00.0" count="4" tclass="0x068000"/>
          <nvlink target="0000:05:00.0" count="5" tclass="0x068000"/>
          <nvlink target="0000:03:00.0" count="4" tclass="0x068000"/>
        </gpu>
      </pci>
      <pci busid="0000:19:00.0" class="0x020000" vendor="0x15b3" device="0x1021" subsystem_vendor="0x15b3" subsystem_device="0x0023" link_speed="32.0 GT/s PCIe" link_width="16">
        <nic>
          <net name="mlx5_5" dev="4" latency="0" speed="400000" port="1" guid="0x74cf1d0003c0639c" maxconn="131072" gdr="1"/>
        </nic>
      </pci>
    </pci>
    <pci busid="0000:26:00.1" class="0x020000" vendor="0x15b3" device="0x101d" subsystem_vendor="0x15b3" subsystem_device="0x0016" link_speed="16.0 GT/s PCIe" link_width="16">
      <nic>
        <net name="mlx5_1" dev="0" latency="0" speed="100000" port="1" guid="0xd617e90003c288a0" maxconn="131072" gdr="1"/>
      </nic>
    </pci>
```

```
struct ncclComm {
  ncclNet_t* ncclNet;
  void* bootstrap;
  struct ncclProxyState* proxyState;
  struct ncclPeerInfo* peerInfo;

struct ncclXml {
  int maxIndex, maxNodes;
  struct ncclXmlNode nodes[1];
};

struct ncclXmlNode {
  char name[MAX_STR_LEN+1];
  struct {
    char key[MAX_STR_LEN+1];
    char value[MAX_STR_LEN+1];
  } attrs[MAX_ATTR_COUNT+1]; // Need an extra one to consume extra params
  int nAttrs;
  int type;
  struct ncclXmlNode* parent;
  struct ncclXmlNode* subs[MAX_SUBS];
  int nSubs;
};

ncclTopoNetState netStates[NCCL_NET_MAX_PLUGINS] = {};

struct ncclTopoNetState {
  int nVirtualNics;
  int nPhysicalNics;
  const char* name;
};

initTransportsRank
    fillInfo(comm, comm->peerInfo+rank, comm->commHash),
        info->cuMemSupport = ncclCuMemEnable();//1
        ncclGpuGdrSupport(comm, &info->gdrSupport));//1
        info->cudaCompCap = comm->minCompCap = comm->maxCompCap = comm->compCap;
        bootstrapAllGather(comm->bootstrap, comm->peerInfo,
        socketRingAllGather
    for
        if (comm->peerInfo[i].hostHash != comm->peerInfo[rank].hostHash) nNodes++;
    do
        for (int i = 0; i < nranks; i++) comm->minCompCap = std::min(comm->minCompCap, comm->peerInfo[i].cudaCompCap);
        for (int i = 0; i < nranks; i++) comm->maxCompCap = std::max(comm->maxCompCap, comm->peerInfo[i].cudaCompCap);
        for    
            if (intraProcRanks == 0) intraProcRank0 = i;
            if (i == rank) intraProcRank = intraProcRanks;
            //进程内的排序
            intraProcRanks++;
    dumpXmlFile = ncclGetEnv("NCCL_TOPO_DUMP_FILE");
    if (dumpXmlFile) {
        NCCLCHECKGOTO(ncclTopoGetSystem(comm, NULL, dumpXmlFile), ret, fail);
    }
    ncclTopoGetSystem
        const char* xmlTopoFile = ncclGetEnv("NCCL_TOPO_FILE");
        if (xmlTopoFile) {
            NCCLCHECKGOTO(ncclTopoGetXmlFromFile(xmlTopoFile, xml, 1), ret, fail);
        } else {
            // Try default XML topology location
            NCCLCHECKGOTO(ncclTopoGetXmlFromFile("/var/run/nvidia-topologyd/virtualTopology.xml", xml, 0), ret, fail);
        }
        xmlAddNode(xml, NULL, "system", &top)
        //(struct ncclXml* xml, struct ncclXmlNode* parent, const char* subName, struct ncclXmlNode** sub)
            s = xml->nodes+xml->maxIndex++;
            s->parent = parent;
            parent->subs[parent->nSubs++] = s;
            strncpy(s->name, subName, MAX_STR_LEN);
        xmlSetAttrInt(top, "version", NCCL_TOPO_XML_VERSION)
            xmlGetAttrIndex(node, attrName, &index)
            //找到kv在数组里的下标
            if (index == -1) {
                strncpy(node->attrs[index].key, attrName, MAX_STR_LEN);
            snprintf(node->attrs[index].value, MAX_STR_LEN, "%d", value);
        ncclTopoFillGpu(xml, busId, &node)
        //system->cpu->pci->gpu->nvlink
                               ->nvlink
            ncclTopoGetPciNode(xml, busId, &node)
            //0000:9b:00.0
                xmlFindTagKv(xml, "pci", pciNode, "busid", busId)
                //找名字是pci且pci.busid=busid的node
                    n = xml->nodes+i;
                    if (strcmp(n->name, tagName) == 0) {
                      const char* value;
                      NCCLCHECK(xmlGetAttr(n, attrName, &value));
                      if (value && strcmp(value, attrValue) == 0) {
                        *node = n;
                        return ncclSuccess;
                      }
                    }
                if (*pciNode == NULL) {
                    NCCLCHECK(xmlAddNode(xml, NULL, "pci", pciNode));
                    //(struct ncclXml* xml, struct ncclXmlNode* parent, const char* subName, struct ncclXmlNode** sub)
                    NCCLCHECK(xmlSetAttr(*pciNode, "busid", busId));
                }
            xmlSetAttrIfUnset(node, "class", "0x03")
            //pci.class=0x03
            ncclTopoGetXmlFromSys(node, xml)
            // Fill info, then parent
                xmlGetAttr(pciNode, "busid", &busId)
                //0000:9b:00.0, 0000:98:00.0
                getPciPath(busId, &path);
                ///sys/devices/pci0000:97/0000:97:01.0/0000:98:00.0/0000:99:01.0/0000:9b:00.0
                ncclTopoSetAttrFromSys(pciNode, path, "class", "class")
                //从sys获取
                //(struct ncclXmlNode* pciNode, const char* path, const char* fileName, const char* attrName) 
                    ncclTopoGetStrFromSys(path, fileName, strValue)
                    //读path/class==0x030200
                        snprintf(filePath, sizeof(filePath), "%s/%s", path, fileName);
                        fopen(filePath, "r")
                        fread(strValue+offset
                    xmlSetAttr(pciNode, attrName, strValue)
                    //pci.class=0x030200
                ncclTopoSetAttrFromSys(pciNode, path, "vendor", "vendor")
                //芯片制造商
                ncclTopoSetAttrFromSys(pciNode, path, "device", "device")
                //芯片的具体型号
                ncclTopoSetAttrFromSys(pciNode, path, "subsystem_vendor", "subsystem_vendor")
                //组装该硬件的厂商
                ncclTopoSetAttrFromSys(pciNode, path, "subsystem_device", "subsystem_device")
                //组装设备的具体型号
                ncclTopoGetStrFromSys(path, "max_link_speed", deviceSpeedStr)
                ncclTopoGetStrFromSys(path, "../max_link_speed", portSpeedStr)
                xmlSetAttr(pciNode, "link_speed", portSpeed < deviceSpeed ? portSpeedStr : deviceSpeedStr)
                ncclTopoGetStrFromSys(path, "max_link_width", strValue)
                ncclTopoGetStrFromSys(path, "../max_link_width", strValue)
                xmlSetAttrInt(pciNode, "link_width", std::min(deviceWidth,portWidth))
                ncclTopoGetStrFromSys(path, "numa_node", numaIdStr)
                //0
                // Go up one level in the PCI tree. Rewind two "/" and follow the upper PCI
                // switch, or stop if we reach a CPU root complex.
                if (checkBDFFormat(path+start+1) == 0) {
                    // This a CPU root complex. Create a CPU tag and stop there.
                    struct ncclXmlNode* topNode;
                    NCCLCHECK(xmlFindTag(xml, "system", &topNode));
                    //找到system
                    NCCLCHECK(xmlGetSubKv(topNode, "cpu", &parent, "numaid", numaIdStr));
                    //找到system.sub.name==cpu&&cpu.numaid==numaIdStr
                    if (parent == NULL) {
                      NCCLCHECK(xmlAddNode(xml, topNode, "cpu", &parent));
                      //system.sub=cpu
                      NCCLCHECK(xmlSetAttrLong(parent, "host_hash", getHostHash()));
                      //cpu.host_hash
                      NCCLCHECK(xmlSetAttr(parent, "numaid", numaIdStr));
                      //cpu.numaid
                    }
                } 
                else if (slashCount == 2) {//为什么是2
                // Continue on the upper PCI switch
                    for (int i = strlen(path)-1; i>0; i--) {
                      if (path[i] == '/') {
                        NCCLCHECK(xmlFindTagKv(xml, "pci", &parent, "busid", path+i+1));
                        //找name==pci&&pci.busid==path+i+1
                        if (parent == NULL) {
                          NCCLCHECK(xmlAddNode(xml, NULL, "pci", &parent));
                          NCCLCHECK(xmlSetAttr(parent, "busid", path+i+1));
                        }
                        //构建上一级pci
                        break;
                      }
                    }
                }
                pciNode->parent = parent;
                parent->subs[subIndex] = pciNode;
                parent->nSubs++;
                if (strcmp(parent->name, "pci") == 0) {
                    NCCLCHECK(ncclTopoGetXmlFromSys(parent, xml));
                    //递归
                } else if (strcmp(parent->name, "cpu") == 0) {
                    NCCLCHECK(ncclTopoGetXmlFromCpu(parent, xml));
                        ncclTopoSetAttrFromSys(cpuNode, cpumaskPath, "cpumap", "affinity")
                }
            NCCLCHECK(ncclNvmlDeviceGetHandleByPciBusId(busId, &nvmlDev));
            //通过 PCI 总线 ID 获取 GPU 的 NVML 设备句柄
            NCCLCHECK(ncclTopoGetXmlFromGpu(node, nvmlDev, xml, gpuNode));
                //gpu对应两个node，一个pci，一个gpu，gpu带nvlinks
                xmlAddNode(xml, pciNode, "gpu", &gpuNode)
                //pci.sub=gpu
                ncclNvmlDeviceGetIndex(nvmlDev, (unsigned int*)&dev)
                xmlSetAttrInt(gpuNode, "dev", dev)
                //gpu.dev
                ncclNvmlDeviceGetCudaComputeCapability
                xmlSetAttrInt(gpuNode, "sm", cudaMajor*10+cudaMinor)
                for (int l=0; l<maxNvLinks; ++l) {
                    ncclNvmlDeviceGetNvLinkCapability
                    ncclNvmlDeviceGetNvLinkState
                    ncclNvmlDeviceGetNvLinkRemotePciInfo
                    char* p = remoteProc.busId;
                    //nvlink 另一端的gpu
                    xmlAddNode(xml, gpuNode, "nvlink", &nvlNode)
                    //gpu.sub.name=nvlink
                    xmlSetAttr(nvlNode, "target", lowerId)
                    //nvlink.target
                for (int s=0; s<gpuNode->nSubs; s++) {
                    xmlGetAttr(sub, "target", &busId)
                    getPciPath(busId, &path);
                    ncclTopoSetAttrFromSys(sub, path, "class", "tclass")
                    //读path/class的值,设nvlink.tclass
        xmlSetAttrInt(node, "keep", 1)
        xmlSetAttrInt(node, "rank", comm->rank)
        xmlInitAttrInt(node, "gdr", comm->peerInfo[comm->rank].gdrSupport)
        ncclTopoNetState* state;
        ncclTopoGetSharedState(&state, comm->ncclNet->name, netStates)
            for (int i = 0; i < NCCL_NET_MAX_PLUGINS; i++) {
                // Empty slot
                if (states[i].name == NULL) {
                    states[i].nVirtualNics = -1;
                    states[i].nPhysicalNics = -1;
                    states[i].name = strdup(name);//IB
                    *state = states + i;
        ncclTopoProcessNet(comm, xml, 0, dumpXmlFile, state,
comm->ncclNet->getProperties, comm->ncclNet->makeVDevice, comm->ncclNet->devices, comm->ncclNet->name)
        //先创建物理网卡xml，ib要使用虚拟网卡，将相近的物理网卡合并成虚拟网卡，将物理网卡标记为删除，创建虚拟网卡xml
        //ib的网卡本身就是merged的
        //(ncclComm_t comm, ncclXml* xml, int coll, const char* dumpXmlFile, ncclTopoNetState* state, 
        //ncclResult_t (*getProperties)(int, ncclNetProperties_t*), 
        //ncclResult_t (*makeVDevice)(int*, ncclNetVDeviceProps_t*), ncclResult_t (*devices)(int*), const char* netName)
            int usePhysicalDevices = (dumpXmlFile || makeVDevice == NULL);//0
            if (state->nPhysicalNics == -1) NCCLCHECK(devices(&state->nPhysicalNics));
/root/paddlejob/workspace/env_run/output/cyy/Nccl/nccl/src/transport/net_ib.cc
                *ndev = ncclNMergedIbDevs;//9
            ncclTopoPopulateNics(comm, xml, 0, state->nPhysicalNics, getProperties, netName, coll, 1, 0)
            //(ncclComm_t comm, ncclXml* xml, int startIndex, int endIndex, 
            //ncclResult_t (*getProperties)(int, ncclNetProperties_t*), const char* netName, int coll, int keep, int virtualNics)
                for (int n = startIndex; n < endIndex; n++)
                    getProperties(n, &props)
/root/paddlejob/workspace/env_run/output/cyy/Nccl/nccl/src/transport/net_ib.cc
                        mergedDev = ncclIbMergedDevs + dev
                        ncclIbGetPhysProperties(mergedDev->vProps.devs[0], props)
                        props->name = mergedDev->devName;
                        props->speed = mergedDev->speed;
                    if (virtualNics) {
                        struct ncclXmlNode* net = NULL;
                        NCCLCHECK(xmlFindTagKv(xml, "net", &net, "name", props.name));
                        // In the event of multithreaded use case, we need to re-discover the shared parent of the given devices for this vNIC
                        // Only run this if the net doesn't exist locally - this may alter the XML state
                        if (net == NULL) NCCLCHECK(ncclTopoGetVNicParent(xml, getProperties, &props.vProps, &parent));
                    }
                    ncclTopoFillNet(xml, props.pciPath, props.name, &netNode, parent)
                    //system->cpu->pci->nic->net
                        else if (pciSysPath) {
                        ///sys/devices/pci0000:25/0000:25:01.0/0000:26:00.1
                            ncclTopoGetPciNode(xml, busId, &parent)
                            ///0000:26:00.1
                            //pci.busid
                            xmlSetAttrIfUnset(parent, "class", "0x02")
                            ncclTopoGetXmlFromSys(parent, xml)
                        }
                        xmlAddNode(xml, parent, "nic", &nicNode)
                        //pci.sub=nic
                        xmlAddNode(xml, nicNode, "net", netNode)
                        //nic.sub=net
                        xmlSetAttr(*netNode, "name", netName)
                        //net.name=netname//mlx5_1
                    xmlGetAttr(netNode, "coll", &colAttr)
                    if (colAttr == NULL || coll != 0 || strcmp(colAttr,"1") != 0) NCCLCHECK(xmlSetAttrInt(netNode, "keep", keep));//true
                    xmlInitAttrInt(netNode, "speed", props.speed)
                    xmlInitAttrInt(netNode, "gdr", gdrSupport)
            if (!usePhysicalDevices) {
                if (state->nVirtualNics == -1) {
                    ncclTopoMakeVNics(comm, xml, makeVDevice, getProperties, state->nPhysicalNics));
                        for (int i = 0; i < physicalDevs; i++) {
                            NCCLCHECKGOTO(getProperties(i, props + i), res, out);
                            struct ncclXmlNode* physNetNode;
                            NCCLCHECKGOTO(xmlFindTagKv(xml, "net", &physNetNode, "name", props[i].name), res, out);
                            physNetNodes[i] = physNetNode;
                            TRACE(NCCL_GRAPH, "Found physical ncclNet node %d %s", i,  props[i].name);
                        }
                        int mergeLevel;
                        mergeLevel = PATH_PORT;
                        NCCLCHECKGOTO(ncclTopoAutoMerge(comm, xml, mergeLevel, placedDevs, props, physNetNodes, physicalDevs, makeVDevice), res, out);
                        //虚拟设备翻倍了？？？
                            ncclCalloc(&paths, nPhysDevs*nPhysDevs);
                            for (int i = 0; i < nPhysDevs; i++) {
                                for (int j = 0; j < nPhysDevs; j++) {
                                    struct ncclXmlNode* nodes[2];
                                    nodes[0] = physNetNodes[i];
                                    nodes[1] = physNetNodes[j];
                                    struct ncclXmlNode* parent;
                                    NCCLCHECKGOTO(ncclTopoGetPath(nodes, 2, &paths[i*nPhysDevs + j], &parent), res, out);
                                    //Find the common parent xmlNode between the given set of nodes
                                        ...
                                        if (common == NULL) {
                                            *path = PATH_DIS;
                                        } else if (strcmp(common->name,"system") == 0) {
                                            *path = PATH_SYS;
                                        } else if (strcmp(common->name, "cpu") == 0) {
                                            *path = PATH_PHB;
                                        } else if (strcmp(common->name, "nic") == 0) {
                                            *path = PATH_PORT;
                                        } else if (strcmp(common->name, "net") == 0) {
                                            *path = PATH_PORT;
                                        } else if (ncclTopoCheckPix(common, nodes, nNodes)) {
                                            *path = PATH_PIX;
                                        } else {
                                            *path = PATH_PXB;
                                        }
                            for (int i = 0; i < nPhysDevs; i++) {
                                vProps.devs[vProps.ndevs++] = i;
                                for (int j = 0; j < nPhysDevs; j++) {
                                    if (paths[i*nPhysDevs + j] <= mergeLevel//no
                                        vProps.devs[vProps.ndevs++] = j;
                                ncclTopoMakeVnic(comm, xml, &vProps, physNetNodes, &netNode, makeVDevice);     
                                    makeVDevice(&vDevIndex, vProps)
                                    //增加一个vdevice，包含vProps里的device
                                        mDev = ncclIbMergedDevs + ncclNMergedIbDevs;
                                        for (int i = 0; i < props->ndevs; i++) {
                                            dev = ncclIbDevs + props->devs[i];
                                            mDev->vProps.devs[mDev->vProps.ndevs++] = props->devs[i];
                                        *d = ncclNMergedIbDevs++;
                    NCCLCHECK(devices(&nDevs));
                    state->nVirtualNics = nDevs - state->nPhysicalNics;
                    //18-9
                    ncclTopoPopulateNics(comm, xml, 0, state->nPhysicalNics, getProperties, netName, coll, 0, 0)
                    //9个net.keep==0
                    ncclTopoPopulateNics(comm, xml, state->nPhysicalNics, state->nPhysicalNics+state->nVirtualNics, getProperties, netName, coll, 1, 1)
                    //9个net.keep=1
        ncclTopoTrimXml(xml)
        //没有删除
            ncclTopoTrimXmlRec
                NCCLCHECK(xmlGetAttr(node, "keep", &str));
                if (str && strcmp(str, "1") == 0) {
                    NCCLCHECK(xmlUnsetAttr(node, "keep"));
                    *keep = 1;
                } else {
                    *keep = 0;
                    for (int s=0; s<nSubs; s++) {
                      int k = 0;
                      NCCLCHECK(ncclTopoTrimXmlRec(subs[s], &k));
                      *keep += k;
                    }
                    if (*keep == 0 && // Trim PCI switches or CPU with no used GPU/NIC under them.
                        (strcmp(node->name, "pci") == 0 || strcmp(node->name, "cpu") == 0)) {
                      NCCLCHECK(xmlRemoveNode(node));
                    }
                }
        rankXml = (struct ncclXml*)(mem+xmlMemSize(NCCL_TOPO_XML_MAX_NODES)*localRank);
        memcpy(rankXml, xml, xmlMemSize(NCCL_TOPO_XML_MAX_NODES));
        ncclTopoConvertXml(rankXml, (uintptr_t)xml->nodes, 1), ret, fail)
        //序列化(减去起始地址
        bootstrapIntraNodeAllGather(comm->bootstrap, localRanks, localRank, nLocalRanks, mem,
        xml->maxIndex = 0;//44，自己的gpu往上，加所有网卡
        for (int i = 0; i < nLocalRanks; i++) {
            struct ncclXml* peerXml = (struct ncclXml*)(mem+xmlMemSize(NCCL_TOPO_XML_MAX_NODES)*i);
            NCCLCHECKGOTO(ncclTopoConvertXml(peerXml, (uintptr_t)peerXml->nodes, 0), ret, fail);
            //反序列化
            NCCLCHECKGOTO(ncclTopoFuseXml(xml, peerXml), ret, fail);
                if (topNodeDst == NULL) {
                    xmlAddTree(dst, NULL, src->nodes);
                    //(struct ncclXml* dst, struct ncclXmlNode* parent, struct ncclXmlNode* srcNode)
                        struct ncclXmlNode* dstNode = dst->nodes+dst->maxIndex++;
                        *dstNode = *srcNode;
                        dstNode->parent = parent;
                            if (parent) {
                                parent->subs[parent->nSubs++] = dstNode;
                        }
                        dstNode->nSubs = 0;
                        // Recursively copy the subtree(s)
                        for (int i=0; i<srcNode->nSubs; i++)
                            NCCLCHECK(xmlAddTree(dst, dstNode, srcNode->subs[i]));
                }
                xmlTopoFuseXmlRecursive(dst, topNodeDst, topNodeSrc)
        }
        //所有GPU，加所有网卡//86
        if (dumpXmlFile && comm->rank == ncclParamTopoDumpFileRank()) {
            INFO(NCCL_ENV, "NCCL_TOPO_DUMP_FILE set by environment to %s", dumpXmlFile);
            NCCLCHECKGOTO(ncclTopoDumpXmlToFile(dumpXmlFile, xml), ret, fail);
        }
```

# ncclTopoGetSystemFromXml
```
ncclTopoAddCpu
  ncclTopoCreateNode(cpu)
  ncclTopoAddPci
    ncclTopoCreateNode(gpu)
    ncclTopoAddGpu
    ncclTopoCreateNode(nic)
    ncclTopoAddNic
      ncclTopoAddNet
        ncclTopoCreateNode(net)
          ncclTopoConnectNodes(nic, net)
          ncclTopoConnectNodes(net, nic)
    ncclTopoCreateNode(pci)
      ncclTopoAddPci
    ncclTopoConnectNodes(node, parent)
    ncclTopoConnectNodes(parent, node)
ncclTopoAddNvLinks
  ncclTopoCreateNode(NVS)
  ncclTopoConnectNodes(gpu, remote)
  ncclTopoConnectNodes(remote, gpu)
ncclTopoConnectCpus
Shift all links to have upLink as last link
```
```
节点
cpu-cpu-pci-gpu-nvswitch(只有一个,和gpu的连接只有一条，多个nvlink带宽累加)-gpu
           -nic-net
       -nic-net

边
nic-net:LINK_NET
pci-gpu,pci-nic,cpu-pci,cpu-nic:LINK_PCI
gpu-nvswitch:LINK_NVL
cpu-cpu:LINK_SYS
```

```
struct ncclComm {
  ncclNet_t* ncclNet;
  void* bootstrap;
  struct ncclProxyState* proxyState;
  struct ncclPeerInfo* peerInfo;
  struct ncclTopoSystem* topo;

struct ncclTopoSystem {
  int systemId;
  uint64_t hostHashes[NCCL_TOPO_MAX_NODES];
  int nHosts;
  struct ncclTopoNodeSet nodes[NCCL_TOPO_NODE_TYPES];
  float maxBw;
  float totalBw;
};

struct ncclTopoNodeSet {
  int count;
  struct ncclTopoNode nodes[NCCL_TOPO_MAX_NODES];//顺序和xml的顺序相同
};

struct ncclTopoNode {
  int type;
  int64_t id;
  // Type specific data
  union {
    struct {
      int dev; // NVML dev number
      int rank;
      int cudaCompCap;
      int gdrSupport;
    }gpu;
    struct {
      int dev; // Plugin dev number
      uint64_t asic;
      int port;
      float bw;
      float latency;
      int gdrSupport;
      int collSupport;
      int maxChannels;
    }net;
    struct {
      int arch;
      int vendor;
      int model;
      cpu_set_t affinity;
    }cpu;
    struct {
      uint64_t device;
    }pci;
  };
  int nlinks;
  struct ncclTopoLink links[NCCL_TOPO_MAX_LINKS];// Sort links in BW descending order
  // Pre-computed paths to GPUs and NICs
  struct ncclTopoLinkList* paths[NCCL_TOPO_NODE_TYPES];
  // Used during search
  uint64_t used;
};

struct ncclTopoLink {
  int type;
  float bw;
  struct ncclTopoNode* remNode;
};
//连接类型，总带宽，连接peer

initTransportsRank
    ncclTopoGetSystem
        if (dumpXmlFile == NULL) ncclTopoGetSystemFromXml(xml, system, comm->peerInfo[comm->rank].hostHash)   
            for (int s=0; s<topNode->nSubs; s++) {
                struct ncclXmlNode* node = topNode->subs[s];
                if (strcmp(node->name, "cpu") == 0) NCCLCHECK(ncclTopoAddCpu(node, *topoSystem));
                    ncclTopoCreateNode(system, &cpu, CPU, NCCL_TOPO_ID(systemId, numaId))
                        system->nodes[type].count++;
                    for (int s=0; s<xmlCpu->nSubs; s++) {
                        struct ncclXmlNode* node = xmlCpu->subs[s];
                        if (strcmp(node->name, "pci") == 0) NCCLCHECK(ncclTopoAddPci(node, system, cpu, systemId, numaId));
                            xmlGetSub(xmlPci, "gpu", &xmlGpu)
                            if (xmlGpu != NULL) {
                                type = GPU;//去掉gpu上一层pci
                                NCCLCHECK(ncclTopoCreateNode(system, &node, type, NCCL_TOPO_ID(systemId, busId)));
                                NCCLCHECK(ncclTopoAddGpu(xmlGpu, system, node));
                            }
                            NCCLCHECK(xmlGetSub(xmlPci, "nic", &xmlNic));
                            if (xmlNic != NULL) {
                                type = NIC;//去掉nic上一层pci
                                // Ignore sub device ID and merge multi-port NICs into one PCI device.
                                struct ncclTopoNode* nicNode = NULL;
                                int64_t localNicId = NCCL_TOPO_LOCAL_NIC_ID(numaId, busId);
                                int64_t id = NCCL_TOPO_ID(systemId, localNicId);
                                NCCLCHECK(ncclTopoGetNode(system, &nicNode, type, id));
                                if (nicNode == NULL) {
                                  NCCLCHECK(ncclTopoCreateNode(system, &nicNode, type, id));
                                  node = nicNode; // Connect it to parent later on
                                }
                                NCCLCHECK(ncclTopoAddNic(xmlNic, system, nicNode, systemId));
                                    for (int s=0; s<xmlNic->nSubs; s++) {
                                        ncclTopoAddNet(xmlNet, system, nic, systemId)
                                            ncclTopoCreateNode
                                            ncclTopoConnectNodes(nic, net, LINK_NET, net->net.bw)
                                            ncclTopoConnectNodes(net, nic, LINK_NET, net->net.bw)
                            } else if (type == PCI) {
                                NCCLCHECK(ncclTopoCreateNode(system, &node, type, NCCL_TOPO_ID(systemId, busId)));
                                NCCLCHECK(xmlGetAttr(xmlPci, "vendor", &str));
                                if (str) node->pci.device += strtol(str, NULL, 0) << 48;
                                NCCLCHECK(xmlGetAttr(xmlPci, "device", &str));
                                if (str) node->pci.device += strtol(str, NULL, 0) << 32;
                                NCCLCHECK(xmlGetAttr(xmlPci, "subsystem_vendor", &str));
                                if (str) node->pci.device += strtol(str, NULL, 0) << 16;
                                NCCLCHECK(xmlGetAttr(xmlPci, "subsystem_device", &str));
                                if (str) node->pci.device += strtol(str, NULL, 0);
                                for (int s=0; s<xmlPci->nSubs; s++) {
                                  struct ncclXmlNode* xmlSubPci = xmlPci->subs[s];
                                  if (strcmp(xmlSubPci->name, "pcilink") != 0) { // PCI links will be added later
                                    NCCLCHECK(ncclTopoAddPci(xmlSubPci, system, node, systemId, numaId));
                            if (node) {
                                NCCLCHECK(ncclTopoConnectNodes(node, parent, LINK_PCI, width*speed/80.0));
                                    //pci可以连接gpu和nic
                                    for (link = node->links; link - node->links != NCCL_TOPO_MAX_LINKS && link->remNode; link++) {
                                        if (link->remNode == remNode && link->type == type) break;
                                    }
                                    if (link->remNode == NULL) node->nlinks++;
                                    link->remNode = remNode;
                                    link->bw += bw;
                                    // Sort links in BW descending order
                                    struct ncclTopoLink linkSave;
                                    memcpy(&linkSave, link, sizeof(struct ncclTopoLink));
                                    while (link != node->links) {
                                        if ((link-1)->bw >= linkSave.bw) break;
                                        memcpy(link, link-1, sizeof(struct ncclTopoLink));
                                        link--;
                                    }
                                NCCLCHECK(ncclTopoConnectNodes(parent, node, LINK_PCI, width*speed/80.0));
                            }
            }
            ncclTopoAddNvLinks(topNode, *topoSystem, NULL, 0)
              if (strcmp(node->name, "nvlink") == 0) {
                    struct ncclTopoNode* gpu = NULL;
                    int64_t pBusId;
                    NCCLCHECK(busIdToInt64(parentBusId, &pBusId));
                    pBusId = NCCL_TOPO_ID(systemId, pBusId);
                    NCCLCHECK(ncclTopoGetNode(system, &gpu, GPU, pBusId));
                    if (gpu == NULL) {
                      WARN("Add NVLink error : could not find GPU %lx", pBusId);
                      return ncclInternalError;
                    }
                    int count;
                    NCCLCHECK(xmlGetAttrInt(node, "count", &count));
                    const char* targetClass;
                    NCCLCHECK(xmlGetAttrStr(node, "tclass", &targetClass));
                    int targetType;
                    NCCLCHECK(kvConvertToInt(targetClass, &targetType, kvDictPciClass));
                    struct ncclTopoNode* remote = NULL;
                    if (targetType == GPU) {
                      // NVL P2P connection to another GPU
                      const char* target;
                      NCCLCHECK(xmlGetAttrStr(node, "target", &target));
                      int64_t busId;
                      NCCLCHECK(busIdToInt64(target, &busId));
                      NCCLCHECK(ncclTopoGetNode(system, &remote, GPU, NCCL_TOPO_ID(systemId, busId)));
                    } else if (targetType == CPU) {
                      // NVL connection to the local CPU
                      NCCLCHECK(findLocalCpu(gpu, &remote, NULL));
                    } else {
                      if (system->nodes[NVS].count == 0) {//nv switch
                        NCCLCHECK(ncclTopoCreateNode(system, &remote, NVS, 0));
                      } else {
                        remote = system->nodes[NVS].nodes;//复用，只有一个？？？
                      }
                    }
                    if (remote) {
                      float nvlBw = ncclTopoNVLinkBw(gpu->gpu.cudaCompCap);
                      NCCLCHECK(ncclTopoConnectNodes(gpu, remote, LINK_NVL, count*nvlBw));
                      if (remote->type != GPU) {
                        NCCLCHECK(ncclTopoConnectNodes(remote, gpu, LINK_NVL, count*nvlBw));
                      }
                    }
              } else {
                if (strcmp(node->name, "cpu") == 0) {
                  NCCLCHECK(ncclGetSystemId(system, node, &systemId));
                }
                const char* busId;
                NCCLCHECK(xmlGetAttr(node, "busid", &busId));
                for (int s=0; s<node->nSubs; s++) {
                  NCCLCHECK(ncclTopoAddNvLinks(node->subs[s], system, busId ? busId : parentBusId, systemId));
                }
            ncclTopoConnectCpus(*topoSystem)
                for (int n=0; n<system->nodes[CPU].count; n++) {
                    struct ncclTopoNode* cpu1 = system->nodes[CPU].nodes+n;
                    for (int p=0; p<system->nodes[CPU].count; p++) {
                      struct ncclTopoNode* cpu2 = system->nodes[CPU].nodes+p;
                      if (n == p || (NCCL_TOPO_ID_SYSTEM_ID(cpu1->id) != NCCL_TOPO_ID_SYSTEM_ID(cpu2->id))) continue;
                      float bw;
                      NCCLCHECK(ncclTopoGetInterCpuBw(cpu1, &bw));
                      NCCLCHECK(ncclTopoConnectNodes(cpu1, cpu2, LINK_SYS, bw));
                    }
                }
            ncclTopoSortSystem
                for (int n=0; n<system->nodes[CPU].count; n++) NCCLCHECK(ncclTopoSort(system->nodes[CPU].nodes+n, NULL));
                      // Shift all links to have upLink as last link
                    if (upNode) {
                        int l=0;
                        while (node->links[l].remNode != upNode) l++;
                        struct ncclTopoLink upLink;
                        memcpy(&upLink, node->links+l, sizeof(struct ncclTopoLink));
                        while (node->links[l+1].remNode) {
                          memcpy(node->links+l, node->links+l+1, sizeof(struct ncclTopoLink));
                          l++;
                        }
                        memcpy(node->links+l, &upLink, sizeof(struct ncclTopoLink));
                    }
                    // Recursively sort the PCI tree
                    for (int l=0; l<node->nlinks; l++) {
                        struct ncclTopoLink* link = node->links+l;
                        if (link->type == LINK_PCI && link->remNode != upNode) NCCLCHECK(ncclTopoSort(link->remNode, node));
                    }
```

# ncclTopoComputePaths
???

广搜：求remNode经过node到baseNode的最短路径

带宽小且跳数大，会被更新

```
cpu PATH_PHB pci PATH_PHB gpu
                 PATH_PHB nic PATH_PHB net 
             nic PATH_PHB net
    PATH_SYS cpu PATH_SYS ...

gpu PATH_NVL nvswitch PATH_NVL gpu
    PATH_PIX pci PATH_PIX nic PATH_PIX net 
                 PATH_PHB cpu PATH_PHB ...
                              PATH_SYS cpu PATH_SYS ...

net LINK_LOC nic PATH_PIX pci PATH_PIX gpu
             cpu PATH_PHB ...
                 PATH_SYS cpu PATH_SYS ...

nvswitch PATH_NVL gpu

net LINK_LOC nic PATH_PIX pci PATH_PIX gpu PATH_PIX nvswitch PATH_PIX gpu
```

类型和整数的对应关系

```
#define GPU 0
#define PCI 1
#define NVS 2
#define CPU 3 // Actually NUMA domains
#define NIC 4
#define NET 5

// We want link types and path types to match as much as possible
#define LINK_LOC 0
#define LINK_NVL 1
// Skipping 2 for PATH_NVB
#define LINK_C2C 3
#define LINK_PCI 4
// Skipping 5 for PATH_PXB
// Skipping 6 for PATH_PXN
// Skipping 7 for PATH_PHB
#define LINK_SYS 8
#define LINK_NET 9
extern const char* topoLinkTypeStr[];

// Local (myself)
#define PATH_LOC 0

// Connection traversing NVLink
#define PATH_NVL 1

// Connection through NVLink using an intermediate GPU
#define PATH_NVB 2

// Connection through C2C
#define PATH_C2C 3

// Connection traversing at most a single PCIe bridge
#define PATH_PIX 4

// Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
#define PATH_PXB 5

// Connection between a GPU and a NIC using an intermediate GPU. Used to enable rail-local, aggregated network send/recv operations.
#define PATH_PXN 6

// Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
#define PATH_PHB 7

// Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
#define PATH_SYS 8

// Connection through the network
#define PATH_NET 9

// New type of path which should precede PATH_PIX
#define PATH_PORT PATH_NVL

// Disconnected
#define PATH_DIS 10
```

GPUDirectP2P level默认PATH_PXB，因为硬件不一定支持？？？大于p2p level的经过CPU，修改带宽

gpu到网卡之间只有pci，两个gpu都是只有pci，不一定是同一网卡，且速度更快，不使用p2p；检查GPU是否支持p2p

不支持p2p也不支持shm，设为PATH_NET，待删除

gpu到net可以经过nvlink+pci，path.type=PATH_PXN，经过gpu，修改带宽。net1没有localGpu。其他gpu都可以经过和net同一pci下的gpu到达net

gdr level默认PATH_PXB，PXN在这里也当成PXB，大于的经过CPU，修改带宽

单机删除net

maxBw: gpu到gpu(单机)/net(多机)的最大带宽

totalBw：gpu最大带宽类型(nv/pci)的带宽

```
szzj-inf-sci-k8s-hzz1-h72ni7-0542:160571:160571 [0] NCCL INFO === System : maxBw 370.8 totalBw 370.8 ===
szzj-inf-sci-k8s-hzz1-h72ni7-0542:160571:160571 [0] NCCL INFO CPU/0-0 (1/1/3)
szzj-inf-sci-k8s-hzz1-h72ni7-0542:160571:160571 [0] NCCL INFO + PCI[48.0] - PCI/0-16000 (1000c0301000100b)
szzj-inf-sci-k8s-hzz1-h72ni7-0542:160571:160571 [0] NCCL INFO               + PCI[48.0] - GPU/0-18000 (0)
szzj-inf-sci-k8s-hzz1-h72ni7-0542:160571:160571 [0] NCCL INFO                             + NVL[370.8] - NVS/0-0
szzj-inf-sci-k8s-hzz1-h72ni7-0542:160571:160571 [0] NCCL INFO               + PCI[48.0] - NIC/0-19000
szzj-inf-sci-k8s-hzz1-h72ni7-0542:160571:160571 [0] NCCL INFO + PCI[48.0] - PCI/0-36000 (1000c0301000100b)
szzj-inf-sci-k8s-hzz1-h72ni7-0542:160571:160571 [0] NCCL INFO               + PCI[48.0] - NIC/0-39000
szzj-inf-sci-k8s-hzz1-h72ni7-0542:160571:160571 [0] NCCL INFO               + PCI[48.0] - GPU/0-38000 (1)
szzj-inf-sci-k8s-hzz1-h72ni7-0542:160571:160571 [0] NCCL INFO                             + NVL[370.8] - NVS/0-0
szzj-inf-sci-k8s-hzz1-h72ni7-0542:160571:160571 [0] NCCL INFO + PCI[48.0] - PCI/0-46000 (1000c0301000100b)
szzj-inf-sci-k8s-hzz1-h72ni7-0542:160571:160571 [0] NCCL INFO               + PCI[48.0] - NIC/0-48000
szzj-inf-sci-k8s-hzz1-h72ni7-0542:160571:160571 [0] NCCL INFO               + PCI[48.0] - GPU/0-49000 (2)
szzj-inf-sci-k8s-hzz1-h72ni7-0542:160571:160571 [0] NCCL INFO                             + NVL[370.8] - NVS/0-0
szzj-inf-sci-k8s-hzz1-h72ni7-0542:160571:160571 [0] NCCL INFO + PCI[48.0] - PCI/0-56000 (1000c0301000100b)
szzj-inf-sci-k8s-hzz1-h72ni7-0542:160571:160571 [0] NCCL INFO               + PCI[48.0] - NIC/0-58000
szzj-inf-sci-k8s-hzz1-h72ni7-0542:160571:160571 [0] NCCL INFO               + PCI[48.0] - GPU/0-59000 (3)
szzj-inf-sci-k8s-hzz1-h72ni7-0542:160571:160571 [0] NCCL INFO                             + NVL[370.8] - NVS/0-0
szzj-inf-sci-k8s-hzz1-h72ni7-0542:160571:160571 [0] NCCL INFO + PCI[24.0] - NIC/0-26001
szzj-inf-sci-k8s-hzz1-h72ni7-0542:160571:160571 [0] NCCL INFO + SYS[22.0] - CPU/0-1
szzj-inf-sci-k8s-hzz1-h72ni7-0542:160571:160571 [0] NCCL INFO CPU/0-1 (1/1/3)
szzj-inf-sci-k8s-hzz1-h72ni7-0542:160571:160571 [0] NCCL INFO + PCI[48.0] - PCI/0-98000 (1000c0301000100b)
szzj-inf-sci-k8s-hzz1-h72ni7-0542:160571:160571 [0] NCCL INFO               + PCI[48.0] - NIC/0-9a000
szzj-inf-sci-k8s-hzz1-h72ni7-0542:160571:160571 [0] NCCL INFO               + PCI[48.0] - GPU/0-9b000 (4)
szzj-inf-sci-k8s-hzz1-h72ni7-0542:160571:160571 [0] NCCL INFO                             + NVL[370.8] - NVS/0-0
szzj-inf-sci-k8s-hzz1-h72ni7-0542:160571:160571 [0] NCCL INFO + PCI[48.0] - PCI/0-b8000 (1000c0301000100b)
szzj-inf-sci-k8s-hzz1-h72ni7-0542:160571:160571 [0] NCCL INFO               + PCI[48.0] - NIC/0-ba000
szzj-inf-sci-k8s-hzz1-h72ni7-0542:160571:160571 [0] NCCL INFO               + PCI[48.0] - GPU/0-bb000 (5)
szzj-inf-sci-k8s-hzz1-h72ni7-0542:160571:160571 [0] NCCL INFO                             + NVL[370.8] - NVS/0-0
szzj-inf-sci-k8s-hzz1-h72ni7-0542:160571:160571 [0] NCCL INFO + PCI[48.0] - PCI/0-c8000 (1000c0301000100b)
szzj-inf-sci-k8s-hzz1-h72ni7-0542:160571:160571 [0] NCCL INFO               + PCI[48.0] - NIC/0-cb000
szzj-inf-sci-k8s-hzz1-h72ni7-0542:160571:160571 [0] NCCL INFO               + PCI[48.0] - GPU/0-ca000 (6)
szzj-inf-sci-k8s-hzz1-h72ni7-0542:160571:160571 [0] NCCL INFO                             + NVL[370.8] - NVS/0-0
szzj-inf-sci-k8s-hzz1-h72ni7-0542:160571:160571 [0] NCCL INFO + PCI[48.0] - PCI/0-d8000 (1000c0301000100b)
szzj-inf-sci-k8s-hzz1-h72ni7-0542:160571:160571 [0] NCCL INFO               + PCI[48.0] - NIC/0-db000
szzj-inf-sci-k8s-hzz1-h72ni7-0542:160571:160571 [0] NCCL INFO               + PCI[48.0] - GPU/0-da000 (7)
szzj-inf-sci-k8s-hzz1-h72ni7-0542:160571:160571 [0] NCCL INFO                             + NVL[370.8] - NVS/0-0
szzj-inf-sci-k8s-hzz1-h72ni7-0542:160571:160571 [0] NCCL INFO + SYS[22.0] - CPU/0-0
szzj-inf-sci-k8s-hzz1-h72ni7-0542:160571:160571 [0] NCCL INFO ==========================================

szzj-inf-sci-k8s-hzz1-h72ni7-0893:504950:504950 [0] NCCL INFO === System : maxBw 48.0 totalBw 370.8 ===
szzj-inf-sci-k8s-hzz1-h72ni7-0893:504950:504950 [0] NCCL INFO CPU/0-0 (1/1/3)
szzj-inf-sci-k8s-hzz1-h72ni7-0893:504950:504950 [0] NCCL INFO + PCI[48.0] - PCI/0-16000 (1000c0301000100b)
szzj-inf-sci-k8s-hzz1-h72ni7-0893:504950:504950 [0] NCCL INFO               + PCI[48.0] - GPU/0-18000 (0)
szzj-inf-sci-k8s-hzz1-h72ni7-0893:504950:504950 [0] NCCL INFO                             + NVL[370.8] - NVS/0-0
szzj-inf-sci-k8s-hzz1-h72ni7-0893:504950:504950 [0] NCCL INFO               + PCI[48.0] - NIC/0-19000
szzj-inf-sci-k8s-hzz1-h72ni7-0893:504950:504950 [0] NCCL INFO                             + NET[50.0] - NET/0-d (0/80cee8000373255c/1/50.000000)
szzj-inf-sci-k8s-hzz1-h72ni7-0893:504950:504950 [0] NCCL INFO + PCI[48.0] - PCI/0-36000 (1000c0301000100b)
szzj-inf-sci-k8s-hzz1-h72ni7-0893:504950:504950 [0] NCCL INFO               + PCI[48.0] - NIC/0-39000
szzj-inf-sci-k8s-hzz1-h72ni7-0893:504950:504950 [0] NCCL INFO                             + NET[50.0] - NET/0-c (0/50e8e8000373255c/1/50.000000)
szzj-inf-sci-k8s-hzz1-h72ni7-0893:504950:504950 [0] NCCL INFO               + PCI[48.0] - GPU/0-38000 (1)
szzj-inf-sci-k8s-hzz1-h72ni7-0893:504950:504950 [0] NCCL INFO                             + NVL[370.8] - NVS/0-0
szzj-inf-sci-k8s-hzz1-h72ni7-0893:504950:504950 [0] NCCL INFO + PCI[48.0] - PCI/0-46000 (1000c0301000100b)
szzj-inf-sci-k8s-hzz1-h72ni7-0893:504950:504950 [0] NCCL INFO               + PCI[48.0] - NIC/0-48000
szzj-inf-sci-k8s-hzz1-h72ni7-0893:504950:504950 [0] NCCL INFO                             + NET[50.0] - NET/0-b (0/40fce8000373255c/1/50.000000)
szzj-inf-sci-k8s-hzz1-h72ni7-0893:504950:504950 [0] NCCL INFO               + PCI[48.0] - GPU/0-49000 (2)
szzj-inf-sci-k8s-hzz1-h72ni7-0893:504950:504950 [0] NCCL INFO                             + NVL[370.8] - NVS/0-0
szzj-inf-sci-k8s-hzz1-h72ni7-0893:504950:504950 [0] NCCL INFO + PCI[48.0] - PCI/0-56000 (1000c0301000100b)
szzj-inf-sci-k8s-hzz1-h72ni7-0893:504950:504950 [0] NCCL INFO               + PCI[48.0] - NIC/0-58000
szzj-inf-sci-k8s-hzz1-h72ni7-0893:504950:504950 [0] NCCL INFO                             + NET[50.0] - NET/0-a (0/eae8000373255c/1/50.000000)
szzj-inf-sci-k8s-hzz1-h72ni7-0893:504950:504950 [0] NCCL INFO               + PCI[48.0] - GPU/0-59000 (3)
szzj-inf-sci-k8s-hzz1-h72ni7-0893:504950:504950 [0] NCCL INFO                             + NVL[370.8] - NVS/0-0
szzj-inf-sci-k8s-hzz1-h72ni7-0893:504950:504950 [0] NCCL INFO + PCI[24.0] - NIC/0-26001
szzj-inf-sci-k8s-hzz1-h72ni7-0893:504950:504950 [0] NCCL INFO               + NET[12.5] - NET/0-9 (0/d6c9b40003c288a0/1/12.500000)
szzj-inf-sci-k8s-hzz1-h72ni7-0893:504950:504950 [0] NCCL INFO + SYS[22.0] - CPU/0-1
szzj-inf-sci-k8s-hzz1-h72ni7-0893:504950:504950 [0] NCCL INFO CPU/0-1 (1/1/3)
szzj-inf-sci-k8s-hzz1-h72ni7-0893:504950:504950 [0] NCCL INFO + PCI[48.0] - PCI/0-98000 (1000c0301000100b)
szzj-inf-sci-k8s-hzz1-h72ni7-0893:504950:504950 [0] NCCL INFO               + PCI[48.0] - NIC/0-9a000
szzj-inf-sci-k8s-hzz1-h72ni7-0893:504950:504950 [0] NCCL INFO                             + NET[50.0] - NET/0-f (0/80e8e8000373255c/1/50.000000)
szzj-inf-sci-k8s-hzz1-h72ni7-0893:504950:504950 [0] NCCL INFO               + PCI[48.0] - GPU/0-9b000 (4)
szzj-inf-sci-k8s-hzz1-h72ni7-0893:504950:504950 [0] NCCL INFO                             + NVL[370.8] - NVS/0-0
szzj-inf-sci-k8s-hzz1-h72ni7-0893:504950:504950 [0] NCCL INFO + PCI[48.0] - PCI/0-b8000 (1000c0301000100b)
szzj-inf-sci-k8s-hzz1-h72ni7-0893:504950:504950 [0] NCCL INFO               + PCI[48.0] - NIC/0-ba000
szzj-inf-sci-k8s-hzz1-h72ni7-0893:504950:504950 [0] NCCL INFO                             + NET[50.0] - NET/0-e (0/58fce8000373255c/1/50.000000)
szzj-inf-sci-k8s-hzz1-h72ni7-0893:504950:504950 [0] NCCL INFO               + PCI[48.0] - GPU/0-bb000 (5)
szzj-inf-sci-k8s-hzz1-h72ni7-0893:504950:504950 [0] NCCL INFO                             + NVL[370.8] - NVS/0-0
szzj-inf-sci-k8s-hzz1-h72ni7-0893:504950:504950 [0] NCCL INFO + PCI[48.0] - PCI/0-c8000 (1000c0301000100b)
szzj-inf-sci-k8s-hzz1-h72ni7-0893:504950:504950 [0] NCCL INFO               + PCI[48.0] - NIC/0-cb000
szzj-inf-sci-k8s-hzz1-h72ni7-0893:504950:504950 [0] NCCL INFO                             + NET[50.0] - NET/0-11 (0/90cee8000373255c/1/50.000000)
szzj-inf-sci-k8s-hzz1-h72ni7-0893:504950:504950 [0] NCCL INFO               + PCI[48.0] - GPU/0-ca000 (6)
szzj-inf-sci-k8s-hzz1-h72ni7-0893:504950:504950 [0] NCCL INFO                             + NVL[370.8] - NVS/0-0
szzj-inf-sci-k8s-hzz1-h72ni7-0893:504950:504950 [0] NCCL INFO + PCI[48.0] - PCI/0-d8000 (1000c0301000100b)
szzj-inf-sci-k8s-hzz1-h72ni7-0893:504950:504950 [0] NCCL INFO               + PCI[48.0] - NIC/0-db000
szzj-inf-sci-k8s-hzz1-h72ni7-0893:504950:504950 [0] NCCL INFO                             + NET[50.0] - NET/0-10 (0/d8e9e8000373255c/1/50.000000)
szzj-inf-sci-k8s-hzz1-h72ni7-0893:504950:504950 [0] NCCL INFO               + PCI[48.0] - GPU/0-da000 (7)
szzj-inf-sci-k8s-hzz1-h72ni7-0893:504950:504950 [0] NCCL INFO                             + NVL[370.8] - NVS/0-0
szzj-inf-sci-k8s-hzz1-h72ni7-0893:504950:504950 [0] NCCL INFO + SYS[22.0] - CPU/0-0
```

```
struct ncclTopoNode {
  int type;
  int64_t id;
  // Type specific data
  union {
    struct {
      int dev; // NVML dev number
      int rank;
      int cudaCompCap;
      int gdrSupport;
    }gpu;
    struct {
      int dev; // Plugin dev number
      uint64_t asic;
      int port;
      float bw;
      float latency;
      int gdrSupport;
      int collSupport;
      int maxChannels;
    }net;
    struct {
      int arch;
      int vendor;
      int model;
      cpu_set_t affinity;
    }cpu;
    struct {
      uint64_t device;
    }pci;
  };
  int nlinks;
  struct ncclTopoLink links[NCCL_TOPO_MAX_LINKS];// Sort links in BW descending order
  // Pre-computed paths to GPUs and NICs
  struct ncclTopoLinkList* paths[NCCL_TOPO_NODE_TYPES];
  // Used during search
  uint64_t used;
};

struct ncclTopoLinkList {
  struct ncclTopoLink* list[NCCL_TOPO_MAX_HOPS];
  int count;
  float bw;
  int type;
};

struct ncclTopoNodeList {
  struct ncclTopoNode* list[NCCL_TOPO_MAX_NODES];
  int count;
};

initTransportsRank
    ncclTopoGetSystem
    ncclTopoComputePaths
    // Compute paths between GPUs and NICs
        for (int c=0; c<system->nodes[CPU].count; c++) {
            NCCLCHECK(ncclTopoSetPaths(system->nodes[CPU].nodes+c, system));
            // breadth-first search to set all paths to that node in the system
                struct ncclTopoNodeList nextNodeList = { { 0 }, 0 };
                nodeList.count = 1; nodeList.list[0] = baseNode;
                getPath(system, baseNode, baseNode->type, baseNode->id, &basePath))
                //baseNode到baseNode的path
                    for (int i=0; i<system->nodes[t].count; i++) {
                        if (system->nodes[t].nodes[i].id == id) {
                          *path = node->paths[t]+i;
                          return ncclSuccess;
                        }
                    }
                basePath->count = 0;
                basePath->bw = LOC_BW;
                basePath->type = PATH_LOC;
                while (nodeList.count) {
                    nextNodeList.count = 0;
                    for (int n=0; n<nodeList.count; n++) {
                      struct ncclTopoNode* node = nodeList.list[n];
                      struct ncclTopoLinkList* path;
                      NCCLCHECK(getPath(system, node, baseNode->type, baseNode->id, &path));
                      //node到baseNode的路径
                      for (int l=0; l<node->nlinks; l++) {
                        struct ncclTopoLink* link = node->links+l;
                        struct ncclTopoNode* remNode = link->remNode;
                        //node的相邻节点
                        if (remNode->paths[baseNode->type] == NULL) {
                          NCCLCHECK(ncclCalloc(remNode->paths+baseNode->type, system->nodes[baseNode->type].count));
                          for (int i=0; i<system->nodes[baseNode->type].count; i++) remNode->paths[baseNode->type][i].type = PATH_DIS;
                        }
                        struct ncclTopoLinkList* remPath;
                        NCCLCHECK(getPath(system, remNode, baseNode->type, baseNode->id, &remPath));
                        //remNode到basenode到路径
                        float bw = std::min(path->bw, link->bw);

                        // allow routing through a GPU only as 1 hop
                        if (node != baseNode && node->type == GPU &&
                            (ncclParamNvbDisable() || link->type != LINK_NVL || remNode->type != GPU || path->count > 1)) continue;
                        //NVLink 被禁用

                        if ((remPath->bw == 0 || remPath->count > path->count) && remPath->bw < bw) {
                          // Find reverse link
                          for (int l=0; l<remNode->nlinks; l++) {
                            if (remNode->links[l].remNode == node && remNode->links[l].type == link->type) {
                              remPath->list[0] = remNode->links+l;
                              break;
                            }
                          }
                          if (remPath->list[0] == NULL) {
                            WARN("Failed to find reverse path from remNode %d/%lx nlinks %d to node %d/%lx",
                                 remNode->type, remNode->id, remNode->nlinks, node->type, node->id);
                            return ncclInternalError;
                          }
                          // Copy the rest of the path
                          for (int i=0; i<path->count; i++) remPath->list[i+1] = path->list[i];
                          remPath->count = path->count + 1;
                          //pci到cpu的跳数是1
                          remPath->bw = bw;
                          // Start with path type = link type. PATH and LINK types are supposed to match.
                          // Don't consider LINK_NET as we only care about the NIC->GPU path.
                          int type = link->type == LINK_NET ? LINK_LOC : link->type;
                          // Differentiate between one and multiple PCI switches
                          if (node->type == PCI && remNode->type == PCI) type = PATH_PXB;
                          // Consider a path going through the CPU as PATH_PHB
                          if (link->type == LINK_PCI && (node->type == CPU || link->remNode->type == CPU)) type = PATH_PHB;
                          // Set 1 hop NVLink as NVB
                          if (node->type == GPU && path->type == PATH_NVL && type == PATH_NVL && remPath->count > 1) type = PATH_NVB;

                          remPath->type = std::max(path->type, type);

                          // Add to the list for the next iteration if not already in the list
                          int i;
                          for (i=0; i<nextNodeList.count; i++) if (nextNodeList.list[i] == remNode) break;
                          if (i == nextNodeList.count) nextNodeList.list[nextNodeList.count++] = remNode;
                        }
                      }
                    }
                    memcpy(&nodeList, &nextNodeList, sizeof(nodeList));
        for (int g=0; g<system->nodes[GPU].count; g++) {
            NCCLCHECK(ncclTopoSetPaths(system->nodes[GPU].nodes+g, system));
        }
        // Set direct paths to NICs.
        for (int n=0; n<system->nodes[NET].count; n++) {
            NCCLCHECK(ncclTopoSetPaths(system->nodes[NET].nodes+n, system));
        }     
        // Set direct paths to NVSwitches.
        for (int n=0; n<system->nodes[NVS].count; n++) {
            NCCLCHECK(ncclTopoSetPaths(system->nodes[NVS].nodes+n, system));
        }
        // Update path for GPUs when we don't want to / can't use GPU Direct P2P
        for (int g=0; g<system->nodes[GPU].count; g++) {
            for (int p=0; p<system->nodes[GPU].count; p++) {
              int p2p;
              ncclTopoCheckP2p(comm, system, system->nodes[GPU].nodes[p].gpu.rank,
                                         system->nodes[GPU].nodes[g].gpu.rank, &p2p, NULL, NULL));
                  //(struct ncclComm* comm, struct ncclTopoSystem* system, int rank1, int rank2,
                              int* p2p, int *read, int* intermediateRank) 
                      *p2p = 0;
                      struct ncclTopoLinkList* path = gpu1->paths[GPU]+g2;
                      // By default don't use P2P across CPU Host Bridges and further apart
                      int p2pLevel = PATH_PXB;

                      int arch, vendor, model;
                      NCCLCHECK(ncclTopoCpuType(system, &arch, &vendor, &model));
                      // Allow P2P between pairs of GPUs on AMD systems
                      if ((arch == NCCL_TOPO_CPU_ARCH_X86 && vendor == NCCL_TOPO_CPU_VENDOR_AMD) && system->nodes[GPU].count <= 2) p2pLevel = PATH_SYS;

                      // User override
                      if (ncclTopoUserP2pLevel == -1)
                        NCCLCHECK(ncclGetLevel(&ncclTopoUserP2pLevel, "NCCL_P2P_DISABLE", "NCCL_P2P_LEVEL"));//-2
                      if (ncclTopoUserP2pLevel != -2) {
                        p2pLevel = ncclTopoUserP2pLevel;
                        goto compare;
                      }

                    compare:
                      // Compute the PCI distance and compare with the p2pLevel.
                      if (path->type <= p2pLevel) *p2p = 1;

                      if (*p2p == 1) {
                        // NCCL_IGNORE_DISABLED_P2P=2 is used by unit tests that don't want to
                        // validate against NVML at all since they are pretending to be on other hw.
                        if (g1 != g2 && (comm == NULL || (info1->hostHash == comm->peerInfo[comm->rank].hostHash &&
                                                          info1->hostHash == info2->hostHash)) && ncclParamIgnoreDisabledP2p() != 2) {
                          int indexes[3] = {-1,-1,-1};
                          int verticeN = 0;
                          NCCLCHECK(ncclNvmlEnsureInitialized());

                          indexes[verticeN++] = system->nodes[GPU].nodes[g1].gpu.dev;
                          if (intermediateIndex != -1) indexes[verticeN++] = system->nodes[GPU].nodes[intermediateIndex].gpu.dev;
                          indexes[verticeN++] = system->nodes[GPU].nodes[g2].gpu.dev;

                          for (int i=1; i < verticeN; i++) {
                            nvmlGpuP2PStatus_t status;
                            status = ncclNvmlDevicePairs[indexes[i-1]][indexes[i-0]].p2pStatusRead;
                            bool good = status == NVML_P2P_STATUS_OK;
                            status = ncclNvmlDevicePairs[indexes[i-1]][indexes[i-0]].p2pStatusWrite;
                            good &= status == NVML_P2P_STATUS_OK;
                            if (!good) {
                              if (!ncclParamIgnoreDisabledP2p()) {
                                if (path->type <= PATH_NVB) {
                                  WARN("P2P is disabled between NVLINK connected GPUs %d and %d. This should not be the case given their connectivity, and is probably due to a hardware issue. If you still want to proceed, you can set NCCL_IGNORE_DISABLED_P2P=1.", indexes[i-1], indexes[i-0]);
                                  return ncclUnhandledCudaError;
                                } else if (path->type < PATH_SYS) {
                                  INFO(NCCL_INIT, "P2P is disabled between connected GPUs %d and %d. You can repress this message with NCCL_IGNORE_DISABLED_P2P=1.", indexes[i-1], indexes[i-0]);
                                }
                              }
                              *p2p = 0;
                            }
                          }
                        }
                      }
              if (p2p == 0) {
                // Divert all traffic through the CPU
                int cpu;
                NCCLCHECK(ncclGetLocalCpu(system, g, &cpu));
                NCCLCHECK(addInterStep(system, CPU, cpu, GPU, p, GPU, g));
                //Node 1 -> CPU -> Node2
                    srcNode->paths[t2][i2].type = std::max(srcNode->paths[tx][ix].type, cpuNode->paths[t2][i2].type);
                    srcNode->paths[t2][i2].bw = std::min(srcNode->paths[tx][ix].bw, cpuNode->paths[t2][i2].bw);
              }
            }
            if (comm == NULL) continue;
            // Remove GPUs we can't (or don't want to) communicate with through P2P or SHM //这里的SHM应该是多进程、线程共享GPU显存
            struct ncclPeerInfo* dstInfo = comm->peerInfo+system->nodes[GPU].nodes[g].gpu.rank;
            for (int p=0; p<system->nodes[GPU].count; p++) {
              if (p == g) continue;
              struct ncclPeerInfo* srcInfo = comm->peerInfo+system->nodes[GPU].nodes[p].gpu.rank;
              int p2p;
              NCCLCHECK(ncclTransports[TRANSPORT_P2P]->canConnect(&p2p, comm, NULL, srcInfo, dstInfo));
              if (p2p == 0) {
                int shm;
                NCCLCHECK(ncclTransports[TRANSPORT_SHM]->canConnect(&shm, comm, NULL, srcInfo, dstInfo));
                if (shm == 0) {
                  // Mark this peer as inaccessible. We'll trim it later.
                  system->nodes[GPU].nodes[p].paths[GPU][g].type = PATH_NET;
                }
              }
            }
        }
        // Update paths for NICs (no GPU Direct, PXN, ...)
        for (int n=0; n<system->nodes[NET].count; n++) {
            struct ncclTopoNode* netNode = system->nodes[NET].nodes+n;

            for (int g=0; g<system->nodes[GPU].count; g++) {
              // Check whether we can access the NIC through another NVLink-connected GPU (PXN)
              struct ncclTopoNode* gpu = system->nodes[GPU].nodes+g;
              if (ncclPxnDisable(comm) != 1) {
                int localGpuIndex;
                ncclTopoGetLocalGpu(system, netNode->id, &localGpuIndex));
                //找最近的gpu
                    ncclTopoGetLocal(system, NET, netIndex, GPU, localGpus, &localGpuCount, NULL)
                    //找最近的node
                        paths = system->nodes[type].nodes[index].paths[resultType];
                        //网卡到gpu
                        for (int i=0; i<system->nodes[resultType].count; i++) {
                            if (paths[i].bw > maxBw || (paths[i].bw == maxBw && paths[i].type < minType)) {
                              maxBw = paths[i].bw;
                              minType = paths[i].type;
                            if (paths[i].bw == maxBw && paths[i].type == minType) {
                              locals[count++] = i;
                            }
                        *localCount = count;
                    for (int c=0; c<MAXCHANNELS; c++) {
                        for (int lg=0; lg<localGpuCount; lg++) {
                          int g = localGpus[lg];
                          struct ncclTopoNode* gpu = system->nodes[GPU].nodes+g;
                          int64_t id;
                          NCCLCHECK(ncclTopoGetLocalNet(system, gpu->gpu.rank, c, &id, NULL));
                          //找最近的网卡，可能有多个，不同channel使用不同net
                              ncclTopoGetLocal(system, GPU, gpu, NET, localNets, &localNetCount, NULL))
                              //gpu到net
                              ncclTopoGetLocal(system, NET, localNets[0], GPU, localGpus, &localGpuCount, NULL)
                              //net到gpu
                              int net = system->nodes[GPU].nodes[gpu].gpu.dev;//0
                              if (isPow2(localNetCount)) net = mirrorBits(net, localNetCount);//0
                              //二进制表示，翻转
                              net += channelId%(DIVUP(localNetCount,localGpuCount));//0
                              if (id) *id = system->nodes[NET].nodes[localNets[net%localNetCount]].id;
                          if (netId == id) {
                            foundGpu = g;
                            goto exit;
                          }
                        }
                      }
                if (localGpuIndex != g && localGpuIndex != -1) {
                  // PXN = PCI + NVLink.
                  struct ncclTopoNode* peerNode = system->nodes[GPU].nodes+localGpuIndex;
                  // Only use PXN for NIC n if remote GPU p ...
                  if (peerNode->paths[NET][n].type <= PATH_PXB && // Is connected to the NIC through PCI
                      peerNode->paths[GPU][g].type <= PATH_NVL && // Is connected to us through NVLink
                      NCCL_TOPO_ID_SYSTEM_ID(peerNode->id) == NCCL_TOPO_ID_SYSTEM_ID(gpu->id) && // Is on the same node as us
                      (peerNode->paths[NET][n].bw > gpu->paths[NET][n].bw || // Has either higher BW to that NIC
                       gpu->paths[NET][n].type > PATH_PXB))                  // or avoids going through a CPU
                  // We can use that GPU as relay to communicate with that NIC.
                  // Only enabling it in the GPU->NIC direction for now to favor
                  // receiving locally and sending remotely (consistent with net.cc)
                  NCCLCHECK(addInterStep(system, GPU, localGpuIndex, GPU, g, NET, n));
                }
              }
              if (gpu->paths[NET][n].type < PATH_PHB) {
                // Update path when we dont want to / can't use GPU Direct RDMA.
                enum ncclTopoGdrMode gdr;
                NCCLCHECK(ncclTopoCheckGdr(system, system->nodes[GPU].nodes[g].gpu.rank, netNode->id, 0, &gdr));
                    int netGdrLevel = PATH_PXB;
                    int distance = gpu->paths[NET][n].type;
                    if (distance == PATH_PXN) {
                        // In case of PXN, use the intermediate GPU distance instead
                        int proxyRank;
                        NCCLCHECK(ncclTopoGetIntermediateRank(system, gpu->gpu.rank, netId, &proxyRank));
                        NCCLCHECK(ncclTopoRankToIndex(system, proxyRank, &g));
                        gpu = system->nodes[GPU].nodes+g;
                        distance = gpu->paths[NET][n].type;//nvl
                    }
                    *gdrMode = ncclTopoGdrModeDefault
                if (gdr == 0) {
                  // We cannot use GPU Direct RDMA, divert all traffic through the CPU local to the GPU
                  int localCpu;
                  NCCLCHECK(ncclGetLocalCpu(system, g, &localCpu));
                  NCCLCHECK(addInterStep(system, CPU, localCpu, NET, n, GPU, g));
                  NCCLCHECK(addInterStep(system, CPU, localCpu, GPU, g, NET, n));
                }
              }
            }
        }
    ncclTopoTrimSystem
        if (system->nodes[GPU].count == comm->nRanks) {
        //不要网卡
            for (int n=system->nodes[NET].count-1; n>=0; n--)
                NCCLCHECKGOTO(ncclTopoRemoveNode(system, NET, n), ret, fail);
                    free(delNode->paths[t]);
                    node->nlinks--;
                    node->links[l].remNode--;
                    system->nodes[type].count--;
        }
    ncclTopoComputePaths
    ncclTopoSearchInit
        int inter = system->nodes[NET].count;//0
        for (int g=0; g<system->nodes[GPU].count; g++) {
            struct ncclTopoNode* gpu = system->nodes[GPU].nodes+g;
            system->maxBw = std::max(system->maxBw, getMaxBw(system, gpu, inter ? NET : GPU));
            //gpu到gpu/net的最大带宽
              for (int i=0; i<system->nodes[type].count; i++) {
                struct ncclTopoLinkList* path = gpu->paths[type]+i;
                float bw = path->bw;
                if (path->count == 0) continue;
                maxBw = std::max(maxBw, bw);
              }
            system->totalBw = std::max(system->totalBw, getTotalBw(system, gpu));
            //gpu最大带宽
              for (int l=0; l<gpu->nlinks; l++) {
                struct ncclTopoLink* link = gpu->links+l;
                if (link->type == LINK_NVL) nvlinkBw += link->bw;
                if (link->type == LINK_PCI) pciBw = link->bw;
              }
              return std::max(pciBw, nvlinkBw);
    ncclTopoPrint
        INFO(NCCL_GRAPH, "=== System : maxBw %2.1f totalBw %2.1f ===", s->maxBw, s->totalBw);
        sprintf(line+offset, "+ %s[%2.1f] - ", topoLinkTypeStr[link->type], link->bw);
```

# 单机graph

(单机内的所有rank的graph是相同的)

graph->crossNic = 0, 多机环的首尾必须是同一个net

设定graph的path类型

从GPU之间最小path类型，不大于topo最大带宽的预定的高速带宽（per channel）开始搜索

```
 *     Intra-node
 * Ring            : GPU a -> GPU b -> .. -> GPU x -> GPU a
 * Tree            : GPU a -> GPU b -> .. -> GPU x
 *
 *     Inter-node
 * Ring            : NET n -> GPU a -> GPU b -> .. -> GPU x -> NET n (or m if crossNic)
 * Simple Tree     : NET n <-> GPU a -> GPU b -> .. -> GPU x
```

机内带宽==机间带宽

sameChannels==1：每个channel节点顺序相同

单机：

1. 按pci顺序搜索一个最大nChannels，

2. 任意起点按最大带宽顺序搜索一个最大nChannels

多机：有nvswitch，按前后rank的顺序搜索

sameChannels==0：不同channel可以有不同的节点顺序

balance tree换tree，再搜索

增大intra类型，增大inter类型再搜索

crossNic可以从0变2

缩小channel带宽再搜索

可能：nChannel翻倍，6*2，channel带宽变成1/2

增大带宽再搜索

tree在单机是链表，多机net0和rank0相互连接

```
szzj-inf-sci-k8s-hzz1-h72ni7-0542:111817:111817 [0] NCCL INFO Pattern 4, crossNic 0, nChannels 12, bw 30.000000/30.000000, type NVL/PIX, sameChannels 1
szzj-inf-sci-k8s-hzz1-h72ni7-0542:111817:111817 [0] NCCL INFO  0 : GPU/0 GPU/1 GPU/2 GPU/3 GPU/4 GPU/5 GPU/6 GPU/7
szzj-inf-sci-k8s-hzz1-h72ni7-0542:111817:111817 [0] NCCL INFO  1 : GPU/0 GPU/1 GPU/2 GPU/3 GPU/4 GPU/5 GPU/6 GPU/7
szzj-inf-sci-k8s-hzz1-h72ni7-0542:111817:111817 [0] NCCL INFO  2 : GPU/0 GPU/1 GPU/2 GPU/3 GPU/4 GPU/5 GPU/6 GPU/7
szzj-inf-sci-k8s-hzz1-h72ni7-0542:111817:111817 [0] NCCL INFO  3 : GPU/0 GPU/1 GPU/2 GPU/3 GPU/4 GPU/5 GPU/6 GPU/7
szzj-inf-sci-k8s-hzz1-h72ni7-0542:111817:111817 [0] NCCL INFO  4 : GPU/0 GPU/1 GPU/2 GPU/3 GPU/4 GPU/5 GPU/6 GPU/7
szzj-inf-sci-k8s-hzz1-h72ni7-0542:111817:111817 [0] NCCL INFO  5 : GPU/0 GPU/1 GPU/2 GPU/3 GPU/4 GPU/5 GPU/6 GPU/7
szzj-inf-sci-k8s-hzz1-h72ni7-0542:111817:111817 [0] NCCL INFO  6 : GPU/0 GPU/1 GPU/2 GPU/3 GPU/4 GPU/5 GPU/6 GPU/7
szzj-inf-sci-k8s-hzz1-h72ni7-0542:111817:111817 [0] NCCL INFO  7 : GPU/0 GPU/1 GPU/2 GPU/3 GPU/4 GPU/5 GPU/6 GPU/7
szzj-inf-sci-k8s-hzz1-h72ni7-0542:111817:111817 [0] NCCL INFO  8 : GPU/0 GPU/1 GPU/2 GPU/3 GPU/4 GPU/5 GPU/6 GPU/7
szzj-inf-sci-k8s-hzz1-h72ni7-0542:111817:111817 [0] NCCL INFO  9 : GPU/0 GPU/1 GPU/2 GPU/3 GPU/4 GPU/5 GPU/6 GPU/7
szzj-inf-sci-k8s-hzz1-h72ni7-0542:111817:111817 [0] NCCL INFO 10 : GPU/0 GPU/1 GPU/2 GPU/3 GPU/4 GPU/5 GPU/6 GPU/7
szzj-inf-sci-k8s-hzz1-h72ni7-0542:111817:111817 [0] NCCL INFO 11 : GPU/0 GPU/1 GPU/2 GPU/3 GPU/4 GPU/5 GPU/6 GPU/7
szzj-inf-sci-k8s-hzz1-h72ni7-0542:32003:32003 [3] NCCL INFO Pattern 1, crossNic 0, nChannels 12, bw 30.000000/30.000000, type NVL/PIX, sameChannels 1
szzj-inf-sci-k8s-hzz1-h72ni7-0542:32003:32003 [3] NCCL INFO  0 : GPU/0 GPU/1 GPU/2 GPU/3 GPU/4 GPU/5 GPU/6 GPU/7
szzj-inf-sci-k8s-hzz1-h72ni7-0542:32003:32003 [3] NCCL INFO  1 : GPU/0 GPU/1 GPU/2 GPU/3 GPU/4 GPU/5 GPU/6 GPU/7
szzj-inf-sci-k8s-hzz1-h72ni7-0542:32003:32003 [3] NCCL INFO  2 : GPU/0 GPU/1 GPU/2 GPU/3 GPU/4 GPU/5 GPU/6 GPU/7
szzj-inf-sci-k8s-hzz1-h72ni7-0542:32003:32003 [3] NCCL INFO  3 : GPU/0 GPU/1 GPU/2 GPU/3 GPU/4 GPU/5 GPU/6 GPU/7
szzj-inf-sci-k8s-hzz1-h72ni7-0542:32003:32003 [3] NCCL INFO  4 : GPU/0 GPU/1 GPU/2 GPU/3 GPU/4 GPU/5 GPU/6 GPU/7
szzj-inf-sci-k8s-hzz1-h72ni7-0542:32003:32003 [3] NCCL INFO  5 : GPU/0 GPU/1 GPU/2 GPU/3 GPU/4 GPU/5 GPU/6 GPU/7
szzj-inf-sci-k8s-hzz1-h72ni7-0542:32003:32003 [3] NCCL INFO  6 : GPU/0 GPU/1 GPU/2 GPU/3 GPU/4 GPU/5 GPU/6 GPU/7
szzj-inf-sci-k8s-hzz1-h72ni7-0542:32003:32003 [3] NCCL INFO  7 : GPU/0 GPU/1 GPU/2 GPU/3 GPU/4 GPU/5 GPU/6 GPU/7
szzj-inf-sci-k8s-hzz1-h72ni7-0542:32003:32003 [3] NCCL INFO  8 : GPU/0 GPU/1 GPU/2 GPU/3 GPU/4 GPU/5 GPU/6 GPU/7
szzj-inf-sci-k8s-hzz1-h72ni7-0542:32003:32003 [3] NCCL INFO  9 : GPU/0 GPU/1 GPU/2 GPU/3 GPU/4 GPU/5 GPU/6 GPU/7
szzj-inf-sci-k8s-hzz1-h72ni7-0542:32003:32003 [3] NCCL INFO 10 : GPU/0 GPU/1 GPU/2 GPU/3 GPU/4 GPU/5 GPU/6 GPU/7
szzj-inf-sci-k8s-hzz1-h72ni7-0542:32003:32003 [3] NCCL INFO 11 : GPU/0 GPU/1 GPU/2 GPU/3 GPU/4 GPU/5 GPU/6 GPU/7

szzj-inf-sci-k8s-hzz1-h72ni7-0893:859900:859900 [3] NCCL INFO Pattern 4, crossNic 2, nChannels 8, bw 48.000000/48.000000, type NVL/PIX, sameChannels 0
szzj-inf-sci-k8s-hzz1-h72ni7-0893:859900:859900 [3] NCCL INFO  0 : NET/0-d GPU/0 GPU/7 GPU/6 GPU/5 GPU/4 GPU/3 GPU/2 GPU/1 NET/0-c
szzj-inf-sci-k8s-hzz1-h72ni7-0893:859900:859900 [3] NCCL INFO  1 : NET/0-c GPU/1 GPU/2 GPU/3 GPU/4 GPU/5 GPU/6 GPU/7 GPU/0 NET/0-d
szzj-inf-sci-k8s-hzz1-h72ni7-0893:859900:859900 [3] NCCL INFO  2 : NET/0-b GPU/2 GPU/1 GPU/0 GPU/7 GPU/6 GPU/5 GPU/4 GPU/3 NET/0-a
szzj-inf-sci-k8s-hzz1-h72ni7-0893:859900:859900 [3] NCCL INFO  3 : NET/0-a GPU/3 GPU/4 GPU/5 GPU/6 GPU/7 GPU/0 GPU/1 GPU/2 NET/0-b
szzj-inf-sci-k8s-hzz1-h72ni7-0893:859900:859900 [3] NCCL INFO  4 : NET/0-f GPU/4 GPU/3 GPU/2 GPU/1 GPU/0 GPU/7 GPU/6 GPU/5 NET/0-e
szzj-inf-sci-k8s-hzz1-h72ni7-0893:859900:859900 [3] NCCL INFO  5 : NET/0-e GPU/5 GPU/6 GPU/7 GPU/0 GPU/1 GPU/2 GPU/3 GPU/4 NET/0-f
szzj-inf-sci-k8s-hzz1-h72ni7-0893:859900:859900 [3] NCCL INFO  6 : NET/0-11 GPU/6 GPU/5 GPU/4 GPU/3 GPU/2 GPU/1 GPU/0 GPU/7 NET/0-10
szzj-inf-sci-k8s-hzz1-h72ni7-0893:859900:859900 [3] NCCL INFO  7 : NET/0-10 GPU/7 GPU/0 GPU/1 GPU/2 GPU/3 GPU/4 GPU/5 GPU/6 NET/0-11
szzj-inf-sci-k8s-hzz1-h72ni7-0893:859900:859900 [3] NCCL INFO Pattern 3, crossNic 0, nChannels 8, bw 48.000000/48.000000, type NVL/PIX, sameChannels 0
szzj-inf-sci-k8s-hzz1-h72ni7-0893:859900:859900 [3] NCCL INFO  0 : NET/0-d GPU/0 GPU/1 GPU/2 GPU/3 GPU/4 GPU/5 GPU/6 GPU/7 NET/0-d
szzj-inf-sci-k8s-hzz1-h72ni7-0893:859900:859900 [3] NCCL INFO  1 : NET/0-c GPU/1 GPU/2 GPU/3 GPU/4 GPU/5 GPU/6 GPU/7 GPU/0 NET/0-c
szzj-inf-sci-k8s-hzz1-h72ni7-0893:859900:859900 [3] NCCL INFO  2 : NET/0-b GPU/2 GPU/3 GPU/4 GPU/5 GPU/6 GPU/7 GPU/0 GPU/1 NET/0-b
szzj-inf-sci-k8s-hzz1-h72ni7-0893:859900:859900 [3] NCCL INFO  3 : NET/0-a GPU/3 GPU/4 GPU/5 GPU/6 GPU/7 GPU/0 GPU/1 GPU/2 NET/0-a
szzj-inf-sci-k8s-hzz1-h72ni7-0893:859900:859900 [3] NCCL INFO  4 : NET/0-f GPU/4 GPU/5 GPU/6 GPU/7 GPU/0 GPU/1 GPU/2 GPU/3 NET/0-f
szzj-inf-sci-k8s-hzz1-h72ni7-0893:859900:859900 [3] NCCL INFO  5 : NET/0-e GPU/5 GPU/6 GPU/7 GPU/0 GPU/1 GPU/2 GPU/3 GPU/4 NET/0-e
szzj-inf-sci-k8s-hzz1-h72ni7-0893:859900:859900 [3] NCCL INFO  6 : NET/0-11 GPU/6 GPU/7 GPU/0 GPU/1 GPU/2 GPU/3 GPU/4 GPU/5 NET/0-11
szzj-inf-sci-k8s-hzz1-h72ni7-0893:859900:859900 [3] NCCL INFO  7 : NET/0-10 GPU/7 GPU/0 GPU/1 GPU/2 GPU/3 GPU/4 GPU/5 GPU/6 NET/0-10
```

```
ncclTopoSearchRec(graph, saveGraph)
//一个channel搜索的起始
    //多机
        ncclTopoSearchRecNet
        //先找net，在找gpu
            ncclTopoSelectNets//localnet或typeInter
            graph->inter[graph->nChannels*2] = net->id;
            //graph->nChannels > 0
                ncclTopoReplayGetGpu
                ncclTopoSearchTryGpu
            //graph->nChannels == 0 || graph->sameChannels == 0
            //try the most local GPUs
                ncclTopoSearchTryGpu(0,0,NET,netid,id2)
    //单机
        //graph->nChannels == 0
            //按照pci顺序选择下一个gpu
            ncclTopoSearchTryGpu(step,FORCED_ORDER_PCI,-1,-1,id2)
            //从id1到id2
                ncclTopoFollowPath(t1,id1,id2,&n2)
                //找到id1到id2之间的path，判断path类型是否满足graph类型，判断path是否有足够带宽满足graph带宽，缩减link带宽
                ncclTopoSearchRecGpu(n2,step)
                //根据step，从id2选择下一个节点
                    //step == ngpus,找到一个channel了，继续找新的channel
                        graph->nChannels++;
                        ncclTopoSearchRec(graph, saveGraph)
                        graph->nChannels--;
                        return
                    //(step == backToNet,system->nodes[NET].count) 
                        ncclTopoSelectNets
                        ncclTopoFollowPath
                        ncclTopoSearchRecGpu(step)
                        ncclTopoFollowPath
                    //step < system->nodes[GPU].count-1
                        //FORCED_ORDER_PCI
                            next[0] = step+1;
                        //FORCED_ORDER_REPLAY
                            ncclTopoReplayGetGpu
                        //0
                            ncclTopoSearchNextGpuSort
                        ncclTopoSearchTryGpu(step+1,t1,n2.id,id3)
                    //step == backToFirstRank
                    //到最后一个rank，下一个应该构成环
                        getGpuIndex
                        ncclTopoFollowPath
                        ncclTopoSearchRecGpu
                        ncclTopoFollowPath
                    //else
                        ncclTopoSearchRecGpu(system, graph, saveGraph, gpu, ngpus, -1, -1, forcedOrder, time));
                ncclTopoFollowPath
                //恢复带宽
        //else
            //重复上个channel的节点顺序
            ncclTopoReplayGetGpu(&id1)
            ncclTopoSearchTryGpu(step,FORCED_ORDER_REPLAY,-1,-1,id1)
        //if (graph->sameChannels == 0 || graph->nChannels == 0) {
            //按照带宽或跳数选择下一个gpu
            ncclTopoSearchTryGpu(step,0,-1,-1,id1)
```

```
struct ncclComm {
  ncclNet_t* ncclNet;
  void* bootstrap;
  struct ncclProxyState* proxyState;
  struct ncclPeerInfo* peerInfo;
  struct ncclTopoSystem* topo;
  struct ncclTopoGraph graphs[NCCL_NUM_ALGORITHMS];

#define NCCL_TOPO_PATTERN_BALANCED_TREE 1   // Spread NIC traffic between two GPUs (Tree parent + one child on first GPU, second child on second GPU)
#define NCCL_TOPO_PATTERN_SPLIT_TREE 2      // Spread NIC traffic between two GPUs (Tree parent on first GPU, tree children on the second GPU)
#define NCCL_TOPO_PATTERN_TREE 3            // All NIC traffic going to/from the same GPU
#define NCCL_TOPO_PATTERN_RING 4            // Ring
#define NCCL_TOPO_PATTERN_NVLS 5            // NVLS+SHARP and NVLS+Tree
#define NCCL_TOPO_PATTERN_COLLNET_DIRECT 6  // Collnet Direct

struct ncclTopoGraph {
  // Input / output
  int id; // ring : 0, tree : 1, collnet : 2, nvls : 3, collnetDirect : 4
  int pattern;
  int crossNic;
  int collNet;
  int minChannels;
  int maxChannels;
  // Output
  int nChannels;
  float bwIntra;
  float bwInter;
  float latencyInter;
  int typeIntra;
  int typeInter;
  int sameChannels;
  int nHops;
  int intra[MAXCHANNELS*NCCL_TOPO_MAX_NODES];//rank
  int64_t inter[MAXCHANNELS*2];
};

float sm90SpeedArrayIntra[] = { 60.0, 50.0, 40.0, 30.0, 24.0, 20.0, 15.0, 12.0, 11.0, 6.0, 3.0 };
float sm90SpeedArrayInter[] = { 48.0, 45.0, 42.0, 40.0, 30.0, 24.0, 22.0, 20.0, 17.5, 15.0, 12.0, 6.0, 3.0, 2.4, 1.2, 0.24, 0.12 };

initTransportsRank
    ncclTopoGetSystem
    ncclTopoComputePaths
    ncclTopoTrimSystem
    ncclTopoComputePaths
    ncclTopoSearchInit
    ncclTopoPrint
    ncclNvlsInit(comm)//no
    ringGraph->id = 0;
    ringGraph->pattern = NCCL_TOPO_PATTERN_RING;
    ringGraph->minChannels = 1;
    ringGraph->maxChannels = MAXCHANNELS/2;//32
    ncclTopoCompute(comm->topo, ringGraph)
        int crossNic = (system->nodes[NET].count > 1) &&
             (graph->pattern == NCCL_TOPO_PATTERN_RING ||
              graph->pattern == NCCL_TOPO_PATTERN_BALANCED_TREE ||
              graph->pattern == NCCL_TOPO_PATTERN_SPLIT_TREE) ? ncclParamCrossNic() : 0;//2
        graph->crossNic = crossNic == 1 ? 1 : 0;//0,环的首尾必须是同一个net?
        if (ngpus > 1) {
            NCCLCHECK(ncclTopoGetGpuMinPath(system, GPU, &minTypeIntra));//PATH_NVL
            //gpu到gpu最短路径类型
            NCCLCHECK(ncclTopoGetGpuMaxPath(system, GPU, &maxTypeIntra));//PATH_NVL
        }
        if (system->nodes[NET].count > 0) {
            NCCLCHECK(ncclTopoGetGpuMinPath(system, NET, &minTypeInter));//PATH_PIX
            //gpu到net最短路径类型
            NCCLCHECK(ncclTopoGetGpuMaxPath(system, NET, &maxTypeInter));//PATH_SYS
            maxTypeIntra = maxTypeInter;//最糟糕的路径
        }
        graph->typeIntra = minTypeIntra;
        graph->typeInter = minTypeInter;
        //从最小开始
        graph->nChannels = 0;
        int trySameChannels = graph->pattern == NCCL_TOPO_PATTERN_NVLS ? 0 : 1;
        graph->sameChannels = trySameChannels;
        NCCLCHECK(ncclTopoGetCompCap(system, &ccMin, NULL));//90
        struct ncclTopoGraph tmpGraph;
        if (system->nodes[NET].count == 0) {
            nspeeds = ccMin >= 100 ? NSPEEDSINTRA_SM100 : (ccMin >= 90 ? NSPEEDSINTRA_SM90 : NSPEEDSINTRA);
            speedArray = ccMin >= 100 ? sm100SpeedArrayIntra : (ccMin >= 90 ? sm90SpeedArrayIntra : speedArrayIntra);
        } else {
            nspeeds = ccMin >= 100 ? NSPEEDSINTER_SM100 : (ccMin >= 90 ? NSPEEDSINTER_SM90 : NSPEEDSINTER);
            speedArray = ccMin >= 100 ? sm100SpeedArrayInter : (ccMin >= 90 ? sm90SpeedArrayInter : speedArrayInter);
        }
        int pass = 1;
        int speedIndex = 0;
        while ((speedArray[speedIndex] > maxBw || speedArray[speedIndex]*graph->minChannels > totalBw) && speedIndex < nspeeds-1) speedIndex++;
        tmpGraph.bwIntra = tmpGraph.bwInter = speedArray[speedIndex];
        //从GPU之间最小path类型，预定的高速带宽（per channel）开始搜索
        search:
            tmpGraph.nChannels = 0;
            ncclTopoSearchRec(system, &tmpGraph, graph, &time)
            // struct ncclTopoGraph* graph, struct ncclTopoGraph* saveGraph, int* time)
            //一个channel搜索的起始
                ncclTopoSearchParams(system, graph->pattern, &backToNet, &backToFirstRank)
                    if (system->nodes[NET].count) {
                        if (pattern == NCCL_TOPO_PATTERN_RING) *backToNet = system->nodes[GPU].count-1;
                        else if (pattern == NCCL_TOPO_PATTERN_SPLIT_TREE) *backToNet = 1;
                        else *backToNet = 0;
                        *backToFirstRank = -1;
                    } else {
                        *backToNet = -1;
                        if (pattern == NCCL_TOPO_PATTERN_RING) *backToFirstRank = system->nodes[GPU].count-1;
                        else *backToFirstRank = -1;
                    }
                //one node ring -1,7
                //one node tree -1,-1
                //multi node ring 7,-1
                //multi node tree 0,-1
                if (system->nodes[NET].count) {
                    // Start from NET
                    ncclTopoSearchRecNet(system, graph, saveGraph, backToNet, backToFirstRank, time);
                        const int bw = graph->bwInter;
                        ncclTopoSelectNets(system, graph->typeInter, -1, nets, &netCount)
                        //localnet或typeInter
                            // First add the preferred NICs
                            for (int g=0; g<system->nodes[GPU].count; g++) {
                                if (gpu != -1 && gpu != g) continue;
                                localNetCount = 0;
                                struct ncclTopoNode* gpu = system->nodes[GPU].nodes+g;
                                for (int c = 0; c<MAXCHANNELS; c++) {
                                  int64_t netId;
                                  NCCLCHECK(ncclTopoGetLocalNet(system, gpu->gpu.rank, c, &netId, NULL));
                                  NCCLCHECK(ncclTopoIdToIndex(system, NET, netId, localNets+localNetCount));
                                  if (localNetCount > 0 && localNets[localNetCount] == localNets[0]) break;
                                  localNetCount++;
                                }
                                // Append NICs to list
                                for (int i=0; i<localNetCount; i++) {
                                  int n = localNets[i];
                                  int found = 0;
                                  while (found<netCount && nets[found] != n) found++;
                                  if (found == netCount) nets[netCount++] = n;
                                }
                            }
                            // Then add others satisfying typeInter
                            for (int t=0; t <= typeInter; t++) {
                            for (int g=0; g<system->nodes[GPU].count; g++) {
                            if (gpu != -1 && gpu != g) continue;
                            localNetCount = 0;
                            struct ncclTopoNode* gpu = system->nodes[GPU].nodes+g;
                            struct ncclTopoLinkList* paths = gpu->paths[NET];
                            for (int n=0; n<system->nodes[NET].count && n<MAXCHANNELS; n++) {
                            if (paths[n].type == t) localNets[localNetCount++] = n;
                            }
                            // Append NICs to list
                            for (int i=0; i<localNetCount; i++) {
                            int n = localNets[i];
                            int found = 0;
                            while (found<netCount && nets[found] != n) found++;
                            if (found == netCount) nets[netCount++] = n;
                            }
                            }
                            }
                            *netCountRet = netCount;//8
                        for (int i=0; i<netCount; i++) {//0, 2, 3, 4, 5, 6, 7, 8
                            int n = nets[(graph->nChannels+i)%netCount];//不同channle使用不同net
                            struct ncclTopoNode* net = system->nodes[NET].nodes+n;
                            if (net->net.bw < bw) continue;    
                            if (graph->pattern == NCCL_TOPO_PATTERN_RING && graph->crossNic == 2
                                && (graph->nChannels & 1) && net->id != graph->inter[(graph->nChannels-1)*2+1]) continue;
                            graph->inter[graph->nChannels*2] = net->id;
                            system->nodes[NET].nodes[i].net.bw -= bw;
                            if (graph->nChannels > 0) {//typeInter==PATH_PXN时，channel==1，不能replay，因为net到gpu是PHB
                                // Try to replay the last channel
                                int g;
                                NCCLCHECK(ncclTopoReplayGetGpu(system, graph, -1, &g));
                                NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, FORCED_ORDER_REPLAY, time, NET, n, g));
                            }
                            if (graph->nChannels == 0 || graph->sameChannels == 0) {//typeInter==PATH_PXN,graph->sameChannels==0可以增加channel至7
                                // Then try the most local GPUs
                                float maxBw = 0;
                                int minHops = 0xfffffff;
                                struct ncclTopoLinkList* paths = net->paths[GPU];
                                for (int g=0; g<system->nodes[GPU].count; g++) {
                                  if (paths[g].bw > maxBw) {
                                    maxBw = paths[g].bw;
                                    minHops = paths[g].count;
                                  } else if (paths[g].bw == maxBw && paths[g].count > 0 && paths[g].count < minHops) {
                                    minHops = paths[g].count;
                                  }
                                }
                                if (maxBw >= bw) {
                                  for (int i=0; i<system->nodes[GPU].count; i++) {
                                    int g = (graph->nChannels+i)%system->nodes[GPU].count;
                                    if (paths[g].bw == maxBw && paths[g].count == minHops) {
                                      NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, 0, time, NET, n, g));
                            system->nodes[NET].nodes[i].net.bw += bw;
                } else {
                    else if (graph->nChannels == 0) {
                        // Try PCI order first
                        ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, FORCED_ORDER_PCI, time, -1, -1, 0));
                        //int step, int backToNet, int backToFirstRank, int forcedOrder, int *time, int type, int index, int g)
                            const uint64_t flag = 1ULL<<(graph->nChannels);
                            struct ncclTopoNode* gpu;
                            //找到index到g的path，判断path类型是否满足graph类型，判断path是否有足够带宽满足graph带宽，缩减link带宽
                            ncclTopoFollowPath(system, graph, type, index, GPU, g, 1, &gpu));
                            // int type1, int index1, int type2, int index2, float mult, struct ncclTopoNode** node)
                            //type1=-1,index1=-1,type2=gpu,index2=0,node=gpu
                            //gpu,0,gpu,1
                                *node = system->nodes[type2].nodes+index2;
                                if (type1 == -1) return ncclSuccess;
                                struct ncclTopoNode* node1 = system->nodes[type1].nodes+index1;
                                struct ncclTopoLinkList* path = node1->paths[type2]+index2;
                                struct ncclTopoNode* node2 = system->nodes[type2].nodes+index2;
                                struct ncclTopoLinkList* revPath = node2->paths[type1]+index1;
                                *node = NULL;
                                float bw = intra ? graph->bwIntra : graph->bwInter;
                                int type = intra ? graph->typeIntra : graph->typeInter;//PATH_NVL,PATH_PIX
                                //满足path类型
                                if (mult == 1 && (path->type > type)) return ncclSuccess;
                                bw *= mult;
                                int step = 0;
                                // Check there is enough bandwidth on paths. 缩小link带宽
                                followPath(path, node1, path->count, bw, &step));
                                    float pciBw = bw;
                                    struct ncclTopoNode* node = start;
                                    for (int step=0; step<maxSteps; step++) {
                                        struct ncclTopoLink* link = path->list[step];
                                        struct ncclTopoLink* revLink = NULL;
                                        float fwBw = link->type == LINK_PCI ? pciBw : bw;
                                        float revBw = 0;
                                        if (link->bw < fwBw || (revBw && revLink->bw < revBw)) { *steps = step; return ncclSuccess; }
                                        //减链路带宽
                                        SUB_ROUND(link->bw, fwBw);
                                            (a = roundf((a-b)*1000)/1000)
                                        node = link->remNode;
                                    }
                                    *steps = maxSteps;
                                if (step < path->count) goto rewind;//增大link带宽
                                graph->nHops += mult*path->count;//经过的节点数
                                *node = system->nodes[type2].nodes+index2;
                                return 
                                rewind://graph->nChannels==6
                                  // Not enough bandwidth : rewind and exit.
                                  NCCLCHECK(followPath(path, node1, step, -bw, &step));
                            if (gpu) {//满足条件
                                gpu->used ^= flag;//gpu在channel上使用了
                                ncclTopoSearchRecGpu(system, graph, saveGraph, gpu, step, backToNet, backToFirstRank, forcedOrder, time));
                                    if (step == ngpus) {
                                        // Determine whether we found a better solution or not
                                        int copy = 0;
                                        graph->nChannels++;
                                        NCCLCHECK(ncclTopoCompareGraphs(system, graph, saveGraph, &copy));
                                        //struct ncclTopoGraph* graph, struct ncclTopoGraph* refGraph,
                                            if (graph->nChannels*graph->bwIntra > refGraph->nChannels*refGraph->bwIntra) {
                                                *copy = 1;
                                                return ncclSuccess;
                                            }
                                            if (graph->nChannels*graph->bwIntra < refGraph->nChannels*refGraph->bwIntra) return ncclSuccess;
                                            if (graph->pattern == refGraph->pattern && graph->crossNic == refGraph->crossNic && graph->nHops < refGraph->nHops) *copy = 1;
                                        if (copy) {
                                          memcpy(saveGraph, graph, sizeof(struct ncclTopoGraph));
                                          if (graph->nChannels == graph->maxChannels) *time = -1;
                                        }
                                        if (graph->nChannels < graph->maxChannels) {
                                          NCCLCHECK(ncclTopoSearchRec(system, graph, saveGraph, time));
                                        }
                                        graph->nChannels--;
                                        return ncclSuccess;
                                    }
                                    graph->intra[graph->nChannels*ngpus+step] = gpu->gpu.rank;
                                    int g = gpu - system->nodes[GPU].nodes;//index
                                    if (step == backToNet) {
                                        // first get back to NIC
                                        if (system->nodes[NET].count) {
                                            getNetIndex(system, graph->inter[graph->nChannels*2], &startNetIndex));
                                            ncclTopoSelectNets(system, graph->typeInter, g, nets, &netCount)
                                            //1；当typeInter==PATH_PXN，8
                                            for (int i=0; i<netCount; i++) {
                                                int n = nets[i];
                                                struct ncclTopoNode* net = system->nodes[NET].nodes+n;
                                                if (graph->pattern == NCCL_TOPO_PATTERN_RING && graph->crossNic == 2) {
                                                  if (graph->nChannels & 1 && net->id != graph->inter[(graph->nChannels-1)*2]) continue;
                                                } else {
                                                  if (graph->crossNic == 0 && (net->net.asic != startNet->net.asic || net->net.port != startNet->net.port)) continue;
                                                    //PATH_PXN才不会被跳过
                                                }
                                                int nextBackToNet = -1;
                                                float bwInterSave = graph->bwInter;
                                                if (graph->pattern == NCCL_TOPO_PATTERN_BALANCED_TREE) {
                                                  // Count half of the bandwidth on each of the first two GPUs
                                                  if (step == 0) nextBackToNet = 1;
                                                  else if (net->id != graph->inter[graph->nChannels*2+1]) continue;
                                                  graph->bwInter /= 2;
                                                }
                                                NCCLCHECK(ncclTopoFollowPath(system, graph, GPU, g, NET, n, 1, &net));
                                                //net->0->net, 0->1
                                                graph->bwInter = bwInterSave;
                                                if (net) {
                                                  graph->inter[graph->nChannels*2+1] = net->id;//环的首尾,13==5+8?
                                                  NCCLCHECK(ncclTopoSearchRecGpu(system, graph, saveGraph, gpu, step, nextBackToNet, backToFirstRank, forcedOrder, time));
                                                  //没有加1？？？
                                                  NCCLCHECK(ncclTopoFollowPath(system, graph, GPU, g, NET, n, -1, &net));
                                                  graph->bwInter = bwInterSave;
                                                }
                                    else if (step < system->nodes[GPU].count-1) {
                                        if (forcedOrder == FORCED_ORDER_PCI) { // Try the PCI order
                                          next[0] = step+1;
                                          count = 1;
                                        } else if (forcedOrder == FORCED_ORDER_REPLAY) { // Try last channel order
                                          NCCLCHECK(ncclTopoReplayGetGpu(system, graph, step, next));
                                          //获取step的下一个idx
                                          count = 1;
                                        }
                                        else { // Normal search
                                          ncclTopoSearchNextGpuSort(system, graph, gpu, next, &count, backToNet == -1 ? 0 : backToNet == step+1 ? 1 : -1 ));
                                          //选带宽大或跳数小的节点；有nvswitch，选前后两个rank
                                          //struct ncclTopoNode* gpu, int* next, int* countPtr, int sortNet)
                                                if (sortNet) NCCLCHECK(getNetPaths(system, graph, &netPaths));
                                                      int64_t netId = graph->inter[graph->nChannels*2];
                                                      int n;
                                                      NCCLCHECK(getNetIndex(system, netId, &n));
                                                      *netPaths=system->nodes[NET].nodes[n].paths[GPU];
                                                for (int i=0; i<count; i++) {
                                                  NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, step+1, backToNet, backToFirstRank, forcedOrder, time, GPU, g, next[i]));
                                                      paths = gpu->paths[GPU];
                                                      int start = gpu-system->nodes[GPU].nodes;
                                                      int count = 0;
                                                      for (int i=1; i<ngpus; i++) {
                                                        int g = (start+i)%ngpus;
                                                        if (paths[g].count == 0) continue; // There is no path to that GPU
                                                        if (system->nodes[GPU].nodes[g].used & flag) continue;
                                                        scores[count].g = g;
                                                        scores[count].startIndex = i;
                                                        scores[count].intraNhops = paths[g].count;
                                                        scores[count].intraBw = paths[g].bw;
                                                        if (netPaths) {
                                                          scores[count].interNhops = netPaths[g].count;
                                                          scores[count].interPciBw = gpuPciBw(system->nodes[GPU].nodes+g);
                                                          scores[count].interBw = netPaths[g].bw;
                                                        }
                                                        count++;
                                                      }
                                                }
                                                //选带宽大或跳数小的
                                                qsort(scores, count, sizeof(struct ncclGpuScore), cmpScore);
                                                        if ((d = (s2->interBw - s1->interBw))) return d;
                                                        if ((d = (s2->interPciBw - s1->interPciBw))) return d;
                                                        if ((d = (s1->interNhops - s2->interNhops))) return d;
                                                        if ((d = (s2->intraBw - s1->intraBw))) return d;
                                                        if ((d = (s1->intraNhops - s2->intraNhops))) return d;
                                                   return s1->startIndex - s2->startIndex;
                                               for (int i=0; i<count; i++) next[i] = scores[i].g;
                                               *countPtr = count;
                                               if (system->nodes[NVS].count) {
                                                // NVSwitches prefer when we talk to a limited set of peers. Try to use neighbors first.
                                                int index = gpu-system->nodes[GPU].nodes;
                                                int i;
                                                int prevGpu = (index-1+ngpus)%ngpus;//7
                                                int nextGpu = (index+1)%ngpus;//1
                                                int firstGpus[2];
                                                int firstGpuCount = 0;
                                                if (graph->pattern == NCCL_TOPO_PATTERN_RING) {
                                                  firstGpus[0] = nextGpu; firstGpus[1] = prevGpu; firstGpuCount = 2;
                                                } else if (graph->pattern == NCCL_TOPO_PATTERN_SPLIT_TREE ||
                                                    graph->pattern == NCCL_TOPO_PATTERN_BALANCED_TREE) {
                                                  firstGpus[0] = prevGpu; firstGpus[1] = nextGpu; firstGpuCount = 2;
                                               else {
                                                  firstGpus[0] = nextGpu; firstGpuCount = 1;//tree
                                                }
                                                int firstGpuRealCount = 0;
                                                for (int g=0; g<firstGpuCount; g++) {
                                                  for (i=0; i<count && next[i] != firstGpus[g]; i++);
                                                  if (i<count) {
                                                    for (; i>0; i--) next[i] = next[i-1];
                                                    next[0] = firstGpus[g];
                                                    firstGpuRealCount++;
                                                  }
                                                }
                                                *countPtr = firstGpuRealCount;
                                           for (int i=0; i<count; i++) {
                                              NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, step+1, backToNet, backToFirstRank, forcedOrder, time, GPU, g, next[i]));
                                            }
                                    } else if (step == backToFirstRank) {
                                        // Find first GPU and loop back to it
                                        int p;
                                        NCCLCHECK(getGpuIndex(system, graph->intra[graph->nChannels*ngpus], &p));
                                        //rank0在system->nodes[GPU]里的idx
                                        //没有把0放在graph->intra[graph->nChannels*ngpus+step]
                                        struct ncclTopoNode* firstGpu;
                                        NCCLCHECK(ncclTopoFollowPath(system, graph, GPU, g, GPU, p, 1, &firstGpu));
                                        //t1,id1,id2,1,&n2
                                        //形成ring
                                        if (firstGpu) {
                                          NCCLCHECK(ncclTopoSearchRecGpu(system, graph, saveGraph, firstGpu, step+1, backToNet, -1, forcedOrder, time));
                                          NCCLCHECK(ncclTopoFollowPath(system, graph, GPU, g, GPU, p, -1, &firstGpu));
                                        }
                                    }
                                    else {//step == backToNet之后就是这个
                                        // Next path
                                        NCCLCHECK(ncclTopoSearchRecGpu(system, graph, saveGraph, gpu, ngpus, -1, -1, forcedOrder, time));
                                      }
                                gpu->used ^= flag;//搜索失败，回退
                                NCCLCHECK(ncclTopoFollowPath(system, graph, type, index, GPU, g, -1, &gpu));//恢复带宽
                            }
                    else {
                      // Also try to replay previous channel
                      int g;
                      ncclTopoReplayGetGpu(system, graph, -1, &g));
                      //获取-1的下一个idx
                      // int step, int* g
                          int nextRank = graph->intra[(graph->nChannels-1)*ngpus+step+1];//0
                          for (int i=0; i<ngpus; i++) if (system->nodes[GPU].nodes[i].gpu.rank == nextRank) {
                            *g = i;
                            return ncclSuccess;
                          }
                      NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, FORCED_ORDER_REPLAY, time, -1, -1, g));
                    }
                    //回退一个channel的搜索
                    if (graph->sameChannels == 0 || graph->nChannels == 0) {
                      // Finally, try all other possibilities unless we are forced to use the same channels
                      for (int g=0; g<system->nodes[GPU].count; g++) {
                      //去掉order，选择不同的起点
                        NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, 0, time, -1, -1, g));
                      }
                    }
```

-

```
initTransportsRank
    ringGraph = &comm->graphs[NCCL_ALGO_RING];
    graphs[NCCL_NUM_ALGORITHMS] = { treeGraph, ringGraph, 
    ncclTopoGetSystem
    ncclTopoComputePaths
    ncclTopoTrimSystem
    ncclTopoComputePaths
    ncclTopoSearchInit
    ncclTopoPrint
    ncclNvlsInit(comm)//no
    ringGraph->id = 0;
    ringGraph->pattern = NCCL_TOPO_PATTERN_RING;
    ringGraph->minChannels = 1;
    ringGraph->maxChannels = MAXCHANNELS/2;
    ncclTopoCompute(comm->topo, ringGraph)
        search:
            tmpGraph.nChannels = 0;
            ncclTopoSearchRec(system, &tmpGraph, graph, &time)
            if (time == -1) goto done;
            if (graph->nChannels*graph->bwInter >= system->totalBw) goto done;
            if (pass == 1) {
                // Try having different channels (except when going through AMD CPUs)
                if (tmpGraph.sameChannels == 1 &&
                    !(cpuArch == NCCL_TOPO_CPU_ARCH_X86 && cpuVendor == NCCL_TOPO_CPU_VENDOR_AMD && tmpGraph.typeIntra == PATH_SYS)) {
                  tmpGraph.sameChannels = 0;
                  goto search;
                }
                tmpGraph.sameChannels = trySameChannels;
                // Try a simpler tree
                if (ccMin >= 90 && tmpGraph.pattern == NCCL_TOPO_PATTERN_BALANCED_TREE) {
                  tmpGraph.pattern = NCCL_TOPO_PATTERN_TREE;
                  goto search;
                }
                tmpGraph.pattern = graph->pattern;
                int maxIntra = system->nodes[NET].count > 0 ? tmpGraph.typeInter : maxTypeIntra;//多机不超过inter
                if (tmpGraph.typeIntra < maxIntra && (graph->nChannels == 0 || tmpGraph.typeIntra < graph->typeIntra)) {
                  tmpGraph.typeIntra += 1;
                  goto search;
                }
                if (system->nodes[NET].count > 0 && tmpGraph.typeInter < maxTypeInter && (graph->nChannels == 0 || tmpGraph.typeInter < graph->typeInter || tmpGraph.typeInter < PATH_PXN)) {
                  tmpGraph.typeInter += 1;
                  goto search;
                }
                tmpGraph.typeInter = minTypeInter;
                if (crossNic == 2 && tmpGraph.crossNic == 0
                    && (graph->pattern == NCCL_TOPO_PATTERN_RING || graph->pattern == NCCL_TOPO_PATTERN_BALANCED_TREE)) {
                  // Try again with crossNic if permitted
                  tmpGraph.crossNic = 2;
                  goto search;
                }
                // Decrease bw until we find a solution
                if ((speedIndex < nspeeds-1) && (graph->nChannels == 0 || (speedArray[speedIndex+1]/graph->bwInter > .49))) {
                  tmpGraph.bwInter = tmpGraph.bwIntra = speedArray[++speedIndex];
                  goto search;
                }
                //恢复tmp带宽
                speedIndex = 0;
                while (speedArray[speedIndex] > maxBw && speedIndex < nspeeds-1) speedIndex++;
                tmpGraph.bwIntra = tmpGraph.bwInter = speedArray[speedIndex];
        done:
          // We have a solution. Start from that solution and move to pass 2.
          if (pass == 1) {
            time = -1;
            NCCLCHECK(ncclTopoDupChannels(graph, ccMin, ngpus));
                if (ccMin > 80 && graph->bwIntra < 50.0 && graph->nChannels > 4) return ncclSuccess;
                int dupChannels = std::min(graph->nChannels*2, graph->maxChannels);//12
                memcpy(graph->intra+graph->nChannels*ngpus, graph->intra, (dupChannels-graph->nChannels)*ngpus*sizeof(int));
                graph->bwIntra /= DIVUP(dupChannels, graph->nChannels);
                    (((x)+(y)-1)/(y))
                    //总带宽变小了？？？
                graph->bwInter /= DIVUP(dupChannels, graph->nChannels);
                graph->nChannels = dupChannels;
            memcpy(&tmpGraph, graph, sizeof(tmpGraph));
            speedIndex = 0;
            while (speedArray[speedIndex] > graph->bwInter && speedIndex < nspeeds-1) speedIndex++;
            tmpGraph.bwIntra = tmpGraph.bwInter = speedArray[speedIndex];
            tmpGraph.minChannels = graph->nChannels;
            pass = 2;
          }
          if (pass == 2) {
            // See if we can increase bw
            if (time != 0 && speedIndex > 0) {
              if (graph->pattern == NCCL_TOPO_PATTERN_RING) {
                // increase bw for Ring
                tmpGraph.bwIntra = tmpGraph.bwInter = speedArray[--speedIndex];
                goto search;
              }
            }
            time = -1;
            memcpy(&tmpGraph, graph, sizeof(tmpGraph));
          }
    ncclTopoPrintGraph(comm->topo, ringGraph)
    treeGraph->minChannels = ringGraph->nChannels;//12，8
    treeGraph->maxChannels = ringGraph->nChannels;
    ncclTopoCompute(comm->topo, treeGraph)
    if (comm->rank == ncclParamGraphDumpFileRank()) {
        struct ncclTopoGraph* dumpGraphs[5] = { ringGraph, treeGraph, collNetDirectGraph, collNetChainGraph, nvlsGraph };
        NCCLCHECKGOTO(ncclTopoDumpGraphs(comm->topo, 5, dumpGraphs), ret, fail);
    }
```

# 多机graph合并

allgather channel信息，nChannel、带宽取最小值，路径类型取最大值

当前node的recv的pre是上个node的send，当前node的send的next是下个node的recv，构成大环，这里面不涉及net

tree是node间构成树，node内是链表，根节点左子树连node内，右子树连其他node

```
int depth = comm->nRanks/nNodes - 1 + log2i(nNodes);
-1
0
1    8
2    9    -1
3    10
4    11
5    12
6    13
7    14
     15
```

comm->nChannels=2*min(treeGraph->nChannels, ringGraph->nChannels)；2 * 12，2 * 8；每个channel既可以用ring也可以用tree，ring和tree的nChannels没有变

以自己为ring起点

```
szzj-inf-sci-k8s-hzz1-h72ni7-0893:1783896:1783896 [0] NCCL INFO Channel 00/16 :  0  7  6  5  4  3  2  1  9 10 11 12 13 14 15  8
szzj-inf-sci-k8s-hzz1-h72ni7-0893:1783896:1783896 [0] NCCL INFO Channel 01/16 :  0  8 15 14 13 12 11 10  9  1  2  3  4  5  6  7
szzj-inf-sci-k8s-hzz1-h72ni7-0893:1783896:1783896 [0] NCCL INFO Channel 02/16 :  0  7  6  5  4  3 11 12 13 14 15  8  9 10  2  1
szzj-inf-sci-k8s-hzz1-h72ni7-0893:1783896:1783896 [0] NCCL INFO Channel 03/16 :  0  1  2 10  9  8 15 14 13 12 11  3  4  5  6  7
szzj-inf-sci-k8s-hzz1-h72ni7-0893:1783896:1783896 [0] NCCL INFO Channel 04/16 :  0  7  6  5 13 14 15  8  9 10 11 12  4  3  2  1
szzj-inf-sci-k8s-hzz1-h72ni7-0893:1783896:1783896 [0] NCCL INFO Channel 05/16 :  0  1  2  3  4 12 11 10  9  8 15 14 13  5  6  7
szzj-inf-sci-k8s-hzz1-h72ni7-0893:1783896:1783896 [0] NCCL INFO Channel 06/16 :  0  7 15  8  9 10 11 12 13 14  6  5  4  3  2  1
szzj-inf-sci-k8s-hzz1-h72ni7-0893:1783896:1783896 [0] NCCL INFO Channel 07/16 :  0  1  2  3  4  5  6 14 13 12 11 10  9  8 15  7
szzj-inf-sci-k8s-hzz1-h72ni7-0893:1783896:1783896 [0] NCCL INFO Channel 08/16 :  0  7  6  5  4  3  2  1  9 10 11 12 13 14 15  8
szzj-inf-sci-k8s-hzz1-h72ni7-0893:1783896:1783896 [0] NCCL INFO Channel 09/16 :  0  8 15 14 13 12 11 10  9  1  2  3  4  5  6  7
szzj-inf-sci-k8s-hzz1-h72ni7-0893:1783896:1783896 [0] NCCL INFO Channel 10/16 :  0  7  6  5  4  3 11 12 13 14 15  8  9 10  2  1
szzj-inf-sci-k8s-hzz1-h72ni7-0893:1783896:1783896 [0] NCCL INFO Channel 11/16 :  0  1  2 10  9  8 15 14 13 12 11  3  4  5  6  7
szzj-inf-sci-k8s-hzz1-h72ni7-0893:1783896:1783896 [0] NCCL INFO Channel 12/16 :  0  7  6  5 13 14 15  8  9 10 11 12  4  3  2  1
szzj-inf-sci-k8s-hzz1-h72ni7-0893:1783896:1783896 [0] NCCL INFO Channel 13/16 :  0  1  2  3  4 12 11 10  9  8 15 14 13  5  6  7
szzj-inf-sci-k8s-hzz1-h72ni7-0893:1783896:1783896 [0] NCCL INFO Channel 14/16 :  0  7 15  8  9 10 11 12 13 14  6  5  4  3  2  1
szzj-inf-sci-k8s-hzz1-h72ni7-0893:1783896:1783896 [0] NCCL INFO Channel 15/16 :  0  1  2  3  4  5  6 14 13 12 11 10  9  8 15  7

szzj-inf-sci-k8s-hzz1-h72ni7-0893:1620032:1620032 [4] NCCL INFO Ring 00 : 5 -> 4 -> 3
szzj-inf-sci-k8s-hzz1-h72ni7-0893:1620032:1620032 [4] NCCL INFO Ring 01 : 3 -> 4 -> 5
szzj-inf-sci-k8s-hzz1-h72ni7-0893:1620032:1620032 [4] NCCL INFO Ring 02 : 5 -> 4 -> 3
szzj-inf-sci-k8s-hzz1-h72ni7-0893:1620032:1620032 [4] NCCL INFO Ring 03 : 3 -> 4 -> 5
szzj-inf-sci-k8s-hzz1-h72ni7-0893:1620032:1620032 [4] NCCL INFO Ring 04 : 12 -> 4 -> 3
szzj-inf-sci-k8s-hzz1-h72ni7-0893:1620032:1620032 [4] NCCL INFO Ring 05 : 3 -> 4 -> 12
szzj-inf-sci-k8s-hzz1-h72ni7-0893:1620032:1620032 [4] NCCL INFO Ring 06 : 5 -> 4 -> 3
szzj-inf-sci-k8s-hzz1-h72ni7-0893:1620032:1620032 [4] NCCL INFO Ring 07 : 3 -> 4 -> 5
szzj-inf-sci-k8s-hzz1-h72ni7-0893:1620032:1620032 [4] NCCL INFO Ring 08 : 5 -> 4 -> 3
szzj-inf-sci-k8s-hzz1-h72ni7-0893:1620032:1620032 [4] NCCL INFO Ring 09 : 3 -> 4 -> 5
szzj-inf-sci-k8s-hzz1-h72ni7-0893:1620032:1620032 [4] NCCL INFO Ring 10 : 5 -> 4 -> 3
szzj-inf-sci-k8s-hzz1-h72ni7-0893:1620032:1620032 [4] NCCL INFO Ring 11 : 3 -> 4 -> 5
szzj-inf-sci-k8s-hzz1-h72ni7-0893:1620032:1620032 [4] NCCL INFO Ring 12 : 12 -> 4 -> 3
szzj-inf-sci-k8s-hzz1-h72ni7-0893:1620032:1620032 [4] NCCL INFO Ring 13 : 3 -> 4 -> 12
szzj-inf-sci-k8s-hzz1-h72ni7-0893:1620032:1620032 [4] NCCL INFO Ring 14 : 5 -> 4 -> 3
szzj-inf-sci-k8s-hzz1-h72ni7-0893:1620032:1620032 [4] NCCL INFO Ring 15 : 3 -> 4 -> 5
szzj-inf-sci-k8s-hzz1-h72ni7-0893:1620032:1620032 [4] NCCL INFO Trees [0] 5/-1/-1->4->3 [1] 5/-1/-1->4->3 [2] 5/-1/-1->4->3 [3] 5/-1/-1->4->3 [4] 5/12/-1->4->-1 [5] -1/-1/-1->4->3 [6] 5/-1/-1->4->3 [7] 5/-1/-1->4->3 [8] 5/-1/-1->4->3 [9] 5/-1/-1->4->3 [10] 5/-1/-1->4->3 [11] 5/-1/-1->4->3 [12] 5/-1/-1->4->12 [13] -1/-1/-1->4->3 [14] 5/-1/-1->4->3 [15] 5/-1/-1->4->3
```

```
struct ncclComm {
  ncclNet_t* ncclNet;
  void* bootstrap;
  struct ncclProxyState* proxyState;
  struct ncclPeerInfo* peerInfo;
  struct ncclTopoSystem* topo;
  struct ncclTopoGraph graphs[NCCL_NUM_ALGORITHMS];
  struct ncclChannel channels[MAXCHANNELS];

struct ncclChannel {
  struct ncclChannelPeer** peers;
  struct ncclDevChannelPeer** devPeers;
  /* devPeer pointer array used for host side access */
  struct ncclDevChannelPeer** devPeersHostPtr;
  struct ncclRing ring;
  int* devRingUserRanks;
  struct ncclTree tree;

  struct ncclTree collnetChain;
  struct ncclDirect collnetDirect;

  struct ncclNvls nvls;

  int id; // index of this channel
  uint32_t workFifoProduced; // +1 successor of last used work fifo byte

  /* comm split sharable resources */
  struct ncclChannelPeer* collnetPeers;
  struct ncclDevChannelPeer* collnetDevPeers;
  struct ncclChannelPeer* nvlsPeers;
  struct ncclDevChannelPeer* nvlsDevPeers;
};

struct ncclRing {
  // Shortcuts for userRanks[1] and userRanks[n-1]
  int prev;
  int next;

  // Maps an internal nccl index to user-specified rank order. This is necessary
  // since we need to know how the user expects data to be ordered across
  // devices. Ordered from current device.
  int* userRanks;

  int index; // This rank's index in the ring
};

struct ncclTree {
  int depth;
  int up;
  int down[NCCL_MAX_TREE_ARITY];
};

struct allGatherInfo {
    struct graphInfo graphInfo[NCCL_NUM_ALGORITHMS];//同步
    struct ncclTopoRanks topoRanks;
    int cpuArch;
    int cpuVendor;
};

struct ncclTopoRanks {
  int ringRecv[MAXCHANNELS];
  int ringSend[MAXCHANNELS];
  int ringPrev[MAXCHANNELS];
  int ringNext[MAXCHANNELS];
  int treeToParent[MAXCHANNELS];
  int treeToChild0[MAXCHANNELS];
  int treeToChild1[MAXCHANNELS];
//首尾前后rank
//当前rank的父、子rank
//用来合并

initTransportsRank
    for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
        allGather3Data[rank].graphInfo[a].pattern = graphs[a]->pattern;
        allGather3Data[rank].graphInfo[a].nChannels = graphs[a]->nChannels;
        allGather3Data[rank].graphInfo[a].sameChannels = graphs[a]->sameChannels;
        allGather3Data[rank].graphInfo[a].bwIntra = graphs[a]->bwIntra;
        allGather3Data[rank].graphInfo[a].bwInter = graphs[a]->bwInter;
        allGather3Data[rank].graphInfo[a].typeIntra = graphs[a]->typeIntra;
        allGather3Data[rank].graphInfo[a].typeInter = graphs[a]->typeInter;
        allGather3Data[rank].graphInfo[a].crossNic = graphs[a]->crossNic;
    }
    comm->nChannels = std::min(treeGraph->nChannels, ringGraph->nChannels);//12,12;8,8
    //为什么min？？？
    ncclTopoPreset(comm, graphs, &allGather3Data[rank].topoRanks)
        int rank = comm->rank;
        int localRanks = comm->topo->nodes[GPU].count;
        for (int c=0; c<nChannels; c++) {
            int* ringIntra = graphs[NCCL_ALGO_RING]->intra+c*localRanks;
            int* treeIntra = graphs[NCCL_ALGO_TREE]->intra+c*localRanks;
            for (int i=0; i<localRanks; i++) {
                if (ringIntra[i] == rank) {
                    topoRanks->ringRecv[c] = ringIntra[0];
                    topoRanks->ringSend[c] = ringIntra[localRanks-1];
                    topoRanks->ringPrev[c] = (i == 0) ? -1 : ringIntra[i-1];
                    topoRanks->ringNext[c] = (i == localRanks-1) ? -1 : ringIntra[i+1];
                }
                //首尾前后rank
                //ring的channel等构成大环后统一处理
                if (treeIntra[i] == rank) {
                    int parentIndex = 0;
                    int child0Index = graphs[NCCL_ALGO_TREE]->pattern == NCCL_TOPO_PATTERN_TREE ? 0 : 1;//1;0
                    int child1Index = graphs[NCCL_ALGO_TREE]->pattern == NCCL_TOPO_PATTERN_SPLIT_TREE ? 1 : 0;//0;0
                    //当前rank的父、子rank
                    topoRanks->treeToParent[c] = treeIntra[parentIndex];
                    topoRanks->treeToChild0[c] = treeIntra[child0Index];
                    topoRanks->treeToChild1[c] = treeIntra[child1Index];
                    channel->tree.up         = i == 0 ? -1 : treeIntra[i-1];
                    channel->tree.down[0]    = i == localRanks-1 ? -1 : treeIntra[i+1];
                    //node内构成链表，先记录，后面只会处理node间的连接
                }
        struct ncclChannel* channel0 = comm->channels;
        struct ncclChannel* channel1 = channel0+nChannels;
        memcpy(channel1, channel0, nChannels*sizeof(struct ncclChannel));
        //nChannels没有*2？？？
    bootstrapAllGather(comm->bootstrap, allGather3Data,
    for (int r=0; r<nranks; r++) {
        int node;
        int firstRank = allGather3Data[r].topoRanks.ringRecv[0];
        for (node=0; node<comm->nNodes && nodesFirstRank[node] != firstRank; node++);
        if (node == comm->nNodes) {
          comm->nNodes++;
          nodesFirstRank[node] = firstRank;
        comm->rankToNode[r] = node;
    for (int r=0; r<comm->nRanks; r++) {
        int node = comm->rankToNode[r];
        comm->rankToLocalRank[r] = comm->nodeRanks[node].localRanks;
        comm->nodeRanks[node].localRanks++;//数量
    }
    comm->node = comm->rankToNode[rank];
    comm->localRankToRank = comm->nodeRanks[comm->node].localRankToRank;
    comm->localRank = comm->rankToLocalRank[rank];
    comm->localRanks = comm->nodeRanks[comm->node].localRanks;
    nChannelsOrig = comm->nChannels;
    for (int i=0; i<nranks; i++) {
        allTopoRanks[i] = &allGather3Data[i].topoRanks;
        // Make sure we align all ranks so that the tuning is consistent across ranks
        for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
          graphs[a]->nChannels = std::min(allGather3Data[i].graphInfo[a].nChannels, graphs[a]->nChannels);
          graphs[a]->sameChannels = std::min(allGather3Data[i].graphInfo[a].sameChannels, graphs[a]->sameChannels);
          graphs[a]->bwIntra = std::min(allGather3Data[i].graphInfo[a].bwIntra, graphs[a]->bwIntra);
          graphs[a]->bwInter = std::min(allGather3Data[i].graphInfo[a].bwInter, graphs[a]->bwInter);
          graphs[a]->typeIntra = std::max(allGather3Data[i].graphInfo[a].typeIntra, graphs[a]->typeIntra);
          graphs[a]->typeInter = std::max(allGather3Data[i].graphInfo[a].typeInter, graphs[a]->typeInter);
          graphs[a]->crossNic = std::max(allGather3Data[i].graphInfo[a].crossNic, graphs[a]->crossNic);
        }
        comm->maxTreePattern = std::max(comm->maxTreePattern, allGather3Data[i].graphInfo[NCCL_ALGO_TREE].pattern);
    }
    comm->nChannels = treeGraph->nChannels = ringGraph->nChannels = std::min(treeGraph->nChannels, ringGraph->nChannels);
    if (comm->nChannels < nChannelsOrig) {
        // We started duplicating channels during Preset(), so we need to move the
        // duplicated channels since we have removed some.
        for (int i=0; i<comm->nChannels; i++) memcpy(comm->channels+comm->nChannels+i, comm->channels+nChannelsOrig+i, sizeof(struct ncclChannel));
    }
    ncclTopoPathAllNVLink(comm->topo, &comm->isAllNvlink)//1
    ncclTopoPostset(comm, nodesFirstRank, nodesTreePatterns, allTopoRanks, rings, graphs, parent)
        if (graphs[NCCL_ALGO_RING]->crossNic == 2 && (nChannels % 2) == 0) {
            for (int r=0; r<comm->nRanks; r++) {
              if (comm->rankToNode[r] % 2 == 1) {
                // Exchange rings
                for (int c=0; c<nChannels; c+=2) {
                  exchangeValues(allTopoRanks[r]->ringRecv+c, allTopoRanks[r]->ringRecv+(c^1));
                  exchangeValues(allTopoRanks[r]->ringSend+c, allTopoRanks[r]->ringSend+(c^1));
                  exchangeValues(allTopoRanks[r]->ringPrev+c, allTopoRanks[r]->ringPrev+(c^1));
                  exchangeValues(allTopoRanks[r]->ringNext+c, allTopoRanks[r]->ringNext+(c^1));
                }
              }
            }
          }
        //如果 c是偶数，c^1会得到 c + 1（下一个奇数）。
        //如果 c是奇数，c^1会得到 c - 1（上一个偶数）。
        //奇数节点的所有rank的奇偶channel交换前后首尾，本身没有交换
        //刚好把intra原顺序反向了
        for (int c=0; c<nChannels;c++) {
            for (int n=0; n<nNodes; n++) {
              int r = firstRanks[n];
              ringRecv[c*nNodes+n] = allTopoRanks[r]->ringRecv[c];
              ringSend[c*nNodes+n] = allTopoRanks[r]->ringSend[c];
              treeToParent[c*nNodes+n] = allTopoRanks[r]->treeToParent[c];
              treeToChild0[c*nNodes+n] = allTopoRanks[r]->treeToChild0[c];
              treeToChild1[c*nNodes+n] = allTopoRanks[r]->treeToChild1[c];
              //单机内的rank，graph是一样的，这些值是一样的
            }
            for (int r=0; r<nranks; r++) {
              ringPrev[c*nranks+r] = allTopoRanks[r]->ringPrev[c];
              ringNext[c*nranks+r] = allTopoRanks[r]->ringNext[c];
            }
        }
        // Connect rings and trees. This should also duplicate the channels.
        connectRings(comm, ringRecv, ringSend, ringPrev, ringNext)
            for (int c=0; c<nChannels; c++) {
                int* recv = ringRecv+c*comm->nNodes;
                int* send = ringSend+c*comm->nNodes;
                int* prev = ringPrev+c*comm->nRanks;
                int* next = ringNext+c*comm->nRanks;
                for (int n=0; n<nNodes; n++) {
                  int recvRank = recv[n];
                  int prevSendRank = send[(n-1+nNodes)%nNodes];
                  prev[recvRank] = prevSendRank;
                  int sendRank = send[n];
                  int nextRecvRank = recv[(n+1)%nNodes];
                  next[sendRank] = nextRecvRank;
                }
                //当前节点的recv的pre是上个节点的send
                //当前节点的send的next是下个节点的recv
            }
        connectTrees(comm, treeToParent, treeToChild0, treeToChild1, treePatterns)
            int depth = comm->nRanks/nNodes - 1 + log2i(nNodes);//7;8
            int t0u, t0d0, t0d1, t0ChildType, t1u, t1d0, t1d1, t1ChildType;
            ncclGetDtree(nNodes, node, &t0u, &t0d0, &t0d1, &t0ChildType, &t1u, &t1d0, &t1d1, &t1ChildType)
            //double binary tree
                // First tree ... use a btree(binary tree)
                ncclGetBtree(nranks, rank, s0, d0_0, d0_1, parentChildType0);
                //节点之间构成tree
                ...
              for (int c=0; c<nChannels; c++) {
                 struct ncclChannel* channel0 = comm->channels+c;
                 struct ncclChannel* channel1 = channel0+nChannels;
                 ttp = treeToParent+c*comm->nNodes;
                 ttc0 = treeToChild0+c*comm->nNodes;
                 ttc1 = treeToChild1+c*comm->nNodes;
                 if (comm->rank == ttp[node]) {
                   NCCLCHECK(setTreeUp(&channel0->tree, t0ChildType == 0 ? ttc0 : ttc1, t0u));
                   NCCLCHECK(setTreeUp(&channel1->tree, t1ChildType == 0 ? ttc0 : ttc1, t1u));
                 }
                 if (comm->rank == ttc0[node]) {
                   NCCLCHECK(setTreeDown(&channel0->tree, ttp, t0d0));
                   NCCLCHECK(setTreeDown(&channel1->tree, ttp, t1d0));
                 }
                 if (comm->rank == ttc1[node]) {
                   NCCLCHECK(setTreeDown(&channel0->tree, ttp, t0d1));
                   NCCLCHECK(setTreeDown(&channel1->tree, ttp, t1d1));
                 }
                 if (comm->rank == ttp[node] ||
                     comm->rank == ttc0[node] ||
                     comm->rank == ttc1[node]) {
                   INFO(NCCL_GRAPH, "Tree %d : %d -> %d -> %d/%d/%d", c,           channel0->tree.up, comm->rank, channel0->tree.down[0], channel0->tree.down[1], channel0->tree.down[2]);
                   INFO(NCCL_GRAPH, "Tree %d : %d -> %d -> %d/%d/%d", c+nChannels, channel1->tree.up, comm->rank, channel1->tree.down[0], channel1->tree.down[1], channel1->tree.down[2]);
                 }
                 channel0->tree.depth = channel1->tree.depth = depth;
        // Duplicate ringPrev/ringNext for ncclBuildRing
        // connectRings只处理了一半
        memcpy(ringPrev+nChannels*nranks, ringPrev, nChannels*nranks*sizeof(int));
        memcpy(ringNext+nChannels*nranks, ringNext, nChannels*nranks*sizeof(int));
        // Set ring prev/next for my rank
        for (int c=0; c<nChannels; c++) {
            struct ncclChannel* channel0 = comm->channels+c;
            struct ncclChannel* channel1 = channel0+nChannels;
            channel0->ring.prev = channel1->ring.prev = ringPrev[c*nranks+comm->rank];
            channel0->ring.next = channel1->ring.next = ringNext[c*nranks+comm->rank];
        }
        nChannels = comm->nChannels = std::min(MAXCHANNELS,nChannels*2);//24;16
        nChannels = comm->nChannels = std::min(std::min(ncclMaxNchannels(), nChannels), comm->config.maxCTAs);
        nChannels = comm->nChannels = copyChannels(comm, nChannels, std::max(ncclMinNchannels(), comm->config.minCTAs), ringPrev, ringNext);//no
        comm->collChannels = comm->nChannels;
        ncclBuildRings(nChannels, rings, comm->rank, comm->nRanks, ringPrev, ringNext), ret, fail);
            snprintf(prefix, sizeof(prefix), "Channel %02d/%02d :", r, nrings);
            for (int r=0; r<nrings; r++) {
                int current = rank;
                for (int i=0; i<nranks; i++) {
                  rings[r*nranks+i] = current;
                  current = next[r*nranks+current];
                }
                //以自己为起点
    for (int c=0; c<comm->nChannels; c++) {
        struct ncclTree* tree = &comm->channels[c].tree;
        snprintf(line+strlen(line), 1023-strlen(line), " [%d] %d/%d/%d->%d->%d",
            c, tree->down[0], tree->down[1], tree->down[2], rank, tree->up);
        INFO(NCCL_GRAPH, "Ring %02d : %d -> %d -> %d", c, comm->channels[c].ring.prev, comm->rank, comm->channels[c].ring.next);
      }
    line[1023] = '\0';
    INFO(NCCL_INIT, "Trees%s", line);
```

# ncclProxyCreate

buffersize设置，comm->buffSizes[ll,ll128,simple]=={524288, 4915200, 4194304}

计算p2pnChannels，在nChannels基础上增加更多channel，32;16，初始化p2pchannel

p2p通信原语的连接什么时候建立？？？

创建两个proxy thread(ncclProxyService, ncclProxyServiceUDS)

p2pSchedule确定集合通信时GPU的顺序, 每个gpu需要和多个gpu进行发送或接收，这里确定发送gpu和接收gpu的顺序，一次可以同时接收和发送一个gpu

```
#define NCCL_PROTO_LL 0
#define NCCL_PROTO_LL128 1
#define NCCL_PROTO_SIMPLE 2

struct ncclComm {
  ncclNet_t* ncclNet;
  void* bootstrap;
  struct ncclProxyState* proxyState;
  struct ncclPeerInfo* peerInfo;
  struct ncclTopoSystem* topo;
  struct ncclTopoGraph graphs[NCCL_NUM_ALGORITHMS];
  struct ncclChannel channels[MAXCHANNELS];
  struct P2pSchedulePair { int sendRank; int recvRank; } *p2pSchedule;

struct ncclChannel {
  struct ncclChannelPeer** peers;[rank]

initTransportsRank        
    computeBuffSizes(comm)
        int64_t envs[NCCL_NUM_PROTOCOLS] = { ncclParamLlBuffSize(), ncclParamLl128BuffSize(), ncclParamBuffSize() };
        int defaults[NCCL_NUM_PROTOCOLS] = { DEFAULT_LL_BUFFSIZE, DEFAULT_LL128_BUFFSIZE, DEFAULT_BUFFSIZE };
        //524288,4915200,4194304
        for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
            comm->buffSizes[p] = envs[p] != -2 ? envs[p] : defaults[p];
        }
        if (comm->nNodes > 1) comm->p2pChunkSize = ncclParamP2pNetChunkSize();//131072
        else if (comm->isAllNvlink) comm->p2pChunkSize = ncclParamP2pNvlChunkSize();//524288
        else comm->p2pChunkSize = ncclParamP2pPciChunkSize();
        // Make sure P2P chunksize is not larger than coll chunksize.
        if (comm->p2pChunkSize * NCCL_STEPS > comm->buffSizes[NCCL_PROTO_SIMPLE]) comm->p2pChunkSize = comm->buffSizes[NCCL_PROTO_SIMPLE]/NCCL_STEPS;
        //NCCL_STEPS==8
        comm->sharedRes->tpP2pChunkSize = comm->p2pChunkSize;  
    ncclTopoComputeP2pChannels(comm)//p2p通信原语
/root/paddlejob/workspace/env_run/output/cyy/Nccl/nccl/src/graph/paths.cc
        comm->p2pnChannels = std::min(comm->nChannels, (int)ncclParamMaxP2pNChannels());
        comm->p2pnChannels = std::max(comm->p2pnChannels, (int)ncclParamMinP2pNChannels());
        int minChannels = comm->p2pnChannels;//24;16
        for (int g=0; g<comm->topo->nodes[GPU].count; g++) {//8
            for (int r=0; r<comm->nRanks; r++) {
              int nChannels;
              ncclTopoGetNchannels(comm, g, r, &nChannels));//36
                path = system->nodes[GPU].nodes[peer].paths[GPU]+g;
                if (path->type == PATH_NVL) {
                  float nvlBw = ncclTopoNVLinkBw(system->nodes[GPU].nodes[g].gpu.cudaCompCap);
                  *nChannels = 2*std::max(1, (int)(path->bw / nvlBw));
                } else {
                  *nChannels = 2;
                }
              if (nChannels >= 0) minChannels = std::min(minChannels, nChannels);
            }
        }
        comm->p2pnChannelsPerPeer = pow2Up(minChannels);//32;2
        comm->p2pnChannels = pow2Up(comm->p2pnChannels);//32;16
        for (int c=comm->nChannels; c<comm->p2pnChannels; c++) NCCLCHECK(initChannel(comm, c));//增加了8个channel
            struct ncclChannel* channel = &comm->channels[channelId];
            int nPeers = nRanks + 1 /* Collnet */ + nvlsRanks /* NVLS */;
            channel->id = channelId;
            ncclCalloc(sharedRes->peers + channelId, sharedRes->tpNRanks)//16
            for (int r = 0; r < nRanks; r++) {
              channel->peers[r] = comm->sharedRes->peers[channelId] + comm->topParentRanks[r];
              ncclAtomicRefCountIncrement(&channel->peers[r]->refCount);
            }
            ncclCudaCallocAsync(&channel->devPeers, nPeers, deviceStream)
    if (comm->sharedRes->owner == comm) {
        comm->sharedRes->tpNLocalRanks = comm->localRanks;
        comm->sharedRes->magic = comm->magic;
        comm->sharedRes->tpNChannels = comm->nChannels;
        comm->sharedRes->tpP2pNChannels = comm->p2pnChannels;
        memcpy(comm->sharedRes->tpRankToLocalRank, comm->rankToLocalRank, sizeof(int) * comm->nRanks);
    }
    //加一个tp前缀而已
    ncclTransportCheckP2pType(comm, &comm->intraNodeP2pSupport, &comm->directMode)//true,false
        for (int i = 0; i < comm->localRanks; ++i) {
          for (int j = i + 1; j < comm->localRanks; ++j) {
            int ipeer = comm->localRankToRank[i];
            int jpeer = comm->localRankToRank[j];
            struct ncclPeerInfo* ipeerInfo = &comm->peerInfo[ipeer];
            struct ncclPeerInfo* jpeerInfo = &comm->peerInfo[jpeer];
            int canConnect = 0;
            NCCLCHECK(ncclTransports[0]->canConnect(&canConnect, comm, NULL, ipeerInfo, jpeerInfo));
            if (!canConnect && supportFlag == true) {
              supportFlag = false;
            }
            if (ipeerInfo->hostHash == jpeerInfo->hostHash && ipeerInfo->pidHash == jpeerInfo->pidHash) directFlag = true;
            if (!supportFlag && directFlag) break;
          }
        }
    ncclProxyCreate(comm)
        memcpy(proxyState->buffSizes, comm->buffSizes, sizeof(comm->buffSizes));
        pthread_create(&comm->proxyState->thread, NULL, ncclProxyService, comm->proxyState)
        //pid,,func,args
        pthread_create(&comm->proxyState->threadUDS, NULL, ncclProxyServiceUDS, comm->proxyState), "pthread_create");
            struct pollfd pollfds[1];
            ncclIpcSocketGetFd(&proxyState->ipcSock, &pollfds[0].fd) 
            pollfds[0].events = POLLIN|POLLHUP;
            while (1) {
                int ret;
                do {
                  ret = poll(pollfds, 1, 500);
                } while (ret < 0 && errno == EINTR);
                if (pollfds[0].revents) {
                  // A request was seen on the UDS fd
                  proxyUDSRecvReq(proxyState, pollfds[0].fd);
                }
    comm->p2pSchedule = ncclMemoryStackAlloc<ncclComm::P2pSchedulePair>(&comm->memPermanent, nRanks);
    do { // Build p2p schedule
    //集合通信，每个gpu需要和多个gpu进行发送或接收，这里确定发送gpu和接收gpu的顺序，一次可以同时接收和发送一个gpu
        int node = comm->node;//nodeid
        int nRanks = comm->nRanks;//gpu num in all nodes
        int local = comm->localRank;
        //Flat 模式：所有 GPU 在同一节点，直接进行 P2P 通信。
        //Hierarchical 模式：GPU 分布在多个节点，先节点内通信，再节点间通信。
        bool flat = false;
        for (int node = 0; node < nNodes; node++) {
          if (nodeRanks[node].localRanks != nLocals) {
            flat = true;
            nNodes = 1; node = 0;
            nLocals = nRanks; local = rank;
            break;
          }
        }
        //视为一个node
        int nNodesPow2 = pow2Up(nNodes);
        int nLocalsPow2 = pow2Up(nLocals);
        uint32_t nodeRound = 0;
        uint32_t nodeDelta = 0;
        int round = 0;
        do {
          if (nodeDelta < nNodes) { // Filter nonsensical node deltas
            int sendNode = (node + nodeDelta) % nNodes;
            int recvNode = (node - nodeDelta + nNodes) % nNodes;
            uint32_t localRound = 0;
            uint32_t localDelta = 0;
            do {
              if (localDelta < nLocals) { // Filter nonsensical node-local deltas
                int sendLocal = (local + localDelta) % nLocals;
                int recvLocal = (local - localDelta + nLocals) % nLocals;
                comm->p2pSchedule[round].sendRank = flat ? sendLocal : nodeRanks[sendNode].localRankToRank[sendLocal];
                comm->p2pSchedule[round].recvRank = flat ? recvLocal : nodeRanks[recvNode].localRankToRank[recvLocal];
                round += 1;
              }
              localRound += 1;
              localDelta = (localDelta + localRound) & (nLocalsPow2 - 1); // Quadratic update
            } while (localRound != nLocalsPow2);
          }
          nodeRound += 1;
          //0, 1, 2, 3, 4, 5, 6, 7, 8
          nodeDelta = (nodeDelta + nodeRound) & (nNodesPow2 - 1); // Quadratic update
          //0, 1, 3, 6, 10, 15, 5, 12, 4, 13, 7, 2, 14, 11, 9, 8
        } while (nodeRound != nNodesPow2);
```

# setup&connect

p2p.setup: 连接自己的proxy，proxy申请buffer，connection和connector保存了buffer地址并返回

与recv和send rank交换ncclConnect

p2p.connect: 获取远程buffer地址

net.setup

1. 连接自己proxy

2. proxy创建ncclProxyConnection,共享内存ncclProxyOpsPool,ncclProxyProgress线程

3. 构建Connection资源，recv rank监听xgbe0;eth0，recv rank返回addr

与recv和send rank交换ncclConnect

net.connect

1. 连接recv rank的proxy listen socket/ accept send rank。每个网卡创建pd，cq，每个网卡创建两个qp，fifoMr，remSizesFifo.mrs。

2. 记录不同协议的buffer的起始位置，不同设备的buffer size。分配buffer。注册mr。send.ll在cpu，recv.ll在gpu

3. memcpy(ncclProxyConnector.transportResources,ncclProxyConnection.transportResources)

```
struct ncclComm {
  ncclNet_t* ncclNet;
  void* bootstrap;
  struct ncclProxyState* proxyState;
  struct ncclPeerInfo* peerInfo;
  struct ncclTopoSystem* topo;
  struct ncclTopoGraph graphs[NCCL_NUM_ALGORITHMS];
  struct ncclChannel channels[MAXCHANNELS];
  struct P2pSchedulePair { int sendRank; int recvRank; } *p2pSchedule;

struct ncclProxyState {
  struct ncclSocket* listenSock;//peerAddresses socket
  struct ncclIpcSocket ipcSock;//peerAddressesUDS socket
  union ncclSocketAddress* peerAddresses;//addr for ProxyService
  uint64_t *peerAddressesUDS; // 

struct ncclChannel {
  struct ncclChannelPeer** peers;[rank]
  /* devPeer pointer array used for host side access */
  struct ncclDevChannelPeer** devPeersHostPtr;//on gpu

struct ncclChannelPeer {
  struct ncclConnector send[NCCL_MAX_CONNS];//负责发送
  struct ncclConnector recv[NCCL_MAX_CONNS];

struct ncclConnector {
  struct ncclProxyConnector proxyConn;
  struct ncclTransportComm* transportComm;//函数指针
  void* transportResources;
  struct ncclConnInfo conn;

struct ncclDevChannelPeer {
  // Stripped version of ncclChannelPeer where we only keep the ncclConnInfo
  // instead of the full ncclConnector.
  struct ncclConnInfo send[NCCL_MAX_CONNS];//ncclConnector.ncclConnInfo
  struct ncclConnInfo recv[NCCL_MAX_CONNS];
};

ncclTransportRingConnect
    ncclTransportP2pConnect
        标记哪个channel需从哪个rank接收，发送到哪个rank，已经有连接的不标记
    ncclTransportP2pSetup
        selectTransport(ncclConnect)
            connector->transportComm->setup(ncclChannelPeer.ncclConnector,ncclConnect)
        bootstrapSend(ncclConnect)
        bootstrapRecv(ncclConnect)
        connector->transportComm->connect(ncclChannelPeer.ncclConnector,ncclConnect)
        cudaMemcpyAsync(ncclDevChannelPeer.ncclConnInfo,ncclChannelPeer.ncclConnector.conn, cudaMemcpyHostToDevice

ncclTransportTreeConnect
    ncclTransportP2pConnect
    ncclTransportP2pSetup
```

```
initTransportsRank        
    comm->runtimeConn = comm->cuMemSupport && ncclParamRuntimeConnect();
    if (comm->runtimeConn) {
        for (int c=0; c<comm->nChannels; c++) {
          NCCLCHECKGOTO(setupChannel(comm, c, rank, nranks, rings+c*nranks), ret, fail);
        }
    } else {
        for (int c=0; c<comm->nChannels; c++) {
            NCCLCHECKGOTO(setupChannel(comm, c, rank, nranks, rings+c*nranks), ret, fail);
                initChannel(comm, channelId))
                struct ncclRing* ring = &comm->channels[channelId].ring;
                // Find our ring-distance from rank zero and reorganize ranks to start with rank.
                int ixZero=0, ixRank=0;
                for (int i=0; i < nranks; i++) {
                    if (ringRanks[i] == 0) ixZero = i;//找到rank0在ring里的下标
                    if (ringRanks[i] == rank) ixRank = i;//0
                }
                ring->index = (ixRank-ixZero + nranks)%nranks;//rank
                for (int i=0; i<nranks; i++) {
                    ring->userRanks[i] = ringRanks[(i+ixRank)%nranks];//距离为i的gpu的rank
                }
        }
        ncclTransportRingConnect(comm), ret, fail);
            for (int c = 0; c < comm->nChannels; c++) {//p2p的8个channel不需要吗
              struct ncclChannel* channel = comm->channels + c;
              ncclTransportP2pConnect(comm, c, 1, &channel->ring.prev, 1, &channel->ring.next, 0), ret, fail);
              //int nrecv, int* peerRecv, int nsend, int* peerSend,//当前rank的前后rank，接收目标，发送目标
                  struct ncclChannel* channel = &comm->channels[channelId];
                  uint64_t mask = 1UL << channel->id;
                  for (int i=0; i<nrecv; i++) {
                    int peer = peerRecv[i];
                    if (peer == -1 || peer >= comm->nRanks || peer == comm->rank || channel->peers[peer]->recv[connIndex].connected) continue;
                    comm->connectRecv[peer] |= mask;
                  }
                  for (int i=0; i<nsend; i++) {
                    int peer = peerSend[i];
                    if (peer == -1 || peer >= comm->nRanks || peer == comm->rank || channel->peers[peer]->send[connIndex].connected) continue;
                    comm->connectSend[peer] |= mask;
                  }
                  //标记哪个通道需从哪个rank接收，发送到哪个rank
            }
            ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_RING], 0),
                struct ncclConnect** data;//char data[CONNECT_SIZE];//用来和其他rank交换信息
                int done = 0;
                for (int i=1; i<comm->nRanks; i++) {
                    int bootstrapTag = (i<<8) + (graph ? graph->id+1 : 0);
                    int recvPeer = (comm->rank - i + comm->nRanks) % comm->nRanks;
                    int sendPeer = (comm->rank + i) % comm->nRanks;
                    //到send和recv的距离相同
                    uint64_t recvMask = comm->connectRecv[recvPeer];
                    uint64_t sendMask = comm->connectSend[sendPeer];
                    int p = i-(done+1);
                    recvData[p] = data[p];
                    int sendChannels = 0, recvChannels = 0;
                    for (int c=0; c<MAXCHANNELS; c++) {
                      if (recvMask & (1UL<<c)) {
                        selectTransport<0>(comm, graph, recvData[p]+recvChannels++, c, recvPeer, connIndex, &type), ret, fail);
                        //(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclConnect* connect, 
                            connector = (type == 1) ? comm->channels[channelId].peers[peer]->send + connIndex :
                                      comm->channels[channelId].peers[peer]->recv + connIndex;
                            for (int t=0; t<NTRANSPORTS; t++) {
                                struct ncclTransport *transport = ncclTransports[t];
                                struct ncclTransportComm* transportComm = type == 1 ? &transport->send : &transport->recv;
                                int ret = 0;
                                NCCLCHECK(transport->canConnect(&ret, comm, graph, myInfo, peerInfo));
                                if (ret) {
                                  connector->transportComm = transportComm;
                                  transportComm->setup(comm, graph, myInfo, peerInfo, connect, connector, channelId, connIndex));
                                  //连接自己的proxy。让proxy申请buffer(xxProxySetup)。proxy，ncclConnector，ncclConnect都保存了buffer地址
                                  //连接自己的proxy，proxy构建共享内存，创建ProxyProgress线程，主线程也获取共享内存。proxy调用recvProxySetup:监听xgbe0;eth0，返回addr
                                  if (transportType) *transportType = t;
                                  return ncclSuccess;
                                }
                            }
                      }
                    }
                    sendData[p] = recvData[p]+recvChannels;
                    for (int c=0; c<MAXCHANNELS; c++) {
                      if (sendMask & (1UL<<c)) {
                        selectTransport<1>(comm, graph, sendData[p]+sendChannels++, c, sendPeer, connIndex, &type), ret, fail);
                            transportComm->setup
                            //p2p类似
                            //net不需要监听
                      }
                    }
                    if (recvChannels) NCCLCHECKGOTO(bootstrapSend(comm->bootstrap, recvPeer, bootstrapTag, recvData[p], sizeof(struct ncclConnect)*recvChannels), ret, fail);
                    if (sendChannels) NCCLCHECKGOTO(bootstrapSend(comm->bootstrap, sendPeer, bootstrapTag, sendData[p], sizeof(struct ncclConnect)*sendChannels), ret, fail);
                    if (sendChannels) NCCLCHECKGOTO(bootstrapRecv(comm->bootstrap, sendPeer, bootstrapTag, sendData[p], sizeof(struct ncclConnect)*sendChannels), ret, fail);
                    if (recvChannels) NCCLCHECKGOTO(bootstrapRecv(comm->bootstrap, recvPeer, bootstrapTag, recvData[p], sizeof(struct ncclConnect)*recvChannels), ret, fail);
                    //24,24
                    if (i-done == maxPeers || i == comm->nRanks-1) {
                        if (sendMask & (1UL<<c)) {
                            struct ncclConnector* conn = comm->channels[c].peers[sendPeer]->send + connIndex;
                            // This connector hasn't completed connection yet
                            if (conn->connected == 0) {
                                NCCLCHECKGOTO(conn->transportComm->connect(comm, sendData[p] + sendDataOffset, 1, comm->rank, conn), ret, fail);
                                //获取其他rank的buffer地址
                                conn->connected = 1;
                                cudaMemcpyAsync(&comm->channels[c].devPeersHostPtr[sendPeer]->send[connIndex], &conn->conn, sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice, hostStream)
                            sendDataOffset++;
                        if (recvMask & (1UL<<c)) {
                            struct ncclConnector* conn = comm->channels[c].peers[recvPeer]->recv + connIndex;
                            if (conn->connected == 0) {
                                NCCLCHECKGOTO(conn->transportComm->connect(comm, recvData[p] + recvDataOffset, 1, comm->rank, conn), ret, fail);
                                conn->connected = 1;
                                CUDACHECKGOTO(cudaMemcpyAsync(&comm->channels[c].devPeersHostPtr[recvPeer]->recv[connIndex], &conn->conn, sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice, hostStream), ret, fail);
                        }
                        recvDataOffset++;
                    done = i;
                //sync ranks here
                comm->connectRecv[recvPeer] = comm->connectSend[sendPeer] = 0UL;//清空，每次连接前需要重新mask
        // Connect Trees
        ncclTransportTreeConnect(comm), ret, fail);
            for (int c = 0; c < comm->nChannels; c++) {
              struct ncclChannel* channel = comm->channels + c;
              NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, NCCL_MAX_TREE_ARITY, channel->tree.down, 1, &channel->tree.up, 0), ret, fail);
              //子节点是接收目标，父节点是发送目标
              NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &channel->tree.up, NCCL_MAX_TREE_ARITY, channel->tree.down, 0), ret, fail);
              //父节点是接收目标，子节点是发送目标
            }
            NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_TREE], 0), ret, fail);
```

# ncclProxyService

分配buffer，创建共享内存，为rdma连接交换元数据

ncclProxyService.ncclProxyMsgInit

1. 创建ncclProxyConnection,共享内存ncclProxyOpsPool并返回

2. 创建ncclProxyProgress

```
struct ncclProxyState {
  struct ncclSocket* listenSock;//peerAddresses socket
  struct ncclIpcSocket ipcSock;//peerAddressesUDS socket
  union ncclSocketAddress* peerAddresses;//addr for ProxyService
  uint64_t *peerAddressesUDS; // 
  struct ncclSocket* peerSocks;//连接proxy的socket
  struct ncclExpectedProxyResponse* expectedResponses;//等待proxy回复的链表
  struct ncclProxyProgressState progressState;

struct ncclProxyProgressState {
  // Used by main threads to send work to progress thread
  struct ncclProxyOpsPool* opsPool;//共享内存
  ncclShmHandle_t handle;//共享内存的地址
  char opsPoolShmSuffix[6];//共享文件的名字（操作系统分配的随机名）

struct ncclProxyOpsPool {
  struct ncclProxyOp ops[MAX_OPS_PER_PEER*NCCL_MAX_LOCAL_RANKS];//nrank个独立链表
  volatile int nextOps;//-1
  volatile int nextOpsEnd;
  volatile int freeOps[NCCL_MAX_LOCAL_RANKS];//空闲op的起点
  pthread_mutex_t mutex;
  pthread_cond_t cond;
};

struct ncclProxyOp {
  int next;//-1为没有

struct ncclProxyLocalPeer {
  struct ncclSocket sock;//接收连接而创建的socket
  ncclProxyAsyncOp* asyncOps;//待处理的链表
};//一个rank

struct ncclProxyAsyncOp {
  int type;
  struct ncclProxyConnection* connection;//pools[][]
  char *reqBuff, *respBuff;
  ncclProxyAsyncOp* next;
};

struct ncclProxyConnectionPool {
  struct ncclProxyConnection** pools;//[bankid,offset]
};

struct ncclProxyConnection {
  struct ncclSocket* sock//proxy.peer.socket

struct ncclProxyInitResp {
  ncclProxyConnection* connection;
  char devShmPath[6]; // ncclProxyOpsPool'path
}

ncclProxyService(proxyState)
    ncclSocketGetFd(proxyState->listenSock, &pollfds[NCCL_MAX_PROXY_CONNECTIONS].fd) 
    while {
        ret = poll(pollfds, NCCL_MAX_PROXY_CONNECTIONS+1, asyncOpCount ? 0 : 500);
        if (pollfds[NCCL_MAX_PROXY_CONNECTIONS].revents) {
            ncclSocketInit(&peers[s].sock) 
            ncclSocketAccept(&peers[s].sock, proxyState->listenSock)
            ncclSocketGetFd(&peers[s].sock, &pollfds[s].fd)
        for (int s=0; s<maxnpeers; s++) {
            res = proxyProgressAsync(op);
            if (pollfds[s].revents & POLLIN) {
                ncclSocketRecv(peer.sock, &asyncOp->connection, sizeof(void*))//可能是null，也可能是过去建立的连接
                ncclSocketRecv(peer.sock, asyncOp->reqBuff, asyncOp->reqSize)
                asyncProxyOpEnqueue(peer, asyncOp);//尾插法
                proxyProgressAsync(asyncOp);
                    ncclProxyMsgInit
                    //创建ncclProxyConnection,共享内存ncclProxyOpsPool,ncclProxyProgress线程,返回前两者
                        proxyConnInit(op->respBuff, &op->connection);
                            *conn = pool->pools[bank]+offset;
                            (*connection)->sock = &peer->sock;//proxy.peer.socket
                            resp->connection = *connection;
                            if ((*connection)->tcomm->proxyProgress) {
                                proxyProgressInit(proxyState)
                                    struct ncclProxyProgressState* state = &proxyState->progressState;
                                    ncclShmOpen(shmPath, (void**)&pool, &state->handle)
                                    //创建共享内存(打开文件，映射内存)
                                        handle->fd = fd;
                                        handle->shmPtr = hptr;
                                        *shmPtr = hptr;
                                    state->opsPool = pool;
                                    pool->nextOps = -1;
                                    for (int r = 0; r < proxyState->tpLocalnRanks; r++) {
                                      pool->freeOps[r] = r*MAX_OPS_PER_PEER;
                                      for (int i=0; i<MAX_OPS_PER_PEER-1; i++) pool->ops[r*MAX_OPS_PER_PEER+i].next = r*MAX_OPS_PER_PEER+i+1;
                                      pool->ops[(r+1)*MAX_OPS_PER_PEER-1].next = -1;
                                    }
                                    pthread_create(&state->thread, NULL, ncclProxyProgress, proxyState)
                                strncpy(resp->devShmPath, progressState->opsPoolShmSuffix, sizeof(resp->devShmPath));//共享文件的名字（随机）
                    ncclSocketSend(op->connection->sock, op->respBuff, op->respSize)
                    //ncclProxyConnection,ncclProxyOpsPool'path
                    asyncProxyOpDequeue(peer, op);
```

```
ncclProxyMsgSetup
    op->connection->tcomm->proxySetup
ncclProxyMsgConnect
    op->connection->tcomm->proxyConnect
ncclProxyMsgSharedInit
    int nChannels = (int) *op->reqBuff;
    res = op->connection->tcomm->proxySharedInit(op->connection, proxyState, nChannels)
```

# ncclProxyProgress

用于跨机buffer之间的数据传输

1. 从共享内存取op

2. 从空闲pool里取args，地址存放在op.connection.proxyAppendPtr，op转args，args->opCount = op->opCount;(id)

3. 同一个connection的args通过nextpeer组成链表，第一个args地址存放在proxyState->progressState->active链表里

4. 调用recvProxyProgress/sendProxyProgress(active)

```
struct ncclProxyProgressState {
  // Used by main threads to send work to progress thread
  struct ncclProxyOpsPool* opsPool;//共享内存
  ncclShmHandle_t handle;//共享内存的地址
  char opsPoolShmSuffix[6];//共享文件的名字（操作系统分配的随机名）
  int nextOps;//-1,下一个要处理的
  struct ncclProxyArgs* active;//正在处理
  struct ncclProxyArgs* pool;//空闲参数池
  struct ncclProxyPool* pools;//第一个是空闲的，其他是用完的

struct ncclProxyOpsPool {
  struct ncclProxyOp ops[MAX_OPS_PER_PEER*NCCL_MAX_LOCAL_RANKS];//nrank个独立链表
  volatile int nextOps;//-1, 下一个要处理的
  volatile int nextOpsEnd;// 要处理的最后一个，尾插
  volatile int freeOps[NCCL_MAX_LOCAL_RANKS];//空闲op的起点
  pthread_mutex_t mutex;
  pthread_cond_t cond;
};
//提交给ncclProxyProgressState后next要恢复成-1

struct ncclProxyPool {
  struct ncclProxyPool *next;
  struct ncclProxyArgs elems[PROXYARGS_ALLOCATE_SIZE];
};

struct ncclProxyArgs {
  struct ncclProxySubArgs subs[NCCL_PROXY_MAX_SUBS];
  proxyProgressFunc_t progress;
  int nsubs;
  struct ncclProxyArgs* next;//在active链表里或pool链表里
  struct ncclProxyArgs* nextPeer;//同一个connection的next
  struct ncclProxyArgs** proxyAppendPtr;//存放自己的地址,意义在哪

struct ncclProxyConnection {
  struct ncclSocket* sock//proxy.peer.socket
  struct ncclProxyArgs **proxyAppendPtr;//指向尾部，头部在state->active
  void* transportResources;

ncclProxyProgress
    struct ncclProxyProgressState* state = &proxyState->progressState;
    int proxyOpAppendCounter = 0;
    /*After each progress, proxyOpAppendCounter will increase by 1 and compare with environment variable
   * ncclParamProgressAppendOpFreq(). If they are equal, we will append proxy ops. This will decrease the
   * frequency of calling ncclProxyGetPostedOps() and reduce the perf impact. */
    do {
        int idle = 1;
        ncclResult_t ret = progressOps(proxyState, state, state->active, &idle);
            op = opStart;
            while (op) {
                ncclResult_t ret = op->progress(proxyState, op);
                    //recvProxyProgress
                if (op->state == ncclProxyOpNone || ret != ncclSuccess) {
                //ncclProxyOpNone代表处理完了
                    removeOp(state, &op, &prevOp));
                    //** opPtr, struct ncclProxyArgs** prevOpPtr
                        freeOp = *opPtr;
                        next = freeOp->next;//在active链表里的下一个
                        *opPtr = next;
                        if (freeOp->nextPeer) {
                            // replace op by nextPeer
                            struct ncclProxyArgs* nextPeer = freeOp->nextPeer;
                            if (*prevOpPtr) {//上一个没有处理完
                              (*prevOpPtr)->next = nextPeer;//connection还有args
                            } else {
                              state->active = nextPeer;//头插
                            }
                            nextPeer->next = next;
                            *(prevOpPtr) = nextPeer;//先跳过nextpeer
                        } else {
                            *(freeOp->proxyAppendPtr) = NULL;//表示没有nextpeer?
                            if (*prevOpPtr) {
                              (*prevOpPtr)->next = next;
                            } else {
                              state->active = next;
                            }
                        }
                        freeOp->next = state->pool;
                        state->pool = freeOp;
                } else {
                    prevOp = op;//没有处理完，直接跳过，后面再处理
                    op = op->next;
                }
        if (idle || !state->active || (++proxyOpAppendCounter == ncclParamProgressAppendOpFreq())) {
            ret = ncclProxyGetPostedOps(proxyState, &added);
                struct ncclProxyProgressState* state = &proxyState->progressState;
                struct ncclProxyOpsPool* pool = state->opsPool;
                state->nextOps = pool->nextOps;
                pool->nextOps = pool->nextOpsEnd = -1;
                uint64_t lastOpCount = 0;
                int lastPeer = -1;
                int count = 0;
                for (int opIndex = state->nextOps; opIndex != -1;) {
                    struct ncclProxyOp* peerOp = pool->ops+opIndex;
                    if ((lastOpCount && peerOp->opCount != lastOpCount) || ((lastPeer != -1) && peer != lastPeer)) count++;
                    if (count == ncclParamProxyAppendBatchSize()+1) break;//16
                    //同一个任务的opcount是相同的，任务数量不超过16
                    lastOpCount = peerOp->opCount;//id
                    lastPeer = peer;
                    ProxyAppend(state, peerOp));
                    //用op生成connection.proxyAppendPtr
                        args = *connection->proxyAppendPtr;
                        if (args) {
                            } else {
                              struct ncclProxyArgs* prevArgs = args;
                              (allocateArgs(state, &args));
                              (ncclProxyOpToArgs(op, args, 0));
                              prevArgs->nextPeer = args;
                              *(args->proxyAppendPtr) = args;
                            }
                        } else {
                            allocateArgs(state, &args)
                            //分配connection.proxyAppendPtr
                                if (state->pool == NULL) {
                                    newElems = newPool->elems;
                                    for (int i=0; i<PROXYARGS_ALLOCATE_SIZE; i++) {
                                      if (i+1 < PROXYARGS_ALLOCATE_SIZE) newElems[i].next = newElems+i+1;
                                    }
                                    state->pool = newElems;
                                    // Save the pool memory block for later resource release
                                    newPool->next = state->pools;
                                    state->pools = newPool;
                                    //头插
                                elem = state->pool;
                                state->pool = state->pool->next;
                                //取头
                                elem->next = elem->nextPeer = NULL;
                                *argsptr = elem;
                            ncclProxyOpToArgs(op, args, 0)
                            //填充connection.proxyAppendPtr
                                //subIndex==0
                                ncclProxySubArgs* sub = args->subs+subIndex;
                                sub->sendMhandle = op->sendMhandle;
                                sub->recvMhandle = op->recvMhandle;
                                sub->sendbuff = op->sendbuff;
                                sub->recvbuff = op->recvbuff;
                                args->opCount = op->opCount;
                                args->nsubs = subIndex+1;
                                args->state = ncclProxyOpReady;
                                args->progress = op->connection->tcomm->proxyProgress;
                            if (state->active == NULL) {
                              // Create the list
                              state->active = args;
                            } else {
                              // Append element at the end of the list
                              struct ncclProxyArgs* last = state->active;
                              while (last->next) last = last->next;
                              last->next = args;
                            }
                            *(args->proxyAppendPtr) = args;//？？？
                    (*added)++;//4
                    int lastOpIndex = opIndex;
                    opIndex = peerOp->next;
                    //待释放的，头插
                    if (freeOp[peer] == -1) {
                      freeOpEnd[peer] = lastOpIndex;
                    } else {
                      peerOp->next = freeOp[peer];
                    }
                    freeOp[peer] = lastOpIndex;
                    state->nextOps = opIndex;
                for (int i = 0; i < proxyState->tpLocalnRanks; i++) {
                    if (freeOp[i] == -1) continue;
                    int newFree = freeOp[i];
                    int oldFree = pool->freeOps[i];//-1
                    pool->ops[freeOpEnd[i]].next = oldFree;
                    if (oldFree == -1) {
                      // Nothing for the main thread to consume, we can set it.
                      pool->freeOps[i] = newFree;
                    }
                }
```

-

```
struct ncclTransportComm {
  ncclResult_t (*setup)(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo*, struct ncclPeerInfo*, struct ncclConnect*, struct ncclConnector*, int channelId, int connIndex);
  ncclResult_t (*connect)(struct ncclComm* comm, struct ncclConnect*, int nranks, int rank, struct ncclConnector*);
  ncclResult_t (*free)(struct ncclConnector*);
  ncclResult_t (*proxySharedInit)(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, int nChannels);
  ncclResult_t (*proxySetup)(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done);
  ncclResult_t (*proxyConnect)(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done);
  ncclResult_t (*proxyFree)(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState);
  ncclResult_t (*proxyProgress)(struct ncclProxyState* proxyState, struct ncclProxyArgs*);
  ncclResult_t (*proxyRegister)(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done);
  ncclResult_t (*proxyDeregister)(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, int* done);
};

struct ncclTransport p2pTransport = {
  "P2P",
  p2pCanConnect,
  { p2pSendSetup, p2pSendConnect, p2pSendFree, NULL, p2pSendProxySetup, NULL, p2pSendProxyFree, NULL, p2pProxyRegister, p2pProxyDeregister },
  { p2pRecvSetup, p2pRecvConnect, p2pRecvFree, NULL, p2pRecvProxySetup, NULL, p2pRecvProxyFree, NULL, p2pProxyRegister, p2pProxyDeregister }
};

struct ncclTransport netTransport = {
  "NET",
  canConnect,
  { sendSetup, sendConnect, sendFree, proxySharedInit, sendProxySetup, sendProxyConnect, sendProxyFree, sendProxyProgress, sendProxyRegBuffer, sendProxyDeregBuffer },
  { recvSetup, recvConnect, recvFree, proxySharedInit, recvProxySetup, recvProxyConnect, recvProxyFree, recvProxyProgress, recvProxyRegBuffer, recvProxyDeregBuffer }
};  
```

# p2pTransport

setup: 连接自己的proxy，申请buffer，connection，connector两者都保存了buffer地址并返回

connect: 获取远程buffer地址

buffer包含结构体ncclxxMem(4KiB)，并对齐

(p2pCuMemProxyInfo*)(ncclProxyConnection->transportResources)->p2pBuff->size

```
struct ncclP2pBuff {
  void* directPtr;
  size_t size;
  ncclIpcDesc ipcDesc;
};

struct p2pCuMemProxyInfo {//connection->transportResources
  struct ncclP2pBuff p2pBuff;
};

struct p2pResources {//connector
  enum p2pType type;
  union {
    struct ncclSendMem* sendDevMem;
    struct ncclRecvMem* recvDevMem;
  };
  void* sendMemIpc;
  int sendMemSameProc;
  void* recvMemIpc;
  int recvMemSameProc;
  // CE memcpy support
  struct p2pShmProxyInfo proxyInfo;
  struct p2pShm* shm;
  struct p2pShm* devShm;
  ncclShmIpcDesc_t desc;
};
recv connector{recvDevMem;//local
sendMemIpc;//remote}
send connector{sendDevMem;//local
recvMemIpc;//remote}

p2pCanConnect
  ncclTopoCheckP2p(comm, comm->topo, info1->rank, info2->rank, ret, NULL, &intermediateRank))
  // Check if NET would work better
  int useNet = 0;
  NCCLCHECK(ncclTopoCheckNet(comm->topo, info1->rank, info2->rank, &useNet));
      if (useNet) {
        *ret = 0;
        return ncclSuccess;
      }
      float speed = gpu1->paths[GPU][g2].bw;
      // Now check the speed each GPU can access the network through PXB or better
      float netSpeed1 = 0, netSpeed2 = 0;
      for (int n=0; n<system->nodes[NET].count; n++) {
        struct ncclTopoLinkList* path = gpu1->paths[NET]+n;
        //gpu到网卡之间只能有pci
        if (path->type <= PATH_PXB && path->bw > netSpeed1) netSpeed1 = path->bw;
        path = gpu2->paths[NET]+n;
        if (path->type <= PATH_PXB && path->bw > netSpeed2) netSpeed2 = path->bw;
      }
      //速度更快，两个网卡之间的速度？？？
      if (netSpeed1 > speed && netSpeed2 > speed) return ncclSuccess;
      *net = 0;
  cudaDeviceCanAccessPeer

p2pRecvSetup(ncclConnector,ncclConnect)
//连接自己的proxy，申请buffer，connection、connector保存了buffer地址并返回
//struct ncclConnect* connectInfo, struct ncclConnector * recv
    struct p2pResources* resources;
    recv->transportResources = resources;
    p2pGetInfo(comm, myInfo, peerInfo, &useRead, &intermediateRank)
    //0,-1
    struct p2pConnectInfo* info = (struct p2pConnectInfo*)connectInfo;
    int recvSize = sizeof(struct ncclRecvMem);//4KiB
    for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) if (!(info->read && p == NCCL_PROTO_SIMPLE)) recvSize += comm->buffSizes[p];
    ALIGN_SIZE(recvSize, CUDA_IPC_MIN);
    if (intermediateRank == -1) {
        info->rank = myInfo->rank;
        if (P2P_SAME_PID(myInfo, peerInfo) && ncclParamP2pDirectDisable() == 0 && useMemcpy == 0) {
          resources->type = P2P_DIRECT;
        } else {
          if (ncclCuMemEnable()) {
            // cuMem API support
            resources->type = P2P_CUMEM;//
          } else {
            // Legacy CUDA IPC
            resources->type = P2P_IPC;
          }
        }
        recv->conn.flags |= info->read ? NCCL_P2P_READ : NCCL_P2P_WRITE;
    }
    req.size = recvSize;
    ncclProxyConnect(comm, TRANSPORT_P2P, 0, info->rank, &recv->proxyConn)
      ncclProxyCallBlocking(comm, proxyConn, ncclProxyMsgInit, &req, sizeof(req), &resp, sizeof(resp))
      //创建ncclProxyConnection,返回
      proxyConn->connection = resp.connection;//pool.connection
      proxyConn->initialized = true;
    ncclProxyCallBlocking(comm, &recv->proxyConn, ncclProxyMsgSetup, &req, sizeof(struct ncclP2pRequest), &info->p2pBuff, sizeof(struct ncclP2pBuff))
    //申请显存，保存在connection资源并返回
    p2pMap(comm, &recv->proxyConn, myInfo, comm->peerInfo+info->rank, &info->p2pBuff, (void**)&resources->recvDevMem, &resources->recvMemIpc));
    保存在connector资源
      if (P2P_SAME_PID(myInfo, peerInfo)) {
      //同一个进程不需要handle
          *devMem = p2pBuff->directPtr;
          *ipcPtr = NULL;
      else {
          // Different PID
          ncclP2pImportShareableBuffer(comm, peerInfo->rank, p2pBuff->size, &p2pBuff->ipcDesc, devMem));
              ncclCuDesc *cuDesc = &ipcDesc->cuDesc;
              cuMemGetAllocationGranularity(&granularity, 
              ALIGN_SIZE(size, granularity);
              ncclProxyClientGetFdBlocking(comm, peer, &cuDesc->data, &fd))
              //Send cuMem handle to remote for conversion to an fd
              cuMemImportFromShareableHandle(&handle, (void *)(uintptr_t)fd, type)
              //fd转handle
              cuMemMap(dptr, size, /* offset */ 0, handle, /* flags */ 0))
              *devMemPtr = (void *)dptr;
          *ipcPtr = *devMem;

p2pRecvProxySetup(ncclConnect,ncclProxyConnection)
//申请显存，保存在connection资源并返回
    int size = req->size;
    struct ncclP2pBuff* p2pBuff = (struct ncclP2pBuff*)respBuff;
    ncclP2pAllocateShareableBuffer(size, req->refcount, &p2pBuff->ipcDesc, &p2pBuff->directPtr)
    p2pBuff->size = size;
    if (ncclCuMemEnable()) {
        memcpy(&proxyInfo->p2pBuff, p2pBuff, sizeof(*p2pBuff));
        connection->transportResources = proxyInfo;   

p2pSendSetup(ncclConnect,ncclProxyConnection)
//获取远程buffer地址
    int sendSize = sizeof(struct ncclSendMem);
    // For P2P Read the SIMPLE buffer is tagged on the end of the ncclSendMem structure
    if (info->read) sendSize += comm->buffSizes[NCCL_PROTO_SIMPLE];//只有read才创建buffer
    ALIGN_SIZE(sendSize, CUDA_IPC_MIN);
    req.size = sendSize;
    ncclProxyConnect(comm, TRANSPORT_P2P, 1, info->rank, &send->proxyConn)
    NCCLCHECK(ncclProxyCallBlocking(comm, &send->proxyConn, ncclProxyMsgSetup, &req, sizeof(struct ncclP2pRequest), &info->p2pBuff, sizeof(struct ncclP2pBuff)));
    NCCLCHECK(p2pMap(comm, &send->proxyConn, myInfo, comm->peerInfo+info->rank, &info->p2pBuff, (void**)&resources->sendDevMem, &resources->sendMemIpc));

p2pSendConnect(ncclConnector,ncclConnect)
    resources = (struct p2pResources*)send->transportResources;
    struct p2pConnectInfo* info = (struct p2pConnectInfo*)connectInfo;//send rank buffer
    p2pMap(comm, &send->proxyConn, comm->peerInfo+rank, comm->peerInfo+info->rank, &info->p2pBuff, (void**)&remDevMem, &resources->recvMemIpc)
    //获取send rank的buffer地址
    resources->recvMemSameProc = P2P_SAME_PID((comm->peerInfo + rank), (comm->peerInfo + info->rank));
    char* buff = (char*)(remDevMem+1);
    for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
        if (info->read && p == NCCL_PROTO_SIMPLE) {
          /* For P2P Read the SIMPLE buffer is local (ncclSendMem) */
          if (resources->sendDevMem == NULL) return ncclInternalError; // We should not use read + memcpy
          send->conn.buffs[p] = (char*)(resources->sendDevMem+1);//本地
        } else {
          send->conn.buffs[p] = buff;
          buff += comm->buffSizes[p];
        }
    }
    send->conn.stepSize = comm->buffSizes[NCCL_PROTO_SIMPLE]/NCCL_STEPS;
```

# netTransport

setup

1. 连接自己proxy

2. proxy创建ncclProxyConnection,共享内存ncclProxyOpsPool,ncclProxyProgress线程

3. 构建Connection资源，recv监听xgbe0;eth0，recv返回addr

```
struct ncclProxyState {
  struct ncclSocket* listenSock;//peerProxyAddress socket，proxy监听
  struct ncclIpcSocket ipcSock;//peerProxyAddressUDS socket
  union ncclSocketAddress* peerAddresses;//peerProxyAddresses，proxy ip用于连接
  uint64_t *peerAddressesUDS; // cuMem API support (UDS)//peerProxyAddressesUDS
  struct ncclSocket* peerSocks;//连接proxy的socket
  struct ncclExpectedProxyResponse* expectedResponses;//等待proxy回复的链表
  struct ncclProxyProgressState progressState;
  struct ncclProxyOps* proxyOps;//[localrank],共享ProxyProgressState.ncclProxyOpsPool

struct ncclProxyProgressState {
  // Used by main threads to send work to progress thread
  struct ncclProxyOpsPool* opsPool;//共享内存
  ncclShmHandle_t handle;//共享内存的地址
  char opsPoolShmSuffix[6];//共享文件的名字（操作系统分配的随机名）

struct ncclProxyOps {
  ncclProxyOpsPool* pool;//main thread 共享ProxyProgress.ncclProxyOpsPool
  ncclShmHandle_t handle;//共享内存的地址
  int count;//0，要处理的数量
  int freeOp;//-1，下一个空闲
  int nextOps;//-1，下一个要处理的
  int nextOpsEnd;//-1，要处理的最后一个，尾插
};
//每次唤醒完就恢复初始值

struct ncclProxyOpsPool {
  struct ncclProxyOp ops[MAX_OPS_PER_PEER*NCCL_MAX_LOCAL_RANKS];//nrank个独立链表
  volatile int nextOps;//-1, 下一个要处理的
  volatile int nextOpsEnd;// 要处理的最后一个，尾插
  volatile int freeOps[NCCL_MAX_LOCAL_RANKS];//空闲op的起点
  pthread_mutex_t mutex;
  pthread_cond_t cond;
};

struct ncclConnector {
  struct ncclProxyConnector proxyConn;//proxy相关
  struct ncclTransportComm* transportComm;//函数指针
  void* transportResources;
  struct ncclConnInfo conn;

struct ncclProxyConnector {
  struct ncclProxyConnection* connection;//pools[][]
  ncclResult_t (*proxyProgress)(struct ncclProxyState* proxyState, struct ncclProxyArgs*); // Copied from transport if necessary

struct ncclProxyInitResp {
  ncclProxyConnection* connection;
  char devShmPath[6]; // ncclProxyOpsPool'path
};

struct ncclProxyConnection {
  struct ncclSocket* sock//proxy.peer.socket
  void* transportResources;//recvNetResources

struct recvNetResources {
  struct connectMap map;
  void* netListenComm;//xgbe0;eth0
  void* netRecvComm;

struct ncclIbListenComm {
  int dev;
  struct ncclSocket sock;//listen on xgbe0;eth0
  struct ncclIbCommStage stage;
};

struct ncclIbHandle {//ncclConnect
  union ncclSocketAddress connectAddr; //xgbe0;eth0 地址
  struct ncclIbCommStage stage;
};

recvSetup/sendSetup(ncclChannelPeer.ncclConnector,ncclConnect)
//连接自己proxy。创建ncclProxyConnection,共享内存ncclProxyOpsPool,ncclProxyProgress线程。构建资源，监听xgbe0;eth0，返回addr
    ncclProxyConnect(connector.ncclProxyConnector)
        ncclSocketInit(ncclProxyState.peerSocks[rank],ncclProxyState.peerAddresses[rank])
        ncclSocketConnect
        //连接自己proxy
        ncclProxyCallBlocking(ncclProxyConnector, ncclProxyMsgInit, &req, sizeof(req), &resp, sizeof(resp))
        //创建ncclProxyConnection,共享内存ncclProxyOpsPool,ncclProxyProgress线程,返回前两者
            ncclProxyCallAsync(ncclProxyConnector, type, reqBuff, reqSize, respSize, opId)
                ncclSocketSend(peerSocks, &ncclProxyConnector->connection, sizeof(void*))
                expectedProxyResponseEnqueue(ProxyState)
            do 
                res = ncclPollProxyResponse(ncclProxyConnector, respBuff);
            while
        //保存返回结果
        ncclProxyConnector->connection = resp.connection;//pool.connection
        if (tcomm->proxyProgress) {
            struct ncclProxyOps* proxyOps = ProxyState->proxyOps + proxyConn->tpLocalRank;
            ncclShmOpen(poolPath, (void**)(&proxyOps->pool), &proxyOps->handle)
            proxyOps->nextOps = proxyOps->nextOpsEnd = proxyOps->freeOp = -1;
        ncclProxyConnector->initialized = true;
    ncclProxyCallBlocking(comm, &recv->proxyConn, ncclProxyMsgSetup, &req, sizeof(req), ncclConnect, sizeof(ncclNetHandle_t)));
    //Proxy构建Connection资源，监听xgbe0;eth0，返回addr
    memcpy((uint8_t*)connectInfo + sizeof(ncclNetHandle_t), &req.useGdr, sizeof(int));

recvProxySetup(ncclProxyConnector.ncclProxyConnection,ncclConnect)
//构建Connection资源，监听xgbe0;eth0，返回addr
    struct recvNetResources* resources;
    connection->transportResources = resources;
    proxyState->ncclNet->listen(req->netDev, ncclConnect, &resources->netListenComm)
    //ncclConnect存放xgbe0;eth0；netListenComm存放socket
    //ncclIbListen(int dev, void* ncclConnect, void** listenComm)
        struct ncclIbListenComm* comm;
        struct ncclIbHandle* handle = (struct ncclIbHandle*) ncclConnect;
        ncclSocketInit(&comm->sock, &ncclIbIfAddr//xgbe0;eth0
        ncclSocketListen(&comm->sock)
        ncclSocketGetAddr(&comm->sock, &handle->connectAddr)
        *listenComm = comm;

sendProxySetup
//构建Connection资源，不需要监听，不返回
    struct sendNetResources* resources;
```

connect

1. 连接recv rank的proxy listen socket/ accept send rank。每个网卡创建pd，cq，每个网卡创建两个qp，fifoMr，remSizesFifo.mrs。

2. 记录不同协议的buffer的起始位置，不同设备的buffer size。分配buffer。注册mr。send.ll在cpu，recv.ll在gpu

3. memcpy(ncclProxyConnector.transportResources,ncclProxyConnection.transportResources)

协议buffer size==comm->buffSizes[ll,ll128,simple]=={524288, 4915200, 4194304}

gpu buffer size==(connectMap*)(ncclProxyConnection->transportResources).mems[NCCL_NET_MAP_DEVMEM].size

send的gpu buffer size做了2MiB对齐，recv没有做对齐，但ncclCuMemAlloc 2MiB对齐，如果从buffer size去分析和实际会有误差

```
struct netSendConnectArgs {
  ncclNetHandle_t handle;//xgbe0;eth0
  int trafficClass;
};

struct ncclIbHandle {
  union ncclSocketAddress connectAddr;//xgbe0;eth0
  struct ncclIbCommStage stage; // Used by the other side when connecting
};

struct ncclIbCommStage {
  enum ncclIbCommState state;
  int offset;
  void* buffer;
  void* comm;//ncclIbSendComm
};

struct ncclIbSendComm {
  struct ncclIbNetCommBase base;
  struct ncclIbSendCommDev devs[NCCL_IB_MAX_DEVS_PER_NIC];
  struct ncclIbSendFifo fifo[MAX_REQUESTS][NCCL_NET_IB_MAX_RECVS];
  struct ncclIbRemSizesFifo remSizesFifo;

struct alignas(32) ncclIbNetCommBase {
  struct ncclSocket sock;//连接xgbe0;eth0
  int nqps;
  struct ncclIbQp qps[NCCL_IB_MAX_QPS];

struct ncclIbQp {
  struct ibv_qp* qp;
  int devIndex;
  int remDevIdx;
};

struct alignas(8) ncclIbSendCommDev {
  struct ncclIbNetCommDevBase base;
  struct ibv_mr* fifoMr;
};

struct ncclIbNetCommDevBase {
  int ibDevN;//网卡下标
  struct ibv_pd* pd;
  struct ibv_cq* cq;
  uint64_t pad[2];
  struct ncclIbGidInfo gidInfo;
};

struct ncclIbRemSizesFifo {
  int elems[MAX_REQUESTS][NCCL_NET_IB_MAX_RECVS];
  uint64_t addr;//remMeta.fifoAddr
  uint32_t rkeys[NCCL_IB_MAX_DEVS_PER_NIC];
  struct ibv_mr* mrs[NCCL_IB_MAX_DEVS_PER_NIC];
};

struct sendNetResources {
  struct connectMap map;
  void* netSendComm;//ncclIbSendComm
  struct ncclSendMem* sendMem;//cpu
  struct ncclRecvMem* recvMem;//cpu
  char* buffers[NCCL_NUM_PROTOCOLS];//ll在cpu
  void* mhandles[NCCL_NUM_PROTOCOLS];//buffer mr

struct connectMap {
  struct connectMapMem mems[NCCL_NET_MAP_MEMS];//不同设备
  // Offsets. 3 MSBs indicate mem bank, 111 indicates NULL.
  struct {
    uint32_t sendMem;
    uint32_t recvMem;
    uint32_t buffs[NCCL_NUM_PROTOCOLS];//ll在cpu
  } offsets;//起始位置（可能在不同设备）
};

struct connectMapMem{
  char* gpuPtr;//
  char* cpuPtr;//==gpuPtr
  int size;//buffer size
  ncclIpcDesc ipcDesc;
  ncclShmIpcDesc_t attachDesc;
  ncclShmIpcDesc_t createDesc;
};

typedef union {
  // Legacy CUDA IPC
  cudaIpcMemHandle_t devIpc;
  // cuMem API support
  struct {
    ncclCuDesc cuDesc;
    CUmemGenericAllocationHandle memHandle;
  };
} ncclIpcDesc;

typedef union {
  uint64_t data; // 
  CUmemFabricHandle handle;
} ncclCuDesc;

struct ncclIbMrCache {
  struct ncclIbMr *slots;
  int capacity, population;
};

struct ncclConnInfo {
  int flags;          // Direct communication / other flags
  char *buffs[NCCL_NUM_PROTOCOLS]; 
  uint64_t *tail;    
  uint64_t *head;     
  int stepSize;       // Step size for the SIMPLE buffer
};

canConnect
    *ret = 1;

sendConnect(ncclConnector,ncclConnect)
    struct connectMap* map = (connectMap*) send->transportResources;
    memcpy(&args.handle, ncclConnect, sizeof(ncclNetHandle_t));//xgbe0;eth0
    if (!map) {
        ncclProxyCallAsync(ncclProxyMsgConnect, &args, sizeof(netSendConnectArgs), sizeof(struct connectMap), opId));
    ret = ncclPollProxyResponse(comm, &send->proxyConn, map, opId)
    send->conn.stepSize = comm->buffSizes[NCCL_PROTO_SIMPLE]/NCCL_STEPS;
    for (int p=0; p<NCCL_NUM_PROTOCOLS; p++)
        send->ncclConnInfo.buffs[p] = NCCL_NET_MAP_GET_POINTER(map, gpu, buffs[p]);
    send->proxyConn.proxyProgress = sendProxyProgress;

sendProxyConnect(netSendConnectArgs,ncclConnector.transportResources)
//连接xgbe0;eth0。每个网卡创建pd，cq，每个网卡创建两个qp，fifoMr，remSizesFifo.mrs。
//记录不同协议的buffer的起始位置，不同设备的buffer size。分配buffer。注册mr。ll在cpu
//memcpy(ncclProxyConnector.transportResources,ncclProxyConnection.transportResources)
    struct sendNetResources* resources = (struct sendNetResources*)(connection->transportResources);
    ret = proxyState->ncclNet->connect(req->handle, &resources->netSendComm);
    //连接xgbe0;eth0。每个网卡创建pd，cq，每个网卡创建两个qp，fifoMr，remSizesFifo.mrs
    //ncclIbConnect(opaqueHandle, void** sendComm)
        struct ncclIbCommStage* stage = &handle->stage;//0000
        struct ncclIbSendComm* comm = (struct ncclIbSendComm*)stage->comm;

        ncclSocketInit(&comm->base.sock, &handle->connectAddr,//xgbe0;eth0
        stage->state = ncclIbCommStateConnect;
        ncclSocketConnect(&comm->base.sock)
        //连接xgbe0;eth0
        mergedDev = ncclIbMergedDevs + dev;

        stage->state = ncclIbCommStateSendDevList;
        memcpy(stage->buffer, &mergedDev->vProps
        //ndevs = 1, devs = {7, 0, 0, 0}}
        ncclSocketProgress(NCCL_SOCKET_SEND, &comm->base.sock, stage->buffer,

        stage->state = ncclIbCommStateRecvDevList;
        ncclSocketProgress(NCCL_SOCKET_RECV, &comm->base.sock, stage->buffer,
        memcpy(&remoteVProps, stage->buffer

        localNqps  = ncclParamIbQpsPerConn() * comm->base.vProps.ndevs; // We must have at least 1 qp per-device
        remoteNqps = ncclParamIbQpsPerConn() * remoteVProps.ndevs;
        comm->base.nqps = remoteNqps > localNqps ? remoteNqps : localNqps; // Select max nqps (local or remote)//2

        comm->ar = 1; // Set to 1 for logic
        for (int i = 0; i < comm->base.vProps.ndevs; i++) {
            int ibDevN = comm->base.vProps.devs[i];
            ncclIbInitCommDevBase(ibDevN, &comm->devs[i].base
                base->ibDevN = ibDevN;//下标
                wrap_ibv_alloc_pd(&ibDev->pd, ibDev->context
                //创建pd
                base->pd = ibDev->pd;
                wrap_ibv_create_cq(&base->cq, ibDev->context, 2*MAX_REQUESTS*ncclParamIbQpsPerConn()
                //创建cq
            comm->ar = comm->ar && ncclIbDevs[ibDevN].ar; // ADAPTIVE_ROUTING - if all merged devs have it enabled
        }

        for (int q = 0; q < comm->base.nqps; q++) {
            ncclIbSendCommDev* commDev = comm->devs + devIndex;
            ncclIbDev* ibDev = ncclIbDevs + commDev->base.ibDevN;
            ncclIbCreateQp(&commDev->base, comm->base.qps + q)
                qpInitAttr.send_cq = base->cq;
                qpInitAttr.recv_cq = base->cq;
                wrap_ibv_create_qp(&qp->qp, base->pd, &qpInitAttr)
                //创建qp

        for (int i = 0; i < comm->base.vProps.ndevs; i++) {
            wrap_ibv_reg_mr(&commDev->fifoMr, commDev->base.pd, comm->fifo, sizeof(struct ncclIbSendFifo)*MAX_REQUESTS*NCCL_NET_IB_MAX_RECVS, IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_REMOTE_READ), ret, fail);
            //131072

        meta.fifoAddr = (uint64_t)comm->fifo;
        stage->state = ncclIbCommStateSend;
        memcpy(stage->buffer, &meta, sizeof(meta));
        ncclSocketProgress(NCCL_SOCKET_SEND, &comm->base.sock, stage->buffer, sizeof(meta)

        stage->state = ncclIbCommStateConnecting;
        ncclSocketProgress(NCCL_SOCKET_RECV, &comm->base.sock, stage->buffer, sizeof(ncclIbConnectionMetadata)
        memcpy(&remMeta, stage->buffer, sizeof(ncclIbConnectionMetadata));
        for (int i = 0; i < remMeta.ndevs; i++) {
            comm->remSizesFifo.addr = remMeta.fifoAddr;
        for (int i=0; i < comm->base.vProps.ndevs; i++) {
            wrap_ibv_reg_mr(comm->remSizesFifo.mrs+i, comm->devs[i].base.pd, &comm->remSizesFifo.elems, sizeof(int)*MAX_REQUESTS*NCCL_NET_IB_MAX_RECVS, IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_READ), ret, fail);
        }

        for (int q = 0; q < comm->base.nqps; q++) {
            ncclIbRtrQp
            ncclIbRtsQp

        stage->state = ncclIbCommStateConnected;
        ncclSocketProgress(NCCL_SOCKET_SEND, &comm->base.sock, &comm->base.ready, sizeof(int)
        *sendComm = comm;        
    connection->needsProxyProgress = 1;   
    struct connectMap* map = &resources->map; 
    for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
      NCCL_NET_MAP_ADD_POINTER(map, 0, p!= NCCL_PROTO_LL && resources->useGdr ? 1 : 0, proxyState->buffSizes[p], buffs[p]);
      //(mapStruct, shared, dev, memSize, offsetName)
          #define NCCL_NET_MAP_MASK_DEVMEM 0x40000000
          #define NCCL_NET_MAP_MASK_SHARED 0x80000000
          #define NCCL_NET_MAP_MASK_USED   0x20000000 
          //>>30,区分不同设备
          #define NCCL_NET_MAP_HOSTMEM 0
          #define NCCL_NET_MAP_DEVMEM 1
          #define NCCL_NET_MAP_SHARED_HOSTMEM 2
          #define NCCL_NET_MAP_SHARED_DEVMEM 3
          int bank = NCCL_NET_MAP_MASK_USED + (dev)*NCCL_NET_MAP_MASK_DEVMEM + (shared)*NCCL_NET_MAP_MASK_SHARED; 
          if dev
              map->offsets.buffs[p]=bank+map->mems[NCCL_NET_MAP_DEVMEM].size;
              map->mems[NCCL_NET_MAP_DEVMEM].size+=memSize;
          else
              map->offsets.buffs[p]=bank+map->mems[NCCL_NET_MAP_HOSTMEM].size;
              map->mems[NCCL_NET_MAP_HOSTMEM].size+=memSize;
      //ll在cpu上
      //记录不同协议的buffer的起始位置，不同设备的buffer size
      resources->buffSizes[p] = proxyState->buffSizes[p];
    }
    NCCL_NET_MAP_ADD_POINTER(map, 0, 0, sizeof(struct ncclSendMem), sendMem);//sendMem起始位置，占据一定buffer
    NCCL_NET_MAP_ADD_POINTER(map, 0, 0, sizeof(struct ncclRecvMem), recvMem);
    ALIGN_SIZE(map->mems[NCCL_NET_MAP_DEVMEM].size, CUDA_IPC_MIN);//2MiB
    ncclP2pAllocateShareableBuffer(map->mems[NCCL_NET_MAP_DEVMEM].size, 0, &map->mems[NCCL_NET_MAP_DEVMEM].ipcDesc,
                                                 (void**)&map->mems[NCCL_NET_MAP_DEVMEM].gpuPtr));
    //(size_t size, int refcount, ncclIpcDesc *ipcDesc, void **ptr)
        //type==CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
        ncclCuMemAlloc(ptr, &handle, type, size)
            prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.requestedHandleTypes = type;
            if (flag) prop.allocFlags.gpuDirectRDMACapable = 1;
            cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM)//2MiB
            ALIGN_SIZE(size, granularity);
            /* Allocate the physical memory on the device */
            CUCHECK(cuMemCreate(&handle, size, &prop, 0));
            /* Reserve a virtual address range */
            CUCHECK(cuMemAddressReserve((CUdeviceptr *)ptr, size, granularity, 0, 0));
            /* Map the virtual address range to the physical allocation */
            CUCHECK(cuMemMap((CUdeviceptr)*ptr, size, 0, handle, 0));
            /* Now allow RW access to the newly mapped memory */
            accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            accessDesc.location.id = currentDev;
            accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            CUCHECK(cuMemSetAccess((CUdeviceptr)*ptr, size, &accessDesc, 1));
            TRACE(NCCL_ALLOC, "CuMem Alloc Size %zu pointer %p handle %llx", size, *ptr, handle);
        memcpy(&ipcDesc->cuDesc.data, &handle, sizeof(handle))
        INFO(NCCL_P2P|NCCL_ALLOC, "Allocated shareable buffer %p size %zu ipcDesc %p", *ptr, size, ipcDesc);
        //这里的值不一定2MiB对齐
    map->mems[NCCL_NET_MAP_DEVMEM].cpuPtr = map->mems[NCCL_NET_MAP_DEVMEM].gpuPtr;
    ncclCudaHostCalloc(&map->mems[NCCL_NET_MAP_HOSTMEM].cpuPtr, map->mems[NCCL_NET_MAP_HOSTMEM].size)
    map->mems[NCCL_NET_MAP_HOSTMEM].gpuPtr = map->mems[NCCL_NET_MAP_HOSTMEM].cpuPtr;

    resources->sendMem = (struct ncclSendMem*) NCCL_NET_MAP_GET_POINTER(map, cpu, sendMem);
    #(mapStruct, cpuOrGpu, offsetName)
        (mapStruct)->mems[NCCL_NET_MAP_OFFSET_BANK(mapStruct, offsetName)].cpuOrGpu##Ptr + ((mapStruct)->offsets.offsetName & NCCL_NET_MAP_MASK_OFFSET))
            ((mapStruct)->offsets.offsetName >> 30)//设备
            #define NCCL_NET_MAP_MASK_OFFSET 0x1fffffff
        //map->mems[NCCL_NET_MAP_HOSTMEM].cpuPtr+map->offsets.sendMem & 0x1fffffff
    resources->recvMem = (struct ncclRecvMem*) NCCL_NET_MAP_GET_POINTER(map, cpu, recvMem);
    for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
        resources->buffers[p] = NCCL_NET_MAP_GET_POINTER(map, cpu, buffs[p]);
        proxyState->ncclNet->regMr(resources->netSendComm, resources->buffers[p], resources->buffSizes[p], NCCL_NET_MAP_DEV_MEM(map, buffs[p]) ? NCCL_PTR_CUDA : NCCL_PTR_HOST, &resources->mhandles[p]));
        //注册mr
            ncclIbRegMrDmaBuf(comm, data, size, type, 0ULL, -1, mhandle);
            //(void* comm, void* data, size_t size, int type, uint64_t offset, int fd, void** mhandle)
                struct ncclIbNetCommBase* base = (struct ncclIbNetCommBase*) comm;
                for (int i = 0; i < base->vProps.ndevs; i++) {
                    struct ncclIbNetCommDevBase* devComm = ncclIbGetNetCommDevBase(base, i);
                    ncclIbRegMrDmaBufInternal(devComm, data, size, type, offset, fd, mhandleWrapper->mrs + i), ret, fail);
                        struct ncclIbMrCache* cache = &ncclIbDevs[base->ibDevN].mrCache;
                        uintptr_t addr = (uintptr_t)data & -pageSize;// 对齐到页边界
                        size_t pages = ((uintptr_t)data + size - addr + pageSize-1)/pageSize;
                        for (int slot=0; /*true*/; slot++) {
                            if (slot == cache->population || addr < cache->slots[slot].addr) { // didn't find in cache
                              if (cache->population == cache->capacity) { // must grow cache
                                cache->capacity = cache->capacity < 32 ? 32 : 2*cache->capacity;
                                NCCLCHECKGOTO(ncclRealloc(&cache->slots, cache->population, cache->capacity), res, returning);
                              }
                              wrap_ibv_reg_mr_iova2(&mr, base->pd, (void*)addr, pages*pageSize, addr, flags),
                              if (slot != cache->population) memmove(cache->slots+slot+1, cache->slots+slot, (cache->population-slot)*sizeof(struct ncclIbMr));
                              //往后移，保持地址从小到大
                              cache->slots[slot].addr = addr;
                              cache->slots[slot].pages = pages;
                              cache->slots[slot].refs = 1;
                              cache->slots[slot].mr = mr;
                              cache->population += 1;
                              *mhandle = mr
                            } else if ((addr >= cache->slots[slot].addr) &&
                                ((addr-cache->slots[slot].addr)/pageSize+pages) <= cache->slots[slot].pages) {///可以使用缓存
                                cache->slots[slot].refs += 1;
                                *mhandle = cache->slots[slot].mr;
    memcpy(respBuff, map, sizeof(struct connectMap))
    //memcpy(ncclProxyConnector.transportResources,ncclProxyConnection.transportResources)

struct recvNetResources {
  struct connectMap map;
  void* netListenComm;//xgbe0;eth0
  void* netRecvComm;//ncclIbRecvComm

struct ncclIbListenComm {
  int dev;
  struct ncclSocket sock;//listen on xgbe0;eth0
  struct ncclIbCommStage stage;
};

struct ncclIbCommStage {
  enum ncclIbCommState state;
  int offset;
  void* buffer;
  void* comm;//ncclIbRecvComm
};

struct ncclIbRecvComm {
  struct ncclIbNetCommBase base;
  struct ncclIbRecvCommDev devs[NCCL_IB_MAX_DEVS_PER_NIC];
  struct ncclIbRemFifo remFifo;
  int sizesFifo[MAX_REQUESTS][NCCL_NET_IB_MAX_RECVS];
  int gpuFlushHostMem;
  int flushEnabled;
};

struct alignas(32) ncclIbNetCommBase {
  struct ncclSocket sock;//接收send连接
  int nqps;
  struct ncclIbQp qps[NCCL_IB_MAX_QPS];

recvConnect(ncclConnector,ncclConnect)
    if (!map) {
        ncclProxyCallAsync(comm, &recv->proxyConn, ncclProxyMsgConnect, &args, sizeof(netRecvConnectArgs), sizeof(struct connectMap), opId));
        //accept，ll在gpu

recvProxyConnect(netSendConnectArgs,ncclConnector.transportResources)
    struct recvNetResources* resources = (struct recvNetResources*)(connection->transportResources);
    ret = proxyState->ncclNet->accept(resources->netListenComm, &resources->netRecvComm, &resources->netDeviceHandle);
    //ncclIbAccept
        stage = &lComm->stage
        rComm = (struct ncclIbRecvComm*)stage->comm
        ncclSocketInit(&rComm->base.sock)
        ncclSocketAccept(&rComm->base.sock, &lComm->sock)
        ncclSocketProgress(NCCL_SOCKET_RECV, &rComm->base.sock,
        memcpy(&remoteVProps, stage->buffer
        memcpy(stage->buffer, &rComm->base.vProps
        ncclSocketProgress(NCCL_SOCKET_SEND, &rComm->base.sock
        ncclSocketProgress(NCCL_SOCKET_RECV, &rComm->base.sock
        memcpy(&remMeta, stage->buffer
        for 
            ncclIbInitCommDevBase
        for 
            ncclIbCreateQp
            ncclIbRtrQp
            ncclIbRtsQp
        for 
            wrap_ibv_reg_mr(&rCommDev->fifoMr,
        *recvComm = rComm;
    proxyState->ncclNet->closeListen(resources->netListenComm)
    for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
      NCCL_NET_MAP_ADD_POINTER(map, 0, resources->useGdr ? 1 : 0, proxyState->buffSizes[p], buffs[p]);
      resources->buffSizes[p] = proxyState->buffSizes[p];
    }
    //ll在gpu
    ncclP2pAllocateShareableBuffer(map->mems[NCCL_NET_MAP_DEVMEM].size, 0, &map->mems[NCCL_NET_MAP_DEVMEM].ipcDesc,
                                                 (void**)&map->mems[NCCL_NET_MAP_DEVMEM].gpuPtr));
    proxyState->ncclNet->regMr(resources->netRecvComm, resources->buffers[p], resources->buffSizes[p], NCCL_NET_MAP_DEV_MEM(map, buffs[p]) ? NCCL_PTR_CUDA : NCCL_PTR_HOST, &resources->mhandles[p]));
```

ProxyProgress

```
recvProxyProgress
//post,receive,transmit,done
    if (args->state == ncclProxyOpReady) {
    // Initialize subs and group them by same recvComm.
        int groupSize = 0;
        int maxRecvs = 1;
        for (int s=0; s<args->nsubs; s++) {
            struct ncclProxySubArgs* sub = args->subs+s;
            groupSize++;
            sub->posted = sub->received = sub->transmitted = sub->done = 0;
            for (int i=0; i<groupSize; i++) sub[-i].groupSize = groupSize;
            if (!sub->reg)
                sub->recvMhandle = resources->mhandles[args->protocol];
        args->state = ncclProxyOpProgress;
    args->idle = 1;
    if (args->state == ncclProxyOpProgress) {
        int p = args->protocol;
        for (int s=0; s<args->nsubs; s+=args->subs[s].groupSize) {
        //几个sub一组
            struct ncclProxySubArgs* subGroup = args->subs+s;
            int subCount = 0;
            for (int i=0; i<subGroup->groupSize; i++) {
            //组内
                struct ncclProxySubArgs* sub = subGroup + i;
                if (sub->posted < sub->nsteps) {///有任务要post
                    } else {
                        ptrs[subCount] = localBuff+buffSlot*stepSize;
                        sizes[subCount] = stepSize*args->sliceSteps;
                    }
                    subCount++;
            if (subCount) {
                uint64_t step = subGroup->posted;
                (proxyState->ncclNet->irecv(resources->netRecvComm, subCount, ptrs, sizes, tags, mhandles, phandles, requestPtr));
                if (*requestPtr) {
                //发布任务给网卡成功
                    for (int i=0; i<subGroup->groupSize; i++) {
                        sub->posted += args->sliceSteps;
                    args->idle = 0;
        if (args->idle == 0) return ncclSuccess;
        //启动任务后先返回
        for (int s=0; s<args->nsubs; s+=args->subs[s].groupSize) {
            struct ncclProxySubArgs* subGroup = args->subs+s;
            if (subGroup->posted > subGroup->received) {//有数据要接收
                NCCLCHECK(proxyState->ncclNet->test(subGroup->requests[step%NCCL_STEPS], &done, sizes));
                if (done) {
                //数据已完全到达并可用
                    for (int i=0; i<subGroup->groupSize; i++) {
                        sub->received += args->sliceSteps;
                        ...
        if (args->done == args->nsubs) {
            args->state = ncclProxyOpNone;
        }

sendProxyProgress
//post,transmit,done
    if (args->state == ncclProxyOpReady) {
        for (int s=0; s<args->nsubs; s++) {
            sub->posted = sub->transmitted = sub->done = 0;
    if (args->state == ncclProxyOpProgress) {
        for (int s=0; s<args->nsubs; s++) {
            if (sub->posted < sub->nsteps && sub->posted < sub->done + maxDepth) {
            // Post buffers to the GPU
                sub->posted += args->sliceSteps;
            if (sub->transmitted < sub->posted && sub->transmitted < sub->done + NCCL_STEPS) {
            // Check whether we received data from the GPU and send it to the network
                if (connFifo[buffSlot].size != -1 && (*recvTail > tail || p == NCCL_PROTO_LL)) {
                // We have something to receive, let's check if it's completely ready.
                    if (ready) {
                        sub->transmitted += args->sliceSteps;
                        ...
```

proxySharedInit

```
sharedNetBuffersInit(proxyState, 1, connection->tpLocalRank, 0, connection->sameProcess, nChannels, NULL, NULL, NULL, NULL));
  if (state->size == 0) {
    state->size = nChannels * NCCL_SHARED_STEPS * proxyState->p2pChunkSize;//16*16*131072
  }
  ncclP2pAllocateShareableBuffer(state->size, 0, &state->ipcDesc, (void**)&state->cudaBuff));
```

# 设置理论带宽和时延

1. 确定每种算法、协议的blocksize

2. 确定原语在不同算法，协议下的带宽和时延

3. 解析NCCL_PROTO，NCCL_ALGO

4. 确定单cuda线程在每种算法、协议下处理的数据量

```
  Algorithm   |                            Tree                  |                            Ring                  |                   CollNetDirect                  |
  Protocol    |             LL |          LL128 |         Simple |             LL |          LL128 |         Simple |             LL |          LL128 |         Simple |
 Max NThreads |            512 |            640 |            512 |            512 |            640 |            512 |              0 |              0 |            640 |
    Broadcast |     0.0/   0.0 |     0.0/   0.0 |     0.0/   0.0 |    49.8/  39.0 |    78.0/ 293.6 |   232.4/ 384.0 |     0.0/   0.0 |     0.0/   0.0 |     0.0/   0.0 |
       Reduce |     0.0/   0.0 |     0.0/   0.0 |     0.0/   0.0 |    49.8/  39.0 |    78.0/ 293.6 |   232.4/ 384.0 |     0.0/   0.0 |     0.0/   0.0 |     0.0/   0.0 |
    AllGather |     0.0/   0.0 |     0.0/   0.0 |     0.0/   0.0 |    23.3/  41.6 |    44.6/ 313.2 |    70.0/ 409.6 |     5.6/   0.0 |     5.6/   0.0 |    44.0/   0.0 |
ReduceScatter |     0.0/   0.0 |     0.0/   0.0 |     0.0/   0.0 |    23.3/  41.6 |    44.6/ 313.2 |    70.0/ 409.6 |     5.6/   0.0 |     5.6/   0.0 |    44.0/   0.0 |
    AllReduce |    25.2/  16.6 |    48.5/ 124.8 |    92.4/ 140.8 |    40.0/  20.8 |    75.2/ 156.6 |   131.6/ 204.8 |     5.6/   0.0 |     5.6/   0.0 |  
```

Overheads in us, algbws in KB/us，小数据看overhead，大数据看带宽

时延=overhead+S/algbw，overhead和通信次数相关，算法带宽和bus带宽有关系，bus带宽和算法及协议相关，S=max(send,recv)

和nccl-tests相反的推理方向

1. tree，ll，ll128的bus带宽要打折扣，且有上限

2. bus带宽转算法带宽，和算法及nranks相关（tree直接0.5）

3. overhead分机内和机间，机内通信次数 * 机内开销+机间通信次数 * 机间开销

时延和环境、数据量、算法、协议、nChannels相关，这里没有涉及buffer size?

```
环境：intel,nranks=2*8
搜索：intraHw=nvlink,hw=net,bwIntra=bwInter=48,nChannels=8,busBw=48*8=384,sameChannels=0
代码：llMaxBw=39,perChMaxTreeBw=41.4,perChMaxRingLL128Bw=36.7,perChMaxTreeLL128Bw=36.7

broadcast/reduce
ll:
busBw = std::min(llMaxBw, busBw * .5);//39
ratio *= (1.0 * nRanks) / nsteps;//1
busBw *= ratio;//39
latencies[coll][a][p] = baseLat[a][p];//6.6
lat = hwLat[hw[a]][a][p];//2.7
latencies[coll][a][p] += nsteps*lat;//49.8

ll128:
busBw = std::min(busBw * (0.92 /*120.0/128.0*/), graphs[a]->nChannels*perChMaxRingLL128Bw);//293.6
ratio *= (1.0 * nRanks) / nsteps;
busBw *= ratio;//293.6
latencies[coll][a][p] = baseLat[a][p];//14
lat = hwLat[hw[a]][a][p];//4
latencies[coll][a][p] += nsteps*lat;//78

simple:
ratio *= (1.0 * nRanks) / nsteps;
busBw *= ratio;//384
latencies[coll][a][p] = baseLat[a][p];//8.4
lat = hwLat[hw[a]][NCCL_ALGO_TREE][p]//14
latencies[coll][a][p] += nsteps*lat;//232.4

allgather/reducescatter
ll:
busBw = std::min(llMaxBw, busBw * .5);//39
ratio *= (1.0 * nRanks) / nsteps;//16/15
busBw *= ratio;//41.6
latencies[coll][a][p] = baseLat[a][p];//6.6
intraLat = hwLat[intraHw[a]][a][p];//0.6
interLat = hwLat[NCCL_HW_NET][a][p];//2.7
netOverhead=1
intraLat = std::max(intraLat, netOverhead);//1
nInterSteps = coll == ncclFuncAllReduce ? 2*(nNodes-1) : nNodes-1;//1
latencies[coll][a][p] += (nsteps-nInterSteps)*intraLat + nInterSteps*interLat;//23.3

ll128:
busBw = std::min(busBw * (0.92 /*120.0/128.0*/), graphs[a]->nChannels*perChMaxRingLL128Bw);//293.6
ratio *= (1.0 * nRanks) / nsteps;//16/15
busBw *= ratio;//313.2
latencies[coll][a][p] = baseLat[a][p];//14
intraLat = hwLat[intraHw[a]][a][p];//1.9
interLat = hwLat[NCCL_HW_NET][a][p];//4
netOverhead=1
intraLat = std::max(intraLat, netOverhead);//1.9
nInterSteps = coll == ncclFuncAllReduce ? 2*(nNodes-1) : nNodes-1;//1
latencies[coll][a][p] += (nsteps-nInterSteps)*intraLat + nInterSteps*interLat;//44.6

simple:
ratio *= (1.0 * nRanks) / nsteps;//16/15
busBw *= ratio;//409.6
latencies[coll][a][p] = baseLat[a][p];//8.4
intraLat = hwLat[intraHw[a]][a][p];//3.4
interLat = hwLat[NCCL_HW_NET][a][p];//14
netOverhead=3
intraLat = std::max(intraLat, netOverhead);//3.4
nInterSteps = coll == ncclFuncAllReduce ? 2*(nNodes-1) : nNodes-1;//1
latencies[coll][a][p] += (nsteps-nInterSteps)*intraLat + nInterSteps*interLat;//70

allreduce
ring:
ll:
busBw = std::min(llMaxBw, busBw * .5);//39
ratio *= (1.0 * nRanks) / nsteps;//16/30
busBw *= ratio;//20.8
latencies[coll][a][p] = baseLat[a][p];//6.6
intraLat = hwLat[intraHw[a]][a][p];//0.6
interLat = hwLat[NCCL_HW_NET][a][p];//2.7
netOverhead=1
intraLat = std::max(intraLat, netOverhead);//1
nInterSteps = coll == ncclFuncAllReduce ? 2*(nNodes-1) : nNodes-1;//2
latencies[coll][a][p] += (nsteps-nInterSteps)*intraLat + nInterSteps*interLat;//40

ll128:
busBw = std::min(busBw * (0.92 /*120.0/128.0*/), graphs[a]->nChannels*perChMaxRingLL128Bw);//293.6
ratio *= (1.0 * nRanks) / nsteps;//16/30
busBw *= ratio;//156.6
latencies[coll][a][p] = baseLat[a][p];//14
intraLat = hwLat[intraHw[a]][a][p];//1.9
interLat = hwLat[NCCL_HW_NET][a][p];//4
netOverhead=1
intraLat = std::max(intraLat, netOverhead);//1.9
nInterSteps = coll == ncclFuncAllReduce ? 2*(nNodes-1) : nNodes-1;//2
latencies[coll][a][p] += (nsteps-nInterSteps)*intraLat + nInterSteps*interLat;//75.2

simple:
ratio *= (1.0 * nRanks) / nsteps;//16/30
busBw *= ratio;//204.8
latencies[coll][a][p] = baseLat[a][p];//8.4
intraLat = hwLat[intraHw[a]][a][p];//3.4
interLat = hwLat[NCCL_HW_NET][a][p];//14
netOverhead=3
intraLat = std::max(intraLat, netOverhead);//3.4
nInterSteps = coll == ncclFuncAllReduce ? 2*(nNodes-1) : nNodes-1;//2
latencies[coll][a][p] += (nsteps-nInterSteps)*intraLat + nInterSteps*interLat;//131.6

tree:
ll:
busBw = std::min(busBw*.92, graphs[a]->nChannels*perChMaxTreeBw);//331.2
busBw = std::min(busBw*1.0/3.8, llMaxBw);//39
busBw *= .85;
ratio *= .5;
busBw *= ratio;//16.6
latencies[coll][a][p] = baseLat[a][p];//6.8
intraLat = hwLat[intraHw[a]][a][p];//0.6
interLat = hwLat[NCCL_HW_NET][a][p];//5
latencies[coll][a][p] += 2 * ((nRanks/nNodes-1) * intraLat + log2i(nNodes) * interLat);//25.2

ll128:
busBw = std::min(busBw*.92, graphs[a]->nChannels*perChMaxTreeBw);//331.2
busBw = std::min(busBw * (nNodes == 1 ? 7.0/9.0 : 120.0/128.0), graphs[a]->nChannels*perChMaxTreeLL128Bw);//293.6
busBw *= .85;
ratio *= .5;
busBw *= ratio;//124.8
latencies[coll][a][p] = baseLat[a][p];//14
intraLat = hwLat[intraHw[a]][a][p];//1.25
interLat = hwLat[NCCL_HW_NET][a][p];//8.5
latencies[coll][a][p] += 2 * ((nRanks/nNodes-1) * intraLat + log2i(nNodes) * interLat);//48.5

simple:
busBw = std::min(busBw*.92, graphs[a]->nChannels*perChMaxTreeBw);//331.2
busBw *= .85;
ratio *= .5;
busBw *= ratio;//140.8
latencies[coll][a][p] = baseLat[a][p];//8.4
intraLat = hwLat[intraHw[a]][a][p];//4
interLat = hwLat[NCCL_HW_NET][a][p];//14
latencies[coll][a][p] += 2 * ((nRanks/nNodes-1) * intraLat + log2i(nNodes) * interLat);//92.4
```

```
//列：单机，两机，多机
//行：单机看gpu，多机看是否为amd
static const double llMaxBws[][3] = {
  /* Volta-N1/Intel-N2/Intel-N4) */ {39.0, 39.0, 20.4},
  /* Ampere-N1/AMD-N2/AMD-N4) */ {87.7, 22.5 /*avg of ring & tree*/, 19.0},
  /* Hopper-N1/AMD-N2/AMD-N4) */ {141.0, 45.0 /*avg of ring & tree*/, 35.0},
  /* Blackwell-N1/AMD-N2/AMD-N4) */ {2*141.0, 2*45.0 /*avg of ring & tree*/, 2*35.0},
};
//行只看gpu
static const double perChMaxRingLL128Bws[][3] = {
  /* Volta (N1/N2/N4) */  {20.0, 20.0, 20.0},
  /* Ampere (N1/N2/N4) */ {20.0, 20.0, 20.0},
  /* Hopper (N1/N2/N4) */ {36.7, 36.7, 36.7},
  /* Blackwell (N1/N2/N4) */ {2*36.7, 2*36.7, 2*36.7},
};
static const double perChMaxTreeLL128Bws[][3] = {
  /* Volta (N1/N2/N4) */  {20.0, 20.0, 20.0},
  /* Ampere (N1/N2/N4) */ {20.0, 20.0, 20.0},
  /* Hopper (N1/N2/N4) */ {36.7, 36.7, 29.0},
  /* Blackwell (N1/N2/N4) */ {2*36.7, 2*36.7, 2*29.0},
};
static const double perChMaxTreeBws[][3] = {
  /* Volta (N1/N2/N4) */  {26.5, 18.5, 10.0},
  /* Ampere (N1/N2/N4) */ {24.0, 23.6, 17.8},
  /* Hopper (N1/N2/N4) */ {38.7, 41.4, 36.0},
  /* Blackwell (N1/N2/N4) */ {2*38.7, 2*41.4, 2*36.0},
};

// Tree { LL, LL128, Simple } , Ring { LL, LL128, Simple }
static const float baseLat  [NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS] = {
       {  6.8, 14.0,  8.4 }, {  6.6, 14.0,  8.4 },  // Tree, Ring
       {    0,    0,    0 }, {    0,    0,    0 },  // Collnet Direct, Chain
       {    0,    0,    0 }, {    0,    0,    0 }}; // NVLS, NVLS Tree
static float hwLat [3][NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS] =
{ /* NVLINK */
  { /* Tree (LL/LL128/Simple)*/ { .6, 1.25, 4.0 }, /* Ring (LL/LL128/Simple)*/ { .6, 1.9, 3.4 },
    /* CollNetDirect (Simple)*/ { 0, 0, 3.7 }, /* CollNetChain (Simple)*/ { 0, 0, 2.8 },
    /* NVLS */ { 0, 0, 25 }, /* NVLSTree */ { 0, 0, 25 } },
  /* PCI */
  { /* Tree (LL/LL128/Simple)*/ { 1.0, 1.9, 4.0 }, /* Ring (LL/LL128/Simple)*/ { 1.0, 2.5, 5.7 },
    /* CollNetDirect (Simple)*/ { 0, 0, 3.7 }, /* CollNetChain (Simple)*/ { 0, 0, 2.8 },
    /* NVLS */ { 0, 0, 0 }, /* NVLSTree */ { 0, 0, 0 } },
  /* NET */
  { /* Tree (LL/LL128/Simple)*/ { 5.0, 8.5, 14 }, /* Ring (LL/LL128/Simple)*/ { 2.7, 4.0, 14.0 },
    /* CollNetDirect (Simple)*/ { 0, 0, 31 }, /* CollNetChain (Simple)*/ { 0, 0, 30 },
    /* NVLS */ { 0, 0, 18 }, /* NVLSTree */ { 0, 0, 14 } }
};
```

```
initTransportsRank
    struct ncclProxyConnector proxyConn;
    ncclProxyConnect(comm, TRANSPORT_NET, 1, comm->rank, &proxyConn)
    //连接自己proxy,创建ncclProxyConnection
    ncclProxyCallBlocking(comm, &proxyConn, ncclProxyMsgSharedInit, &comm->p2pnChannels, sizeof(int), NULL, 0), ret, fail);
    //分配buffer
    if (ncclPxnDisable(comm) == 0) {//节点内的gpu通过网络通信
      int nranks;
      NCCLCHECKGOTO(ncclTopoGetPxnRanks(comm, &pxnPeers, &nranks), ret, fail);
      for (int r=0; r<nranks; r++) {//7
        ncclProxyConnect(comm, TRANSPORT_NET, 1, pxnPeers[r], &proxyConn), ret, fail);
        //连接别人的proxy,创建ncclProxyConnection
        ncclProxyCallBlocking(comm, &proxyConn, ncclProxyMsgSharedInit, &comm->p2pnChannels, sizeof(int), NULL, 0), ret, fail);
        //分配buffer
      }
    }
    ncclTopoTuneModel(comm, comm->minCompCap, comm->maxCompCap, graphs), ret, fail);
        comm->maxThreads[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE] =
        int compCapIndex = minCompCap >= 100 ? BLACKWELL_COMPCAP_IDX : (minCompCap >= 90 ? HOPPER_COMPCAP_IDX : minCompCap >= 80 ? AMPERE_COMPCAP_IDX : VOLTA_COMPCAP_IDX);
        int index2 = nNodes <= 2 ? nNodes-1 : 2;
        int index1 = nNodes == 1 ? compCapIndex :
               (comm->cpuVendor == NCCL_TOPO_CPU_VENDOR_AMD || comm->cpuVendor == NCCL_TOPO_CPU_VENDOR_MIXED) ? 1 : 0;
        double llMaxBw = llMaxBws[index1][index2];
        double perChMaxTreeBw = perChMaxTreeBws[compCapIndex][index2];
        double perChMaxRingLL128Bw = perChMaxRingLL128Bws[compCapIndex][index2];
        double perChMaxTreeLL128Bw = perChMaxTreeLL128Bws[compCapIndex][index2];
        float ppn = (float)nRanks / nNodes;
        for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) intraHw[a] = graphs[a]->typeIntra == LINK_NVL ? NCCL_HW_NVLINK : NCCL_HW_PCI;
        for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) hw[a] = nNodes == 1 ? intraHw[a] : NCCL_HW_NET;
        //带宽类型
        for (int coll=0; coll<NCCL_NUM_FUNCTIONS; coll++) {
            int nsteps = coll == ncclFuncAllReduce ? 2*(nRanks-1) :
              coll == ncclFuncReduceScatter || coll == ncclFuncAllGather ? nRanks-1 :
              nRanks;  //  ncclFuncBroadcast = 0, ncclFuncReduce = 1,
            for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
                if ((coll == ncclFuncBroadcast || coll == ncclFuncReduce) && a != NCCL_ALGO_RING) continue;
                if ((coll == ncclFuncReduceScatter || coll == ncclFuncAllGather)
                  && a != NCCL_ALGO_PAT && a != NCCL_ALGO_RING
                  && a != NCCL_ALGO_NVLS && a != NCCL_ALGO_COLLNET_DIRECT) continue;
                for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
                    float bw = nNodes <= 2 || collnet ? graphs[a]->bwIntra : graphs[a]->bwInter;//多机这两个是相等的
                    float busBw = graphs[a]->nChannels * bw;
                    if (a == NCCL_ALGO_RING && p == NCCL_PROTO_LL) { busBw = std::min(llMaxBw, busBw * .5); }
                    if (a == NCCL_ALGO_RING && p == NCCL_PROTO_LL128) busBw = std::min(busBw * (0.92 /*120.0/128.0*/), graphs[a]->nChannels*perChMaxRingLL128Bw);
                    if (a == NCCL_ALGO_TREE && coll == ncclFuncAllReduce) busBw = std::min(busBw*.92, graphs[a]->nChannels*perChMaxTreeBw);
                    if (a == NCCL_ALGO_TREE && p == NCCL_PROTO_LL) busBw = std::min(busBw*1.0/3.8, llMaxBw);
                    if (a == NCCL_ALGO_TREE && p == NCCL_PROTO_LL128) busBw = std::min(busBw * (nNodes == 1 ? 7.0/9.0 : 120.0/128.0), graphs[a]->nChannels*perChMaxTreeLL128Bw);
                    if (a == NCCL_ALGO_TREE && comm->maxTreePattern == NCCL_TOPO_PATTERN_TREE) busBw *= .85;
                    if (!(a != NCCL_ALGO_RING && (coll == ncclFuncAllGather || coll == ncclFuncReduceScatter))) {
                     //a==ring||(coll!=allgather&&coll!=reducescatter)
                        float ratio = 1.0f;
                        if (a == NCCL_ALGO_RING) ratio *= (1.0 * nRanks) / nsteps;
                        else if (a == NCCL_ALGO_NVLS || a == NCCL_ALGO_NVLS_TREE) ratio *= 5.0/6.0;
                        else ratio *= .5;
                        busBw *= ratio;
                    comm->bandwidths[coll][a][p] = busBw;
                    comm->latencies[coll][a][p] = baseLat[a][p];
                    float intraLat = hwLat[intraHw[a]][a][p];
                    float interLat = ppn == 1 ? hwLat[NCCL_HW_NET][NCCL_ALGO_TREE][p] : hwLat[NCCL_HW_NET][a][p];
                    interLat += graphs[a]->latencyInter;
                    if (p == NCCL_PROTO_SIMPLE) interLat += graphs[a]->latencyInter;
                    if (a == NCCL_ALGO_RING) {
                      float lat = hwLat[hw[a]][a][p];
                      if ((coll == ncclFuncReduce || coll == ncclFuncBroadcast)) {
                        if (graphs[a]->sameChannels) {
                          comm->latencies[coll][a][p] += lat;
                        } else {
                          if (p == NCCL_PROTO_SIMPLE) lat = hwLat[hw[a]][NCCL_ALGO_TREE][p]; // Add some chunk latency, waiting for proper chunk modeling
                          comm->latencies[coll][a][p] += nsteps*lat;
                        }
                      } else {
                        // Inter-node rings still have to launch nsteps * net overhead.
                        float netOverhead = 0.0;
                        if (nNodes > 1) {
                          netOverhead = getNetOverhead(comm);
                          if (p == NCCL_PROTO_SIMPLE) netOverhead *= 3;
                        }
                        intraLat = std::max(intraLat, netOverhead);
                        int nInterSteps = nNodes == 1 ? 0 : coll == ncclFuncAllReduce ? 2*(nNodes-1) : nNodes-1;
                        comm->latencies[coll][a][p] += (nsteps-nInterSteps)*intraLat + nInterSteps*interLat;
                      }
                    } else if (a == NCCL_ALGO_TREE) {
                      if (coll == ncclFuncAllReduce) {
                        comm->latencies[coll][a][p] +=
                          2 * ((nRanks/nNodes-1) * intraLat + log2i(nNodes) * interLat);
                      }
                    }
        int protoEnable[NCCL_NUM_FUNCTIONS*NCCL_NUM_PROTOCOLS];
        int algoEnable[NCCL_NUM_FUNCTIONS*NCCL_NUM_ALGORITHMS];
        const char *protoStr = ncclGetEnv("NCCL_PROTO");
        if (protoStr) {
            NCCLCHECK(parseList(protoStr, ncclFuncStr, NCCL_NUM_FUNCTIONS, ncclProtoStr, NCCL_NUM_PROTOCOLS, protoEnable));
        const char *algoStr = ncclGetEnv("NCCL_ALGO");
        if (algoStr) {
            NCCLCHECK(parseList(algoStr, ncclFuncStr, NCCL_NUM_FUNCTIONS, ncclAlgoStr, NCCL_NUM_ALGORITHMS, algoEnable));
        comm->threadThresholds[][]=
    mode = ncclLaunchModeParallel;
    modeOld = __atomic_exchange_n(&ncclParamLaunchMode, mode, __ATOMIC_RELAXED);
    devCommSetup(comm)
        comm->devComm = &devCommAndChans->comm;
    bootstrapIntraNodeBarrier
```

algbw和overhead还要根据数据量调整，在原语执行时，具体看ncclTopoGetAlgoTime

上面的时延是使用所有channel的时延，在原语执行时，根据每个线程的处理的数据量，可能缩小nChannel，应该降低带宽

10MiB数据量，理论时延us

```
===================================================================================================================================================================================
no   name           commId               rank0 comm           nranks   nNodes   Dev  localRanks nchannels  coll           min_size     max_size     algo   proto    nc_used  count 
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
1    -              0xc3fc0f81d9ec9fe3   0x55d738795150       16       2        0    8          16         Broadcast      41943040     41943040     RING   LL128    16       5     
                                                                                                           Reduce         41943040     41943040     RING   LL128    16       3     
                                                                                                           AllGather      2621440      2621440      RING   SIMPLE   16       3     
                                                                                                           ReduceScatter  2621440      2621440      RING   SIMPLE   16       3     
                                                                                                           AllReduce      41943040     41943040     RING   LL128    16       3     
===================================================================================================================================================================================

78+41943040/1000/293.6=220.857773
78+41943040/1000/293.6
70+41943040/1000/409.6=172.399994
70+41943040/1000/409.6
75.2+41943040/1000/156.6=343.058289
```

event测得实际时延，大2个量级

```
ncclCommInitRank time = 43945.258 ms
ncclBroadcast time = 5.733 ms
ncclReduce time = 12.164 ms
ncclAllGather time = 4.178 ms
ncclReduceScatter time = 11.315 ms
ncclAllReduce time = 13.018 ms
```

# ncclPrepareTasks

Collective Networks，NVLink Shared​​是两项加速技术，应该是没有启用？

任务按trafficBytes(字节数的倍数)大小排队

group depth==0时，开始通信

一个group里的所有task按原语，reduce op，数据类型分队

队列内，trafficBytes接近的task合并，选择在使用所有channel下时延最短的算法和协议（微调），根据数据量可能缩小nChannels和线程数，确定nChannel和nWarps

根据算法和协议确定kernel

合并的task又拆分，参数使用合并的参数，又加入collTaskQueue

```
typedef enum {
  ncclFuncBroadcast = 0,
  ncclFuncReduce = 1,
  ncclFuncAllGather = 2,
  ncclFuncReduceScatter = 3,
  ncclFuncAllReduce = 4,
  ncclFuncSendRecv = 5,
  ncclFuncSend = 6,
  ncclFuncRecv = 7,
  ncclNumFuncs = 8
} ncclFunc_t;

struct ncclInfo {
  ncclFunc_t coll;
  const char* opName;
  // NCCL Coll Args
  const void* sendbuff;
  void* recvbuff;
  size_t count;
  ncclDataType_t datatype;
  ncclRedOp_t op;
  int root; // peer for p2p operations
  ncclComm_t comm;
  cudaStream_t stream;
  // Algorithm details
  int chunkSteps;
  int sliceSteps;
};

struct ncclTaskColl {
  size_t count;
  ncclDataType_t datatype;
  size_t trafficBytes;//count*elementSize*ncclFuncTrafficPerByte ???
  int32_t nMaxChannels:8;//gridsize
  int32_t nWarps:8;//blocksize
  int32_t algorithm:8, protocol:8;
  uint32_t isCollnet:1, isNvls:1;
  uint32_t devFuncId:30;

struct ncclKernelPlanner {
  struct ncclTaskCollSorter collSorter;
  //group内的所有任务
  struct ncclIntruQueue<struct ncclTaskColl, &ncclTaskColl::next> collTaskQueue;
  //相同类型可以合并为组，按大小分组设置参数

struct ncclTaskCollSorter {
  struct ncclTaskColl* head;
  struct ncclTaskColl* tail;
  int binEdge;
  struct ncclTaskColl** bins[BinCount];//按数据量分桶
};

__thread struct ncclComm* ncclGroupCommHead = nullptr;//comm
__thread struct ncclIntruQueue<struct ncclAsyncJob, &ncclAsyncJob::next> ncclAsyncJobs;
__thread struct ncclGroupJob ncclGroupJobMain;
__thread struct ncclGroupJob *ncclGroupJobMainPtr = NULL;

struct ncclGroupJob {
  struct ncclAsyncJob base;
  struct ncclComm **groupCommHeadPtr;
  struct ncclIntruQueue<struct ncclAsyncJob, &ncclAsyncJob::next> *asyncJobsPtr;

ncclBroadcast
    struct ncclInfo info = { ncclFuncBroadcast, "Broadcast",
        sendbuff, recvbuff, count, datatype, ncclSum, root, comm, stream, /* Args */
        BROADCAST_CHUNKSTEPS, BROADCAST_SLICESTEPS };
    ncclEnqueueCheck(&info)
        ncclGroupStartInternal()
            ncclGroupDepth++;
        taskAppend
            struct ncclKernelPlanner *planner = &comm->planner;
            if (info->coll == ncclFuncSend || info->coll == ncclFuncRecv) {
            } else {
                ncclGroupCommJoin(info->comm);
                    struct ncclComm** pp = &ncclGroupCommHead;
                    comm->groupNext = *pp;
                    *pp = comm;
                    ncclGroupBlocking = comm->config.blocking;//1
                t->func = info->coll;
                t->sendbuff = info->sendbuff;
                t->recvbuff = info->recvbuff;
                t->count = info->count;//元素数量
                t->root = info->root;
                t->datatype = info->datatype;
                if (t->func == ncclFuncAllGather || t->func == ncclFuncBroadcast) {
                    t->count *= elementSize;
                    t->datatype = ncclInt8;
                    elementSize = 1;    
                }
                t->trafficBytes = t->count*elementSize*ncclFuncTrafficPerByte(t->func, comm->nRanks);
                  switch (func) {
                      case ncclFuncAllReduce: return 2;
                      case ncclFuncAllGather: return nRanks;
                      case ncclFuncReduceScatter: return nRanks;
                      default: return 1;
                //？？
                planner->nTasksColl += 1;
                ncclTaskCollSorterInsert(&planner->collSorter, t, t->trafficBytes);
                    int bin = u32fpEncode(std::min(MaxSize, size)>>UnitLog2, BitsPerPow2);
                    bin = BinCount-1 - bin; // descending bin
                    //任务按大小降序排列（大任务优先执行）
                    if (me->bins[bin] == nullptr) {
                        if (me->binEdge <= bin) {
                          me->binEdge = bin+1;
                          me->bins[bin] = me->tail ? &me->tail->next : &me->head;
                          //记录尾巴的下一个或者头部
                          me->tail = x;
                        } else {
                          // Find successor non-empty bin after this one.
                          int succ = bin+1;
                          while (me->bins[succ] == nullptr) succ++;
                          // What was our successor's head's previous is now our head's previous.
                          me->bins[bin] = me->bins[succ];
                          me->bins[succ] = &x->next;//前一个bin的末尾节点的next的地址，解引用就是bin的首节点
                        }
                    }
                    // Push a new head for this bin.//头插
                    x->next = *me->bins[bin];
                    *me->bins[bin] = x;
        ncclGroupEndInternal()
            if ((--ncclGroupDepth) > 0) goto exit;
            ncclGroupJobMain.groupCommHeadPtr = &ncclGroupCommHead;
            ncclGroupJobMain.asyncJobsPtr = &ncclAsyncJobs;
            ncclGroupJobMainPtr = &ncclGroupJobMain;
            /* blocking group */
            groupLaunch(&ncclGroupJobMainPtr->base, internalSimInfoPtr)
                struct ncclGroupJob *gjob = (struct ncclGroupJob*) job_;
                struct ncclComm *groupCommHeadMain = *gjob->groupCommHeadPtr;//comm 
                struct ncclIntruQueue<struct ncclAsyncJob, &ncclAsyncJob::next> *asyncJobsMain = gjob->asyncJobsPtr;
                asyncJobLaunch(asyncJobsMain, groupAbortFlag
                    if (!ncclIntruQueueEmpty(asyncJobsMain)) {
                struct ncclComm* cliqueHead = groupCommHeadMain;
                comm = cliqueHead;
                // Called once per ncclGroup to organize the user submitted tasks in
                // comm->planner so that they can be peeled off into plans.
                ncclPrepareTasks(comm, algoNeedConnect, &needConnect, simInfo)
                    task = ncclTaskCollSorterDequeueAll(&planner->collSorter);
                        struct ncclTaskColl* head = me->head;
                        if (head != nullptr) memset(me, 0, sizeof(*me));
                    struct ncclTaskColl* tasksByFnOpTy[ncclNumFuncs*ncclNumDevRedOps*ncclNumTypes];
                    //将单链表分类成多个链表，类内按从小到大排序
                    int fnOpTyIndices[ncclNumFuncs*ncclNumDevRedOps*ncclNumTypes];//记录类型
                    while (task != nullptr) {
                        struct ncclTaskColl* next = task->next;
                        int index = ((int)task->func*ncclNumDevRedOps + (int)task->opDev.op)*ncclNumTypes + (int)task->datatype;
                        // Add to set of (fn,op,ty) indices on first occurrence
                        if (tasksByFnOpTy[index] == nullptr) fnOpTyIndices[fnOpTyCount++] = index;
                        // Add to LIFO for this (fn,op,ty)
                        task->next = tasksByFnOpTy[index];
                        tasksByFnOpTy[index] = task;//头插
                        // Next task
                        task = next;
                    }
                    struct ncclIntruQueue<struct ncclTaskColl, &ncclTaskColl::next> collBins[2][2] = {};
                    for (int cursor=0; cursor < fnOpTyCount; cursor++) {
                        struct ncclTaskColl* aggBeg = tasksByFnOpTy[fnOpTyIndices[cursor]];
                        getCollNetSupport(comm, aggBeg, &collNetSupport)
                        nvlsSupport = comm->nvlsSupport && (ncclNvlsSupported(aggBeg->opDev.op, aggBeg->datatype) || aggBeg->func == ncclFuncAllGather)
                        //同一类这两个参数是一样的？
                        int nTasksPerChannel = divUp(comm->planner.nTasksColl, comm->nChannels);
                        do {
                            struct ncclTaskColl* aggEnd = aggBeg->next;
                            struct ncclTaskColl agg = *aggBeg;
                            // We aggregate operations that are within 4X size of each other.
                            while (aggEnd != nullptr && aggEnd->trafficBytes < 4*aggBeg->trafficBytes) {
                                agg.count += aggEnd->count;
                                agg.trafficBytes += aggEnd->trafficBytes;
                                aggEnd = aggEnd->next;
                            }
                            //合并相同类的大小接近的任务
                            getAlgoInfo(comm, &agg, collNetSupport, nvlsSupport, nTasksPerChannel, simInfo));
                            //选择最快的算法和协议
                                size_t nBytes = ncclTypeSize(info->datatype)*ncclFuncMaxSendRecvCount(info->func, comm->nRanks, info->count);
                                      return func == ncclFuncAllGather || func == ncclFuncReduceScatter ? nRanks*count : count;
                                      //max(send,recv)
                                updateCollCostTable(comm, info, nBytes, collNetSupport, nvlsSupport, numPipeOps, (float **)collCostTable));
                                //获得各种时延信息
                                    for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
                                        for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
                                            ncclTopoGetAlgoTime(comm, info->func, a, p, nBytes, numPipeOps, &table[a][p]));
                                                float bw = comm->bandwidths[coll][algorithm][protocol];
                                                float lat = comm->latencies[coll][algorithm][protocol];
                                                int logSize = log2i(nBytes>>6);//MB
                                                if (algorithm == NCCL_ALGO_TREE && coll == ncclFuncAllReduce && logSize >= 0 && logSize < 23) bw *= treeCorrectionFactor[protocol][logSize];
                                                if (algorithm == NCCL_ALGO_RING && protocol == NCCL_PROTO_SIMPLE && comm->nNodes > 1
                                                  && coll == ncclFuncAllReduce && nBytes/(comm->nChannels*comm->nRanks) >= 64) {
                                                    lat *= comm->minCompCap < 80 ? 1.9 : 1.4; // Plateau effect of ring
                                                }
                                                // Tree pipelining saves latency in aggregation cases
                                                int latCount = algorithm == NCCL_ALGO_RING ? numPipeOps : DIVUP(numPipeOps, NCCL_MAX_DEV_WORK_BATCH_COLLS);
                                                //任务的数量
                                                *time = lat * latCount + nBytes / (1000 * bw);
                                topoGetAlgoInfo(comm, info, nBytes, (float **)collCostTable, simInfo));
                                //选择最快的算法和协议
                                    for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
                                        for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
                                          if (table[a][p] == NCCL_ALGO_PROTO_IGNORE) continue;
                                          if (table[a][p] >= 0.0 && table[a][p] < minTime) {
                                            algorithm = a;
                                            protocol = p;
                                            minTime = table[a][p];
                                          }
                                        }
                                    }
                                    info->algorithm = algorithm;
                                    info->protocol = protocol;
                                    int nc = comm->nChannels;
                                    int nt = comm->maxThreads[info->algorithm][info->protocol];
                                    int threadThreshold = comm->threadThresholds[info->algorithm][info->protocol];
                                    while (nBytes < nc * nt * threadThreshold) {
                                      if (nc >= 2) nc--;
                                      else break;
                                    }     
                                    //根据数据量降低nChannels               
                                    while (nBytes < nc * nt * threadThreshold) {
                                      if (nt % 128 == 0) nt /= 2;
                                      else break;
                                    }
                                    //降低线程数量
                                    info->nMaxChannels = nc;
                                    info->nWarps = nt/WARP_SIZE;
                            agg.devFuncId = ncclDevFuncId(agg.func, agg.opDev.op, agg.datatype, agg.algorithm, agg.protocol);
                                return ncclDevFuncRowToId[row];
                            do {
                                struct ncclTaskColl* next = aggBeg->next;
                                aggBeg->algorithm = agg.algorithm;
                                aggBeg->protocol = agg.protocol;
                                if (aggBeg->protocol == NCCL_PROTO_LL) aggBeg->trafficBytes *= 4;
                                aggBeg->nMaxChannels = agg.nMaxChannels;
                                aggBeg->nWarps = agg.nWarps;
                                aggBeg->devFuncId = agg.devFuncId;
                                aggBeg->isCollnet = isCollnet;
                                aggBeg->isNvls = isNvls;
                                ncclIntruQueueEnqueue(&collBins[isCollnet][isNvls], aggBeg);
                                //按大小分组设置参数，重新插回去，没有合并
                                aggBeg = next;
                            } while (aggBeg != aggEnd);
                        } while (aggBeg != nullptr);
                        //处理完一类
                    for (int isCollnet=0; isCollnet <= 1; isCollnet++) {
                        for (int isNvls=0; isNvls <= 1; isNvls++) {
                          ncclIntruQueueTransfer(&planner->collTaskQueue, &collBins[isCollnet][isNvls]);
                        }
                    }
```

# scheduleCollTasksToPlan

一个task生成一个work，加在collWorkQueue

一个plan可以容纳多个task，它们的总trafficBytes均摊到它们的总channel，确定每个task占据的channel

每个work确定每个channel的元素数量，根据不同channel的元素数量生成不同proxyop配置（pattern）

```
ncclFuncBroadcast->ring->ncclPatternPipelineFrom
ncclFuncReduce->ring->ncclPatternPipelineTo
ncclFuncReduceScatter->ring->ncclPatternRing
ncclFuncAllGather->ring->ncclPatternRing
ncclFuncAllReduce->ring->ncclPatternRingTwice
ncclFuncAllReduce->tree->ncclPatternTreeUpDown
```

对work的每个channel，生成batch，加在planner.chan[id]->workBatchQueue

对work的每个channel，如果当前rank需要发送或接收数据，且对方跨机，生成proxyop，加在planner.chan[id]->proxyOpQueue

proxyOp.opCount = plan->collOpCount++，一个task的所有proxyop的opcount相同

```
struct ncclTaskColl {
  size_t count;
  ncclDataType_t datatype;
  size_t trafficBytes;//count*elementSize*ncclFuncTrafficPerByte ???
  int32_t nMaxChannels:8;//gridsize
  int32_t nWarps:8;//blocksize
  int32_t algorithm:8, protocol:8;
  uint32_t isCollnet:1, isNvls:1;
  uint32_t devFuncId:30;

struct ncclKernelPlanner {
  struct ncclTaskCollSorter collSorter;
  //group内的所有任务
  struct ncclIntruQueue<struct ncclTaskColl, &ncclTaskColl::next> collTaskQueue;
  //相同类型可以合并为组，按大小分组设置参数
  struct ncclIntruQueue<struct ncclWorkList, &ncclWorkList::next> collWorkQueue;
  struct WipPlan {
    struct Channel {
      struct {
        int workBytes; // Sum size of work metadata referenced by this batch.
        int nP2ps; // Number of p2p works in this batch
        int p2pRounds[NCCL_MAX_DEV_WORK_P2P_PER_BATCH]; // which rounds are present in this batch.
      } wipBatch; // work-in-progress batch which will be next tail of workBatchQueue
      int nWorkBatchesP2p; // number of p2p batches for this channel.
      struct ncclIntruQueue<struct ncclWorkBatchList, &ncclWorkBatchList::next> workBatchQueue;//有数据传输
      struct ncclIntruQueue<struct ncclProxyOp, &ncclProxyOp::enqNext> proxyOpQueue;//有跨机数据传输
    } channels[MAXCHANNELS];
  } wipPlan;

struct alignas(16) ncclWorkList {
  struct ncclWorkList* next;
  enum ncclDevWorkType workType;
  int size; // Size of struct following this node//work数据的大小，紧接在ncclWorkList后面
  // ncclDevWorkColl, ncclDevWorkColLReg, ncclDevWorkP2p[]...
};

struct alignas(16) ncclDevWorkColl {
  // Running on channels [channelLo..channelHi], hi is inclusive.
  //   nChannels == (channelHi - channelLo) + 1
  uint32_t channelLo:8, channelHi:8;

struct ncclWorkBatchList {
  struct ncclWorkBatchList* next;
  struct ncclDevWorkBatch batch;
};

struct alignas(16) ncclDevWorkBatch {
  union {
    struct {
      // nextExtends: should next one be merged into this one.
      // nextJump=0: end of this channel's batch list
      // nextJump>0: batches[thisIndex+nextJump] is next batch in this list
      uint32_t nextJump:14, nextExtends:1;
      uint32_t workType:2, funcId:15;
    };
    // Unioning bitfields with underlying type hints compiler to emit the best
    // SASS LD/ST accesses.
    uint32_t flags;
  };
  // Rolling offset in fifo where this batch's work structs begin
  uint32_t offsetBase;//sizeof(ncclDevKernelArgs) + (workNode->size)s;//它所属的work的数据到kernelarg开头的距离
  // Set of relative offsets from offsetBase for this channel's subset of the batch:
  // For each bit index i in offsetMask, find work at fifo offset: offsetBase + i*sizeof(WorkStructType)
  uint64_t offsetBitset;
};

struct ncclKernelPlan {
  int collOpCount; // Number of collectives in this plan.
  struct ncclIntruQueue<struct ncclTaskColl, &ncclTaskColl::next> collTaskQueue;//从planner取
  size_t workBytes; // Sum size of all work (in the fifo) in bytes.// workQueue[...]->size
  struct ncclIntruQueue<struct ncclWorkList, &ncclWorkList::next> workQueue;//从planner取
  int nWorkBatches; // Number of work batches.
  uint64_t channelMask; // bitset of which channels are present//grid size
  int threadPerBlock;//blocksize
  void *kernelFn;//函数指针

struct ncclProxyOp {
  struct ncclProxyConnection* connection;
  ssize_t nbytes;
  uint64_t opCount;//id==plan.collOpCount++>>1

groupLaunch
    ncclPrepareTasks
    comm = groupCommHeadMain;
    ncclTasksRegAndEnqueue(comm)
        task = ncclIntruQueueHead(&planner->collTaskQueue);
        while (task != nullptr) {
            devWork.sendbuff = (void*)task->sendbuff;
            workNode->size = sizeof(struct ncclDevWorkColl);
            memcpy((void*)(workNode+1), (void*)&devWork, workNode->size);
            ncclIntruQueueEnqueue(&planner->collWorkQueue, workNode);
            task = task->next;
        }
    doLaunches(groupCommHeadMain)
        ncclLaunchPrepare
            do {
                plan->reclaimer.fn = reclaimPlan;//释放plann相关的结构体，头插回pool，所以会复用地址
                if (planner->nTasksColl != 0) {
                    scheduleCollTasksToPlan(comm, plan, &budget)
                    //planner里的task,work转到plan，多个work可能形成batch，batch是保存在planner
                        size_t workBytes = 0;
                        struct ncclTaskColl* task = ncclIntruQueueHead(&planner->collTaskQueue);
                        struct ncclWorkList* workNode = ncclIntruQueueHead(&planner->collWorkQueue);
                        while (task != nullptr) {
                          int nBatches = divUp(nPlanColls, 4); // Rough guess: 4 colls per batch.
                          if (!testBudget(budget, nBatches, workBytes + workNode->size)) goto plan_full;
                          nPlanColls += 1;//一个plan里面有多个work
                          workBytes += workNode->size;
                          int kind = 2*task->isCollnet + task->isNvls;
                          trafficBytes[kind] += std::max(MinTrafficPerChannel, task->trafficBytes);//多个task的数据量
                          nChannels[kind] += task->nMaxChannels;//多个task的channel
                          nChannels[kind] = std::min(nChannels[kind], nMaxChannels[kind]);
                          task = task->next;
                          workNode = workNode->next;
                        }
                        while (nPlanColls!=0 && !ncclIntruQueueEmpty(&planner->collTaskQueue)) {
                        //把trafficBytes划分到不同channel
                            struct ncclTaskColl* task = ncclIntruQueueHead(&planner->collTaskQueue);
                            struct ncclWorkList* workNode = ncclIntruQueueHead(&planner->collWorkQueue);
                            struct ncclDevWorkColl* devWork = (struct ncclDevWorkColl*)(workNode+1);
                            int kind = 2*task->isCollnet + task->isNvls;
                            if (kind != kindPrev) {
                              trafficPerChannel = std::max<size_t>(MinTrafficPerChannel, trafficBytes[kind]/nChannels[kind]);
                              //channel均分总trafficBytes
                              kindPrev = kind;
                              channelId = 0;
                              currentTraffic = 0;
                            }
                            ...
                            } else { // not task->isCollnet
                                size_t cellSize = divUp(divUp(MinTrafficPerChannel, (size_t)trafficPerByte), 16) * 16;
                                //一个channel最小能够承载的trafficBytes，可以多承载
                                size_t cells = divUp(task->count*elementSize, cellSize);
                                //cell的数量
                                size_t cellsPerChannel = std::min(cells, divUp(trafficPerChannel, trafficPerCell));
                                //一个chnnel可以有多个cell
                                if (channelId+1 == nMaxChannels[kind]) { // On last channel everything goes to "lo"
                                    cellsLo = cells;
                                } else {
                                    cellsLo = std::min(cells, divUp((trafficPerChannel-currentTraffic),trafficPerCell));
                                    //当前channel能够容纳的cell数量
                                int nMidChannels = (cells-cellsLo)/cellsPerChannel;
                                //需要新增的channel
                                size_t cellsHi = (cells-cellsLo)%cellsPerChannel;
                                //需要再占用不足一条channel的cell数量
                                int nChannels = (cellsLo!=0 ? 1 : 0) + nMidChannels + (cellsHi!=0 ? 1 : 0);
                                //总共占用nChannel
                                if (cellsHi == 0 && nMidChannels != 0) {
                                    cellsHi = cellsPerChannel;
                                    nMidChannels -= 1;
                                }
                                //最后一个channel
                                if (cellsLo == 0) { // Least channel skipped. Make the next channel the new least.
                                    channelId += 1;
                                    if (nMidChannels == 0) { cellsLo = cellsHi; cellsHi = 0; }
                                    else { cellsLo = cellsPerChannel; nMidChannels -= 1; }
                                }
                                //第一个channel
                                size_t countMid = nMidChannels!=0 ? cellsPerChannel*elementsPerCell : 0;
                                //中间一个channel的trafficBytes
                                size_t countLo = cellsLo*elementsPerCell;
                                size_t countHi = cellsHi*elementsPerCell;
                                nChannels = (countLo!=0 ? 1 : 0) + nMidChannels + (cellsHi!=0 ? 1 : 0);
                                devWork->channelLo = channelId;//0
                                devWork->channelHi = channelId + nChannels-1;//15
                                devWork->cbd.countLo = countLo;
                                devWork->cbd.countMid = countMid;
                                devWork->cbd.countHi = countHi;
                                //每个channel的元素数量
                                size_t globalBytesPerElement = elementSize*ncclFuncMaxSendRecvCount(task->func, comm->nRanks, 1);
                                struct ncclProxyOp proxyOpLo, proxyOpMid, proxyOpHi;
                                if (countLo != 0) {
                                    calcCollChunking(comm, task, /*nChannels=*/1, globalBytesPerElement*countLo, &chunkSize, &directFlags, &proxyOpLo));
                                    //根据不同的数据量生成proxyop配置
                                        switch (info->func) {
                                            case ncclFuncBroadcast:
                                            pattern = info->algorithm == NCCL_ALGO_TREE ? ncclPatternTreeDown : ncclPatternPipelineFrom;
                                            break;
                                            case ncclFuncReduce:
                                            pattern = info->algorithm == NCCL_ALGO_TREE ? ncclPatternTreeUp : ncclPatternPipelineTo;
                                            break;
                                            case ncclFuncReduceScatter:
                                            pattern =
                                              ncclPatternRing;
                                            break;
                                            case ncclFuncAllGather:
                                            pattern =
                                              ncclPatternRing;
                                            break;
                                            case ncclFuncAllReduce:
                                            pattern =
                                              info->algorithm == NCCL_ALGO_TREE ? ncclPatternTreeUpDown :
                                              ncclPatternRingTwice;
                                        proxyOp->nsteps = nstepsPerLoop * nLoops * chunkSteps;
                                        proxyOp->chunkSize = chunkSize;
                                        proxyOp->protocol = info->protocol;
                                        proxyOp->algorithm = info->algorithm;
                                        proxyOp->pattern = pattern;//算法的实现
                                        proxyOp->coll = info->func;
                                    devWork->cbd.chunkGrainsLo = chunkSize/grainSize;
                                }
                                if (countHi != 0) {
                                    NCCLCHECK(calcCollChunking(comm, task, /*nChannels=*/1, globalBytesPerElement*countHi, &chunkSize, &directFlags, &proxyOpHi));
                                    devWork->cbd.chunkGrainsHi = chunkSize/grainSize;
                                }
                                if (nMidChannels != 0) {
                                    NCCLCHECK(calcCollChunking(comm, task, /*nChannels=*/1, globalBytesPerElement*countMid, &chunkSize, &directFlags, &proxyOpMid));
                                    devWork->cbd.chunkGrainsMid = chunkSize/grainSize;
                                }
                                // Update the current channel and vacant traffic budget.
                                if (countHi != 0) {
                                    channelId += nChannels-1;
                                    currentTraffic = cellsHi*elementsPerCell*trafficPerElement;
                                } else if (nMidChannels != 0) {
                                    channelId += nChannels;
                                    currentTraffic = 0;
                                } else {
                                    currentTraffic += cellsLo*elementsPerCell*trafficPerElement;
                                }
                                if (currentTraffic >= trafficPerChannel && channelId+1 != nMaxChannels[kind]) {
                                    channelId += 1;
                                    currentTraffic = 0;
                                }
                                uint64_t proxyOpId = uint64_t(plan->collOpCount++)<<1 | 0;
                                //一个任务里不同channel的proxyop.id相同
                                for (int c=devWork->channelLo; c <= (int)devWork->channelHi; c++) {
                                    proxyOp->channelId = c;
                                    proxyOp->opCount = proxyOpId;
                                    proxyOp->task.coll = task;
                                    addWorkBatchToPlan(comm, plan, c, workNode->workType, task->devFuncId, plan->workBytes)
                                        ncclKernelPlanner::WipPlan::Channel* chan = &comm->planner.wipPlan.channels[channelId];
                                        bool newBatch = (chan->workBatchQueue.tail == nullptr);
                                        if (newBatch || extendBatch) {
                                            if (!newBatch) batch->nextExtends = extendBatch; 
                                            ncclIntruQueueEnqueue(&chan->workBatchQueue, batchNode);
                                            batch->offsetBase = workOffset;
                                            plan->nWorkBatches += 1;
                                    addProxyOpIfNeeded(comm, plan, proxyOp)
                                        ncclProxySaveOp(comm, op, &needed)
                                            switch (op->pattern) {
                                            case ncclPatternRing:
                                            case ncclPatternRingTwice:
                                            case ncclPatternPipelineFrom:
                                            case ncclPatternPipelineTo: {
                                              struct ncclRing* ring = &channel->ring;
                                              if (NeedProxy(proxyRecv, op->pattern, op->root, ring, comm->nRanks)) {
                                                  //需要发送或接收数据就需要proxy
                                                  //bcast: 作为recv，root不需要接受，不是root，要proxy去接收；作为send，下一个不是root，就是有下一个，要proxy去发送
                                                  if (pattern == ncclPatternRing || pattern == ncclPatternRingTwice) return true;
                                                    const int myrank = 0, nextrank = 1, prevrank = nranks-1;
                                                    int index = pattern == ncclPatternPipelineFrom ?
                                                      /*                            no recv /  no send    if root = */
                                                      /* bcast  */ (type == proxyRecv ?   myrank : nextrank ):
                                                      /* reduce */ (type == proxyRecv ? prevrank :   myrank );
                                                    int rank = ring->userRanks[index];
                                                    return (root != rank);
                                                  SaveProxy(comm, channel, proxyRecv, ring->prev, op, 0, justInquire));
                                                  //和某个peer的connector
                                                    peerComm = channel->peers[peer];
                                                    connector = type == proxyRecv ? peerComm->recv+connIndex : peerComm->send+connIndex;
                                                    if (connector->proxyConn.proxyProgress == NULL) return ncclSuccess;
                                                    //跨机才需要proxy
                                                    if (justInquire) *justInquire = true;
                                                    else {
                                                        op->peer = peer;
                                                        NCCLCHECK(ncclLocalOpAppend(comm, &connector->proxyConn, op));
                                                    }
                                              }
                                              if (NeedProxy(proxySend, op->pattern, op->root, ring, comm->nRanks)) {
                                                NCCLCHECK(SaveProxy(comm, channel, proxySend, ring->next, op, 0, justInquire));
                                              }
                                            } break;
                                            case ncclPatternTreeUp:
                                            case ncclPatternTreeDown:
                                            case ncclPatternTreeUpDown: {
                                              if (op->pattern != ncclPatternTreeDown) { // Tree up
                                                struct ncclTree* tree = &channel->tree;
                                                for (int i=0; i<NCCL_MAX_TREE_ARITY; i++) {
                                                  NCCLCHECK(SaveProxy(comm, channel, proxyRecv, tree->down[i], op, 0, justInquire));
                                                }
                                                NCCLCHECK(SaveProxy(comm, channel, proxySend, tree->up, op, 0, justInquire));
                                              }
                                              if (op->pattern != ncclPatternTreeUp) { // Tree down
                                                struct ncclTree* tree = &channel->tree;
                                                for (int i=0; i< NCCL_MAX_TREE_ARITY; i++) {
                                                  NCCLCHECK(SaveProxy(comm, channel, proxySend, tree->down[i], op, 0, justInquire));
                                                }
                                                NCCLCHECK(SaveProxy(comm, channel, proxyRecv, tree->up, op, 0, justInquire));
                                              }
                                            } break;
                                        if (needed) {//跨机
                                            ncclIntruQueueEnqueue(&comm->planner.wipPlan.channels[op->channelId].proxyOpQueue, q)
                            plan->channelMask |= (2ull<<devWork->channelHi) - (1ull<<devWork->channelLo);//通过减法生成连续通道的掩码
                            plan->threadPerBlock = std::max(plan->threadPerBlock, task->nWarps*WARP_SIZE);
                            if (!plan->kernelSpecialized) {
                              plan->kernelFn = ncclDevKernelForFunc[task->devFuncId];
                                  extern void* const ncclDevKernelList[] = {
                                      (void*)ncclDevKernel_Broadcast_RING_LL
                              plan->kernelSpecialized = ncclDevKernelForFuncIsSpecialized[task->devFuncId];
                            }
                            ncclIntruQueueDequeue(&planner->collTaskQueue);
                            ncclIntruQueueDequeue(&planner->collWorkQueue);
                            nPlanColls -= 1;
                            planner->nTasksColl -= 1;
                            ncclIntruQueueEnqueue(&plan->collTaskQueue, task);
                            ncclIntruQueueEnqueue(&plan->workQueue, workNode);
                            plan->workBytes += workNode->size;
                if (planner->nTasksColl == 0 && planner->nTasksP2p != 0) {
                    scheduleP2pTasksToPlan
```

```
DEFINE_ncclDevKernel(Broadcast_RING_LL, ncclFuncBroadcast, FuncCopy, int8_t, NCCL_ALGO_RING, NCCL_PROTO_LL, 342)

#define DEFINE_ncclDevKernel(suffix, coll, redop, ty, algo, proto, specializedFnId) \
  __global__ void ncclDevKernel_##suffix(ncclDevKernelArgs4K NCCL_GRID_CONSTANT const args4K) { \
    ncclKernelMain<specializedFnId, RunWorkBatch<coll, ty, redop<ty>, algo, proto>>(&args4K.args); \
  }
```

# finishPlan

planner.channels[].workBatchQueue，每次遍历一遍所有channel，取出队头，加在kernelarg后面，直到空

plan.proxyOpQueue=planner.wipPlan.channels[].proxyOpQueue，按id排序

把所有tasks划分成plans，加入planQueue

```
struct ncclKernelPlanner {
  struct ncclTaskCollSorter collSorter;
  //group内的所有任务
  struct ncclIntruQueue<struct ncclTaskColl, &ncclTaskColl::next> collTaskQueue;
  //相同类型可以合并为组，按大小分组设置参数
  struct ncclIntruQueue<struct ncclWorkList, &ncclWorkList::next> collWorkQueue;
  struct WipPlan {
    struct Channel {
      struct {
        int workBytes; // Sum size of work metadata referenced by this batch.
        int nP2ps; // Number of p2p works in this batch
        int p2pRounds[NCCL_MAX_DEV_WORK_P2P_PER_BATCH]; // which rounds are present in this batch.
      } wipBatch; // work-in-progress batch which will be next tail of workBatchQueue
      int nWorkBatchesP2p; // number of p2p batches for this channel.
      struct ncclIntruQueue<struct ncclWorkBatchList, &ncclWorkBatchList::next> workBatchQueue;//有数据传输
      struct ncclIntruQueue<struct ncclProxyOp, &ncclProxyOp::enqNext> proxyOpQueue;//有跨机数据传输
    } channels[MAXCHANNELS];
  } wipPlan;

struct ncclKernelPlan {
  int collOpCount; // Number of collectives in this plan.
  struct ncclIntruQueue<struct ncclTaskColl, &ncclTaskColl::next> collTaskQueue;//从planner取
  size_t workBytes; // Sum size of all work (in the fifo) in bytes.// workQueue[...]->size
  struct ncclIntruQueue<struct ncclWorkList, &ncclWorkList::next> workQueue;//从planner取
  int nWorkBatches; // Number of work batches.
  uint64_t channelMask; // bitset of which channels are present//grid size
  int threadPerBlock;//blocksize
  void *kernelFn;//函数指针
  enum ncclDevWorkStorageType workStorageType;//ncclDevWorkStorageTypeArgs
  struct ncclDevKernelArgs* kernelArgs;
  size_t kernelArgsSize;//sizeof(kernelarg)+nWorkBatches*sizeof(struct ncclDevWorkBatch)+workBytes
  struct ncclIntruQueue<struct ncclProxyOp, &ncclProxyOp::enqNext> proxyOpQueue;//从plan取，按id排序
  struct ncclIntruQueue<struct ncclKernelPlan, &ncclKernelPlan::next> planQueue;
  bool persistent; // aka captured in a graph

struct alignas(16) ncclDevKernelArgs {
  struct ncclDevComm* comm;
  uint64_t channelMask;
  enum ncclDevWorkStorageType workStorageType;
  uint32_t workMask;//~0u
  void* workBuf;
  // A channel's first batch is at `blockIdx.x`. Use `nextJump` to follow rest of list.
  // struct ncclDevWorkBatch batches[];
};
//这两个连续储存
struct alignas(16) ncclDevWorkBatch {
  union {
    struct {
      // nextExtends: should next one be merged into this one.
      // nextJump=0: end of this channel's batch list
      // nextJump>0: batches[thisIndex+nextJump] is next batch in this list
      uint32_t nextJump:14, nextExtends:1;
      uint32_t workType:2, funcId:15;
    };
    // Unioning bitfields with underlying type hints compiler to emit the best
    // SASS LD/ST accesses.
    uint32_t flags;
  };
  // Rolling offset in fifo where this batch's work structs begin
  uint32_t offsetBase;//sizeof(ncclDevKernelArgs) + (workNode->size)s;//它所属的work的数据到kernelarg开头的距离
  // Set of relative offsets from offsetBase for this channel's subset of the batch:
  // For each bit index i in offsetMask, find work at fifo offset: offsetBase + i*sizeof(WorkStructType)
  uint64_t offsetBitset;
};

groupLaunch
    doLaunches(groupCommHeadMain)
        ncclLaunchPrepare
            do{
                finishPlan(comm, plan);
                    wipChannels = comm->planner.wipPlan.channels;
                    size_t batchBytes = plan->nWorkBatches*sizeof(struct ncclDevWorkBatch);
                    plan->workStorageType = ncclDevWorkStorageTypeArgs;
                    plan->kernelArgsSize = sizeof(struct ncclDevKernelArgs) + batchBytes;
                    plan->kernelArgsSize += (plan->workStorageType == ncclDevWorkStorageTypeArgs) ? workBytes : 0;
                    plan->kernelArgs->comm = comm->devComm;
                    uint64_t hasBatchMask = plan->channelMask;
                    struct ncclDevWorkBatch* batchPrev[MAXCHANNELS] = {}; // {0...}
                    struct ncclDevWorkBatch* batchZero = (struct ncclDevWorkBatch*)(plan->kernelArgs+1);
                    //记录所有batch
                    int batchIx = 0;
                    //每次遍历一遍所有channel，记录队头，直到空
                    while (hasBatchMask != 0) {
                        uint64_t tmpMask = hasBatchMask; // channels with a batch for this round.
                        do {
                            int c = popFirstOneBit(&tmpMask);
                            if (!ncclIntruQueueEmpty(&wipChannels[c].workBatchQueue)) {
                                struct ncclWorkBatchList* batchNode = ncclIntruQueueDequeue(&wipChannels[c].workBatchQueue);
                                if (batchPrev[c] != nullptr) {
                                    batchPrev[c]->nextJump = int(&batchZero[batchIx] - batchPrev[c]);
                                    //相同channel的上一个到下一个的距离
                                }
                                batchPrev[c] = &batchZero[batchIx];
                                batchZero[batchIx++] = batchNode->batch;
                            }
                            if (ncclIntruQueueEmpty(&wipChannels[c].workBatchQueue)) {
                            hasBatchMask ^= 1ull<<c;
                            }
                        } while (tmpMask != 0);
                    uint64_t headIds[MAXCHANNELS];
                    int nHeads = 0;
                    int channelUbound = 0;
                    for (int c=0; c < MAXCHANNELS; c++) {
                        struct ncclProxyOp* op = ncclIntruQueueHead(&wipChannels[c].proxyOpQueue);
                        headIds[c] = op ? op->opCount : uint64_t(-1);
                        if (op) nHeads += 1;
                        if (op) plan->hasProxyOps = true;
                        if (op) channelUbound = c+1;
                    }
                    while (nHeads != 0) {
                        int c = -1;
                        uint64_t minId = uint64_t(-1);
                        // Find channel with least proxy-op id. We store the heads[c]->opCount in
                        // headIds[c] to remove indirect loads from this loop.
                        for (int c1=0; c1 < channelUbound; c1++) {
                          uint64_t id = headIds[c1];
                          id = (id>>1 | id<<63); // Move tag bit to order collectives before p2p's
                          if (id < minId) { c = c1; minId = id; }
                        }
                        struct ncclProxyOp* op = ncclIntruQueueDequeue(&wipChannels[c].proxyOpQueue);
                        struct ncclProxyOp* opNext = ncclIntruQueueHead(&wipChannels[c].proxyOpQueue);
                        headIds[c] = opNext ? opNext->opCount : uint64_t(-1);
                        nHeads -= opNext ? 0 : 1;
                        ncclIntruQueueEnqueue(&plan->proxyOpQueue, op);//按id顺序加入
                    }
                if (plan->workBytes != 0) {
                    ncclIntruQueueEnqueue(&planner->planQueue, plan);
                    nPlans += 1;
                }
            } while (planner->nTasksColl + planner->nTasksP2p != 0);
            struct ncclKernelPlan* planHead = ncclIntruQueueHead(&planner->planQueue);
            planner->unlaunchedPlansHead = planHead;
```

# 启动kernel，唤醒proxyprogress

kernel参数：

```
plan.kernelarg
workbatches
works
```

1. 把works数据拷贝到workbatches后面，上面三者连续，是kernel参数，一个plan里的所有task一个kernel

2. launch kernel

3. comm->collOpCount += plan->collOpCount，op->opCount = (comm->collOpCount<<1) + oldId，变成全局id，保存所有proxyop到共享内存

4. 只唤醒一次proxyprogress

5. comm->opCount++;//plan的数量

6. 启动完所有的plan

单机input-(kernel)>buffer-(kernel)>output

多机input-(kernel)>buffer-(rdma)>buffer-(kernel)>output

user buffer和nccl buffer的数据交换由main thread通过kernel完成

网络通信由proxyprogress线程通过rdma完成

```
struct ncclComm {
  ncclNet_t* ncclNet;
  void* bootstrap;
  struct ncclProxyState* proxyState;
  struct ncclPeerInfo* peerInfo;
  struct ncclTopoSystem* topo;
  struct ncclTopoGraph graphs[NCCL_NUM_ALGORITHMS];
  struct ncclChannel channels[MAXCHANNELS];
  struct P2pSchedulePair { int sendRank; int recvRank; } *p2pSchedule;
  // Counter for tracking CUDA launches (P2P and collectives included)//plan的数量
  uint64_t opCount;
  // Collective operation counter//一个plan可以有多个task
  uint64_t collOpCount;

struct ncclProxyState {
  struct ncclSocket* listenSock;//peerProxyAddress socket，proxy监听
  struct ncclIpcSocket ipcSock;//peerProxyAddressUDS socket
  union ncclSocketAddress* peerAddresses;//peerProxyAddresses，proxy ip用于连接
  uint64_t *peerAddressesUDS; // cuMem API support (UDS)//peerProxyAddressesUDS
  struct ncclSocket* peerSocks;//连接proxy的socket
  struct ncclExpectedProxyResponse* expectedResponses;//等待proxy回复的链表
  struct ncclProxyProgressState progressState;
  struct ncclProxyOps* proxyOps;//[localrank],共享ProxyProgressState.ncclProxyOpsPool

struct ncclProxyOps {
  ncclProxyOpsPool* pool;//main thread 共享ProxyProgress.ncclProxyOpsPool
  ncclShmHandle_t handle;//共享内存的地址
  int count;//0，要处理的数量
  int freeOp;//-1，下一个空闲
  int nextOps;//-1，下一个要处理的
  int nextOpsEnd;//-1，要处理的最后一个，尾插
};
//每次唤醒完就恢复初始值

struct ncclProxyOpsPool {
  struct ncclProxyOp ops[MAX_OPS_PER_PEER*NCCL_MAX_LOCAL_RANKS];//nrank个独立链表
  volatile int nextOps;//-1, 下一个要处理的
  volatile int nextOpsEnd;// 要处理的最后一个，尾插
  volatile int freeOps[NCCL_MAX_LOCAL_RANKS];//空闲op的起点
  pthread_mutex_t mutex;
  pthread_cond_t cond;
};

struct ncclProxyOp {
  struct ncclProxyConnection* connection;
  ssize_t nbytes;
  uint64_t opCount;//id==plan.collOpCount++>>1//comm.collOpCount++>>1

doLaunches
    ncclLaunchPrepare
    while (true) {
        moreRounds |= comm->planner.unlaunchedPlansHead != nullptr;
        if (moreRounds) {
          // Pop next unlaunched kernel
          struct ncclKernelPlan* plan = comm->planner.unlaunchedPlansHead;
          if (plan != nullptr) {
            comm->planner.unlaunchedPlansHead = plan->next;
            ncclLaunchKernelBefore_NoUncapturedCuda
                uploadWork
                    size_t batchBytes = plan->nWorkBatches*sizeof(struct ncclDevWorkBatch);
                    fifoBufHost = (void*)plan->kernelArgs;
                    fifoCursor = sizeof(ncclDevKernelArgs) + batchBytes;
                    //kernelarg,batch,
                    fifoMask = ~0u;
                    struct ncclDevWorkBatch* batchZero = (struct ncclDevWorkBatch*)(plan->kernelArgs+1);
                    for (int b=0; b < plan->nWorkBatches; b++) {
                        batchZero[b].offsetBase += fifoCursor;
                    }
                    struct ncclWorkList* workNode = ncclIntruQueueHead(&plan->workQueue);
                    while (workNode != nullptr) {
                        char* dst = (char*)fifoBufHost;
                        char* src = (char*)(workNode+1);
                        for (int n = workNode->size; n != 0; n -= 16) {
                          memcpy(
                            __builtin_assume_aligned(dst + (fifoCursor & fifoMask), 16),
                            __builtin_assume_aligned(src, 16),
                            16
                          );
                          fifoCursor += 16;
                          src += 16;
                        }
                        workNode = workNode->next;
                    }
            ncclLaunchKernel(comm, plan), result, failure);
                int nChannels = countOneBits(plan->channelMask);
                ///1的数量
                void* sym = plan->kernelFn;//函数指针
                dim3 grid = {(unsigned)nChannels, 1, 1};
                dim3 block = {(unsigned)plan->threadPerBlock, 1, 1};
                cudaStream_t launchStream = planner->streams->stream;
                void* extra[] = {
                    CU_LAUNCH_PARAM_BUFFER_POINTER, plan->kernelArgs,
                    CU_LAUNCH_PARAM_BUFFER_SIZE, &plan->kernelArgsSize,//连续的三者的大小
                    CU_LAUNCH_PARAM_END
                };    
                cudaGetFuncBySymbol(&fn, sym)
                //we have an extra level of hierarchy where we can group together several blocks within the Grid
                unsigned int clusterSize = (compCap >= 90) ? comm->config.cgaClusterSize : 0;
                ...
                cuLaunchKernelEx(&launchConfig, fn, nullptr, extra)
          }
          if (plan != nullptr) {
            ncclLaunchKernelAfter_NoCuda(comm, plan), result, failure);
                hostStreamPlanTask(comm, plan)
                    uploadProxyOps(comm, plan));
                        uint64_t collOpCount = comm->sharedRes->collOpCount;
                        comm->sharedRes->collOpCount += plan->collOpCount;
                        comm->collOpCount += plan->collOpCount;
                        struct ncclProxyOp* op = ncclIntruQueueHead(&plan->proxyOpQueue);
                        while (op != nullptr) {
                            } else { // coll
                              op->opCount = (collOpCount<<1) + oldId;//    plan.id->global.id
                            }
                            ncclProxySaveOp(comm, op, nullptr)
                                ncclLocalOpAppend(comm, &connector->proxyConn, op)
                                    struct ncclProxyOps* proxyOps = comm->proxyState->proxyOps;
                                    proxyOps += proxyConn->tpLocalRank;
                                    struct ncclProxyOpsPool* pool = proxyOps->pool;
                                    //共享内存
                                    int opIndex = proxyOps->freeOp;//空闲下标
                                    if (opIndex != -1) {//空闲用完了或第一次进入
                                        op = pool->ops+opIndex;
                                        proxyOps->freeOp = op->next;
                                    } else {
                                        freeOp = pool->freeOps[tpLocalRank]//起始空闲下标
                                        while ((freeOpNew = __sync_val_compare_and_swap(pool->freeOps+tpLocalRank, freeOp, -1)) != freeOp) freeOp = freeOpNew;
                                        //pool->freeOps[tpLocalRank]=-1
                                        opIndex = freeOp;
                                        op = pool->ops+opIndex;
                                        proxyOps->freeOp = op->next;//下一个空闲
                                    memcpy(op, proxyOp, sizeof(struct ncclProxyOp));//写共享内存
                                    op->next = -1;
                                    op->connection = proxyConn->connection;
                                    if (proxyOps->nextOps == -1) {
                                        proxyOps->nextOps = proxyOps->nextOpsEnd = opIndex;
                                        //下一个要处理的
                                    } else {
                                        pool->ops[proxyOps->nextOpsEnd].next = opIndex;
                                        proxyOps->nextOpsEnd = opIndex;
                                    }
                                    ++proxyOps->count
                            op->opCount = oldId;//??
                            op = op->enqNext;
                    ncclProxyStart(comm));
                        for (int r = 0; r < comm->sharedRes->tpNLocalRanks; r++) {
                            struct ncclProxyOps* ops = proxyOps + r;
                            if (ops->pool == NULL || ops->nextOps == -1) continue;
                            ncclProxyPost(ops->pool, ops->nextOps, ops->nextOpsEnd));
                                if (pool->nextOps == -1) {//proxyprogress已处理完所有数据
                                    pool->nextOps = nextOps;
                                    pthread_cond_signal(&pool->cond);
                                } else {
                                    pool->ops[pool->nextOpsEnd].next = nextOps;
                                }
                                pool->nextOpsEnd = nextOpsEnd;
                            ops->nextOps = ops->nextOpsEnd = -1;
                            ops->count = 0;
                        }
                        comm->opCount++;///plan的数量
                    if (!plan->persistent) {
                        // Notify main thread of our reclaiming. This will reclaim plan concurrently.
                        ncclIntruQueueMpscEnqueue(&comm->callbackQueue, &plan->reclaimer);
                    }
          }
        } else { // Final round.
          ncclLaunchFinish(comm), result, failure);
              ncclIntruQueueConstruct(&planner->planQueue);
        }
      if (!moreRounds) break;
    }
ncclCommPollCallbacks(comm, /*waitSome=*/false)
    struct ncclCommCallback* cb = ncclIntruQueueMpscDequeueAll(&comm->callbackQueue, waitSome);
    while (cb != nullptr) {
        struct ncclCommCallback* next = cb->next;
        ncclResult_t res1 = cb->fn(comm, cb); // may reclaim memory of cb
```

自带record

```
cudaStreamGetCaptureInfo_v2
cudaEventRecord
cudaGetFuncBySymbol
ncclDevKernel_Broadcast_RING_LL
cudaEventRecord
cudaStreamWaitEvent
```

```
Nsight Systems frames
libcuda.so.575.51.03!0x7f863eca1ec6
libnccl.so.2.26.5!cudaEventRecord
libnccl.so.2.26.5!ncclStreamWaitStream(...)
libnccl.so.2.26.5!ncclLaunchPrepare(...)
libnccl.so.2.26.5!doLaunches(...)
libnccl.so.2.26.5!groupLaunch(...)
libnccl.so.2.26.5!ncclGroupEndInternal(...)
libnccl.so.2.26.5!ncclEnqueueCheck(...)
libnccl.so.2.26.5!pncclBroadcast
overlap!main
libc.so.6!__libc_start_call_main
libc.so.6!__libc_start_main@@GLIBC_2
overlap!_start
```

# buffer size

## 理论分析

决定buffer size的有5个因素：算法(ring, tree)，协议(ll, ll128, simple)，是否跨机(net, p2p)，read/write，nChannel

算法决定了拓扑图的边，一条单向边需要两个connector，分别位于两个GPU上，recv connector负责接收，send connector负责发送，有独立的buffer。

单机ring有8条边（单向环），tree有14条边（双向链表），但两个rank之间也不会创建两条相同方向的边，所以，ring+tree一共是15条边，30个connector

边的两端进行通信的协议有3种，每种都有单独的buffer

默认值 ll=524288, ll128=4915200, simple=4194304, sum=9.6MB，所有协议的buffer是连续分布的，要2MiB对齐

边的类型：

1. p2p一般所有buffer分配在recv connector，有个开关决定simple buffer的分布，read=1，simple buffer分配在send connector

2. net在send connector和recv connector都有buffer，send connector的ll buffer分布在cpu

3. 还有额外的pxn(节点内GPU通过网卡通信)buffer：分配在send connector，大小nGPU * p2pnChannels * NCCL_SHARED_STEPS * p2pChunkSize，NCCL_SHARED_STEPS=16

同构channel就是在上面的基础上复制，异构channel需要独立分析边的数量和类型。

| 边的类型 | connector | 显存 per connector per channel (read=0) |
| ---- | --------- | ------------------------------------- |
| p2p  | recv      | ll + ll128 + simple                   |
| p2p  | send      | 0                                     |
| net  | recv      | ll + ll128 + simple                   |
| net  | send      | ll128 + simple                        |

总结：NCCL会通过搜索算法建立拓扑图，决定所有channel的边的数量，类型。搜索的过程不好理论分析（这块代码比较复杂，可以看源码或我的文档），它和硬件环境（GPU，NVlink，NIC，PCIE的带宽以及它们的连接）关系很大，可以直接获取搜索的结果，不同卡的连接是不一样的，要分别获取，再结合上面表格进行计算。搜索结果在NCCL log里：

```
  char line[1024];
  line[0]='\0';
  for (int c=0; c<comm->nChannels; c++) {
    struct ncclTree* tree = &comm->channels[c].tree;
    snprintf(line+strlen(line), 1023-strlen(line), " [%d] %d/%d/%d->%d->%d",
        c, tree->down[0], tree->down[1], tree->down[2], rank, tree->up);
    INFO(NCCL_GRAPH, "Ring %02d : %d -> %d -> %d", c, comm->channels[c].ring.prev, comm->rank, comm->channels[c].ring.next);
  }
  line[1023] = '\0';
  INFO(NCCL_INIT, "Trees%s", line);
```

```
szzj-inf-sci-k8s-hzz1-h72ni7-0893:1620032:1620032 [4] NCCL INFO Ring 00 : 5 -> 4 -> 3
szzj-inf-sci-k8s-hzz1-h72ni7-0893:1620032:1620032 [4] NCCL INFO Ring 01 : 3 -> 4 -> 5
szzj-inf-sci-k8s-hzz1-h72ni7-0893:1620032:1620032 [4] NCCL INFO Ring 02 : 5 -> 4 -> 3
szzj-inf-sci-k8s-hzz1-h72ni7-0893:1620032:1620032 [4] NCCL INFO Ring 03 : 3 -> 4 -> 5
szzj-inf-sci-k8s-hzz1-h72ni7-0893:1620032:1620032 [4] NCCL INFO Ring 04 : 12 -> 4 -> 3
szzj-inf-sci-k8s-hzz1-h72ni7-0893:1620032:1620032 [4] NCCL INFO Ring 05 : 3 -> 4 -> 12
szzj-inf-sci-k8s-hzz1-h72ni7-0893:1620032:1620032 [4] NCCL INFO Ring 06 : 5 -> 4 -> 3
szzj-inf-sci-k8s-hzz1-h72ni7-0893:1620032:1620032 [4] NCCL INFO Ring 07 : 3 -> 4 -> 5
szzj-inf-sci-k8s-hzz1-h72ni7-0893:1620032:1620032 [4] NCCL INFO Ring 08 : 5 -> 4 -> 3
szzj-inf-sci-k8s-hzz1-h72ni7-0893:1620032:1620032 [4] NCCL INFO Ring 09 : 3 -> 4 -> 5
szzj-inf-sci-k8s-hzz1-h72ni7-0893:1620032:1620032 [4] NCCL INFO Ring 10 : 5 -> 4 -> 3
szzj-inf-sci-k8s-hzz1-h72ni7-0893:1620032:1620032 [4] NCCL INFO Ring 11 : 3 -> 4 -> 5
szzj-inf-sci-k8s-hzz1-h72ni7-0893:1620032:1620032 [4] NCCL INFO Ring 12 : 12 -> 4 -> 3
szzj-inf-sci-k8s-hzz1-h72ni7-0893:1620032:1620032 [4] NCCL INFO Ring 13 : 3 -> 4 -> 12
szzj-inf-sci-k8s-hzz1-h72ni7-0893:1620032:1620032 [4] NCCL INFO Ring 14 : 5 -> 4 -> 3
szzj-inf-sci-k8s-hzz1-h72ni7-0893:1620032:1620032 [4] NCCL INFO Ring 15 : 3 -> 4 -> 5
szzj-inf-sci-k8s-hzz1-h72ni7-0893:1620032:1620032 [4] NCCL INFO Trees [0] 5/-1/-1->4->3 [1] 5/-1/-1->4->3 [2] 5/-1/-1->4->3 [3] 5/-1/-1->4->3 [4] 5/12/-1->4->-1 [5] -1/-1/-1->4->3 [6] 5/-1/-1->4->3 [7] 5/-1/-1->4->3 [8] 5/-1/-1->4->3 [9] 5/-1/-1->4->3 [10] 5/-1/-1->4->3 [11] 5/-1/-1->4->3 [12] 5/-1/-1->4->12 [13] -1/-1/-1->4->3 [14] 5/-1/-1->4->3 [15] 5/-1/-1->4->3
```

验证：

```
static void LogOnceFunc(struct ncclComm* comm) {
  static int logOnce = 0;
  if (logOnce == 0) {
    logOnce = 1;
    static char* names[3]={"LL","LL128","Simple"};
    for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
      INFO(NCCL_ALLOC, "%s Buffer size %d", names[p], comm->buffSizes[p]);
    }
    INFO(NCCL_ALLOC, "PXN send Buffer size %d = 8 * p2pnChannels(%d) * 16 * p2pChunkSize(%d)", 8 * comm->p2pnChannels * 16 * comm->p2pChunkSize, comm->p2pnChannels, comm->p2pChunkSize);
  }
}

ncclTransportP2pSetup{
  size_t edgeNum [3][2]={{0,0},{0,0},{0,0}};
  //p2p=0/net=2,recv=0/send=1
  size_t read = 0;
  size_t p2pRecv = 4096;
  size_t p2pSend = 4096;
  size_t netRecv = 0;
  size_t netSend = 0;

edgeNum[type][0]++;
if (type==0) {
    read = (comm->channels[c].peers[recvPeer]->recv + connIndex)->conn.flags & NCCL_P2P_READ ? 1 : 0;
}
edgeNum[type][1]++;

  LogOnceFunc(comm);
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) if (!(read && p == NCCL_PROTO_SIMPLE)) p2pRecv += comm->buffSizes[p];
  if (read) p2pSend += comm->buffSizes[NCCL_PROTO_SIMPLE];
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) netRecv += comm->buffSizes[p];
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) if (!(p == NCCL_PROTO_LL)) netSend += comm->buffSizes[p];
  ALIGN_SIZE(p2pRecv, CUDA_IPC_MIN);
  ALIGN_SIZE(p2pSend, CUDA_IPC_MIN);
  ALIGN_SIZE(netRecv, CUDA_IPC_MIN);
  ALIGN_SIZE(netSend, CUDA_IPC_MIN);
  INFO(NCCL_ALLOC, "P2P recv Buffer size %zu = %zu * %zu", edgeNum[0][0]*p2pRecv, edgeNum[0][0], p2pRecv);
  INFO(NCCL_ALLOC, "P2P send Buffer size %zu = %zu * %zu", edgeNum[0][1]*p2pSend, edgeNum[0][1], p2pSend);
  INFO(NCCL_ALLOC, "NET recv Buffer size %zu = %zu * %zu", edgeNum[2][0]*netRecv, edgeNum[2][0], netRecv);
  INFO(NCCL_ALLOC, "NET send Buffer size %zu = %zu * %zu", edgeNum[2][1]*netSend, edgeNum[2][1], netSend);
}
```

## trace

可以通过NCCL原生log来获取buffer size，需要设置环境变量NCCL_DEBUG_SUBSYS=ALLOC，经过对比，和理论分析是一致。每创建一个connector都会产生这个log，这里面的size需要2MiB对齐，全部加起来就是buffer size大小

```
p2p.cc:ncclP2pAllocateShareableBuffer(已有)
  INFO(NCCL_P2P|NCCL_ALLOC, "Allocated shareable buffer %p size %zu ipcDesc %p", *ptr, size, ipcDesc);

p2p.cc:p2pRecvSetup/p2pSencSetup
  int tmpsize = info->p2pBuff.size;
  ALIGN_SIZE(tmpsize, CUDA_IPC_MIN);
  INFO(NCCL_ALLOC, "CTRACE P2P Recv Buffer Size %d", tmpsize);
net.cc:recvProxyConnect/sendProxyConnect
    int tmpsize = map->mems[NCCL_NET_MAP_DEVMEM].size;
    ALIGN_SIZE(tmpsize, CUDA_IPC_MIN);
    INFO(NCCL_ALLOC, "CTRACE NET Send Buffer Size %d", tmpsize);
net.cc:sharedNetBuffersInit
    int tmpsize = map->mems[NCCL_NET_MAP_DEVMEM].size;
    ALIGN_SIZE(tmpsize, CUDA_IPC_MIN);
    INFO(NCCL_ALLOC, "CTRACE SHARED_NET Send Buffer Size %d", tmpsize);
```

验证理论和trace

```
import re
import os
import argparse
import subprocess
import pandas as pd
import concurrent.futures
from tqdm import tqdm
from collections import defaultdict

cudaDev_pattern = re.compile(
    r'cudaDev (\d+)',
    re.IGNORECASE
)
trace_origin_pattern = re.compile(
    r'.* Allocated shareable buffer .* size (\d+) ipcDesc .*',
    re.IGNORECASE
)

trace_new_pattern = re.compile(
    r'.* CTRACE (P2P|NET|SHARED_NET) (Recv|Send) Buffer Size (\d+)',
    re.IGNORECASE
)
compute_proto_size_pattern = re.compile(
    r'.* (LL|LL128|Simple) Buffer Size (\d+)',
    re.IGNORECASE
)
compute_buffer_size_pattern = re.compile(
    r'.* (P2P|NET|PXN) (recv|send) Buffer Size (\d+)',
    re.IGNORECASE
)

def parse_nccl_log(log_content):
    cudaDev=0
    trace_origin=[]
    trance_new={"P2P":{"Recv": [], "Send": []}, "NET": {"Recv": [], "Send": []}, "SHARED_NET": {"Recv": [], "Send": []}}
    compute_proto_size={"LL": 0, "LL128": 0, "Simple": 0}
    compute_buffer_size={"P2P": {"recv": [], "send": []}, "NET": {"recv": [], "send": []}, "PXN": {"recv": [], "send": []}}

    lines = log_content.strip().split("\n")
    for line in lines:
        line = line.strip()

        cudaDev_match = cudaDev_pattern.search(line)
        if cudaDev_match:
            cudaDev = int(cudaDev_match.group(1))
            continue
        trace_origin_match = trace_origin_pattern.search(line)
        if trace_origin_match:
            trace_origin.append(int(trace_origin_match.group(1)))
            continue
        trace_new_match = trace_new_pattern.search(line)
        if trace_new_match:
            trance_new[trace_new_match.group(1)][trace_new_match.group(2)].append(int(trace_new_match.group(3)))
            continue
        compute_proto_size_match = compute_proto_size_pattern.search(line)
        if compute_proto_size_match:
            compute_proto_size[compute_proto_size_match.group(1)]=int(compute_proto_size_match.group(2))
            continue
        compute_buffer_size_match = compute_buffer_size_pattern.search(line)
        if compute_buffer_size_match:
            compute_buffer_size[compute_buffer_size_match.group(1)][compute_buffer_size_match.group(2)].append(int(compute_buffer_size_match.group(3)))
            continue
    return cudaDev,trace_origin,trance_new,compute_proto_size,compute_buffer_size

def print_status(trace_origin,trance_new,compute_buffer_size):
    trace_origin_aligned = sum([(a+2048*1024-1)//(2048*1024)*(2048*1024) for a in trace_origin])/1024/1024
    trace_origin=sum(trace_origin)/1024/1024
    tmp=0
    for k,v in trance_new.items():
        for kk,vv in v.items():
            tmp += sum(vv)
    trance_new=tmp/1024/1024
    tmp=0
    for k,v in compute_buffer_size.items():
        for kk,vv in v.items():
            tmp += sum(vv)
    compute_buffer_size=tmp/1024/1024
    return trace_origin,trace_origin_aligned,trance_new,compute_buffer_size

def process_log_file(log_path):
    with open(log_path, 'r') as f:
        cudaDev,trace_origin,trance_new,compute_proto_size,compute_buffer_size = parse_nccl_log(f.read())
        trace_origin,trace_origin_aligned,trance_new,compute_buffer_size = print_status(trace_origin,trance_new,compute_buffer_size)
    return cudaDev,trace_origin,trace_origin_aligned,trance_new,compute_proto_size,compute_buffer_size

def main():
    parser = argparse.ArgumentParser(description="NCCL通信组特征分析工具")
    parser.add_argument("-i", "--input", help="logfile path")
    parser.add_argument("-d", "--dir", help="log目录，处理目录下所有.log文件")
    parser.add_argument("-o", "--output", default='trace_buffer.txt', help="可选，若不指定则为trace_buffer.txt")
    args = parser.parse_args()

    output_file = args.output
    if args.dir: 
        log_tasks = []
        for fname in os.listdir(args.dir):
            if fname.endswith('.log'):
                log_path = os.path.join(args.dir, fname)
                log_tasks.append(log_path)
        # 多进程并行处理
        results = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_log_file, log_path) for log_path in log_tasks]
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        results.append(result)  # (base_name, lines)
                except Exception as exc:
                    print(f"日志处理出错: {exc}")
        sorted_results = sorted(results, key=lambda x: x[0])
        print(results[0][4])
        for result in sorted_results:
            print("cudaDev", result[0])
            print("trace_origin", result[1])
            print("trace_origin_aligned", result[2])
            print("trance_new", result[3])
            print("compute_buffer_size", result[5])
            print()
        print(max(result[5] for result in results))
    elif args.input:
        # 单文件处理
        result = process_log_file(args.input)
        print("cudaDev", result[0])
        print("trace_origin", result[1])
        print("trace_origin_aligned", result[2])
        print("trance_new", result[3])
        print("compute_buffer_size", result[5])    
    else:
        print("请指定 -i <logfile> 或 -d <logdir>")
        exit(1)


if __name__ == "__main__":
    main()
```

```
{'LL': 524288, 'LL128': 4915200, 'Simple': 4194304}

cudaDev 7
trace_origin 670.375//log
trace_origin_aligned 672.0
trance_new 672.0//分配
compute_buffer_size 672.0//理论

672.0
```

使用cudaMemGetInfo测显存，不一致。一些在GPU上的数据结构，不算buffer

```
Fri Jul 25 19:12:03 2025[1,7]<stdout>:Rank 7: free memory before init = 64164.250000 MiB
Fri Jul 25 19:13:03 2025[1,7]<stdout>:Rank 7: free memory before buffer = 63304.250000 MiB
Fri Jul 25 19:13:10 2025[1,7]<stdout>:Rank 7: free memory after buffer = 62536.250000 MiB
Fri Jul 25 19:13:15 2025[1,7]<stdout>:Rank 7: free memory after init = 62532.250000 MiB
860, 768, 4
```

# 原语时延

## 理论分析

- 一些论文对原语时延的建模是一个线性函数，可以用nccl-tesets作为基准去拟合，但没什么作用

- nccl代码计算了不同原语在不同算法和协议下的时延，用于tuning，不考虑计算开销。经过测试，ncc理论时延比cuda event的测量结果小了两个数量级，数字也对不上。

- nccl-tests在大数据下可以用来验证bus带宽是否充分使用，不能计算理论时延。它是先统计时延，然后通过一些公式算出bus带宽。

无结果

## trace

- 在nccl层面，在rdma的执行前后计时，写到log里。缺点是不能用于单机通信，单机只能用event测kernel时间了。

- 在paddle层面，在通信前后插入cudaEventRecord，在框架中开一个线程轮询cudaEventQuery，可以获得差值。

光有时延还是不够的，要能够获得绝对时间，这样可以建立时间轴，多个GPU的情况可以整合到一起，分析系统的问题

#### 方案1

原语的执行主要分为两部分：user buffer和nccl buffer之间通过kernel完成数据交换。如果有跨机通信，两个机器之间的nccl buffer通信由proxyprogress使用rdma完成。

send计时开始：nccl buffer有数据，GPU已经工作

recv计时开始：接收到网络数据

计时结束，rdma任务完成

和cuda event不是完全相等的，cuda event测的是kernel的时间

16卡，数据量100MiB float，只列出rank0，rdma的时间要小于cudaEvent测的时间

```
             broadcast, reduce,    allgather, reducescatter, allreduce
rdma         21.528965, 80.293637, 16.579797, 66.932333,     83.971568
event        28.413,    102.010,   17.664,    68.968,        86.146
```

需要增加的log

enqueue.cc

```
#include <chrono>
extern std::chrono::steady_clock::time_point timer_start;

scheduleCollTasksToPlan
    if (comm->rank == 0) {
      INFO(NCCL_TUNING, "%s: %ld Bytes -> Algo %s proto %s channel{Lo..Hi}={%d..%d} comm %p progressState %p", 
        ncclFuncToString(task->func), task->count * ncclTypeSize(task->datatype), ncclAlgoToString(task->algorithm),
        ncclProtoToString(task->protocol), devWork->channelLo, devWork->channelHi, comm, &comm->proxyState->progressState);
```

proxy.cc

```
#include <chrono>

extern std::chrono::steady_clock::time_point timer_start;

removeOp
  auto delta = std::chrono::steady_clock::now() - timer_start;
  double timestamp = std::chrono::duration_cast<std::chrono::duration<double>>(delta).count()*1000;
  INFO(NCCL_TUNING, "progressState %p taskID %ld end at %lf ms", state, (*opPtr)->opCount, timestamp);
```

net.cc

```
#include <chrono>
std::chrono::steady_clock::time_point timer_start = std::chrono::steady_clock::now();

sendProxyProgress
        if (connFifo[buffSlot].size != -1 && (*recvTail > tail || p == NCCL_PROTO_LL)) {
          // We have something to receive, let's check if it's completely ready.
          if (sub->transmitted == 0) {
            auto delta = std::chrono::steady_clock::now() - timer_start;
            double timestamp = std::chrono::duration_cast<std::chrono::duration<double>>(delta).count()*1000;
            INFO(NCCL_TUNING, "progressState %p taskID %ld send at %lf ms", &proxyState->progressState, args->opCount, timestamp);
          }

recvProxyProgress
        if (done) {
        //数据已完全到达并可用
          if (subGroup->received == 0) {
            auto delta = std::chrono::steady_clock::now() - timer_start;
            double timestamp = std::chrono::duration_cast<std::chrono::duration<double>>(delta).count()*1000;
            INFO(NCCL_TUNING, "progressState %p taskID %ld recv at %lf ms", &proxyState->progressState, args->opCount, timestamp);
          }
```

#### 方案2

pp trace
