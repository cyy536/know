# server

CUDA supported arches: 7.0;7.5;8.0;8.6;8.9;9.0

需要单卡64GB内存

```
NVIDIA A100-SXM4-40GB 华东 Linux
MACHcULgp7QyhEJe9GM0OdFP 
CPU：
Intel(R) Xeon(R) Silver 4210 CPU 503G 40核
本机总量：
40核CPU 503G内存
每张GPU分配：
5核CPU 62G内存
硬盘：
系统盘： 20G
数据盘： 50GB NVME (关机保留24小时/可扩容至1350GB)
网络：
上行带宽：400 Mbps/s
下行带宽：400 Mbps/s

NVIDIA A100-PCIE-40GB 
MACHKjpJ5jXWPBpFcVOZvrXl 
CPU：
Intel(R) Xeon(R) Gold 6442Y 503G 96核
本机总量：
96核CPU 503G内存
每张GPU分配：
12核CPU 62G内存
硬盘：
系统盘： 20G
数据盘： 50GB SSD (关机保留24小时/可扩容至12110GB)
网络：
上行带宽：400 Mbps/s
下行带宽：800 Mbps/s
pcie4*16

NVIDIA RTX 6000D
MACHijDnyvTTvjvUcnZADOeM
AMD EPYC 7282 16-Core Processor 
pcie4*16

NVIDIA A800 80GB PCIe
MACH1OXg3UYw7ecLvB9uw85H 
CPU：
Intel(R) Xeon(R) Gold 6338 CPU 96G 32核
本机总量：
32核CPU 96G内存
每张GPU分配：
32核CPU 96G内存
硬盘：
系统盘： 20G
数据盘： 50GB SSD (关机保留24小时/可扩容至560GB)
网络：
上行带宽：102 Mbps/s
下行带宽：350 Mbps/s

MACHVEK91UXJN1E4eSctujkj 
pcie4*16
```

```
docker-pytorch2.4.0
```

# model

```
MHA
LlamaForCausalLM
meta-llama/Llama-2-13b
https://paddlenlp.readthedocs.io/zh/latest/website/index.html

GQA
Llama3.1-8B
Gemma2ForCausalLM
google/gemma-2-27b
export HF_TOKEN="hf_xxxx"
export VLLM_ATTENTION_BACKEND=FLASHINFER
pip install flashinfer-python==0.2.0
RuntimeError: Error in function 'PrefillSplitQOKVIndptr' at /usr/local/lib/python3.11/dist-packages/flashinfer/data/include/flashinfer/attention/scheduler.cuh:515: kv_indptr[1]-1 - kv_indptr[0]0 should be non-negative
注释掉raise，使用flashattn
```

# rate

llama等48*5 second

gemma等60*5

先找一个差距足够大的请求，再找时延相等的请求，一个一个点调

# test

```
import asyncio
import random
import time
import torch
import vllm
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from vllm.utils import random_uuid
from datasets import load_dataset

def get_prompts_from_sharegpt():
    # sampling_params = SamplingParams(temperature=0.7, min_tokens=1024)
    sampling_params = SamplingParams(temperature=0.7, min_tokens=4096)
    dataset = load_dataset("Crystalcareai/Code-feedback-sharegpt-renamed", split="train")

    prompts = []
    for entry in dataset:
        convs = entry.get("messages", [])
        if convs and len(convs) > 0:
            user_msg = next((m["value"] for m in convs if m["role"] in ["human", "user"]), None)
            if user_msg:
                prompts.append(user_msg)
    return prompts,sampling_params

def get_prompts_from_tldr():
    sampling_params = SamplingParams(
        temperature=0.0, # 摘要任务通常使用 greedy search 保证稳定性
        # min_tokens=512,
        # max_tokens=1024,
        min_tokens=1024,
        max_tokens=2048,        
    )
    dataset = load_dataset("CarperAI/openai_summarize_tldr", split="train")

    return [entry["prompt"] for entry in dataset],sampling_params

def get_prompts_from_writingprompts():
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        # min_tokens=1024,#llama
        min_tokens=4096,#gemma
    )
    dataset = load_dataset("euclaise/writingprompts", split="train")
    def format_writing_prompts(entry):
        return f"Write a story based on this prompt: {entry['prompt']}\n\nStory:"
    return [format_writing_prompts(entry) for entry in dataset],sampling_params
   
def get_wait_time(prompt_num,arrival_rate):
    random.seed(10)
    return [random.expovariate(arrival_rate) for _ in range(prompt_num)]

async def run_poisson_requests(engine, prompt_list: list, wait_list: list, sampling_params: SamplingParams):
    tasks = []

    for prompt,wait_time in zip(prompt_list,wait_list):
        await asyncio.sleep(wait_time)
        
        request_id = random_uuid()
        # 异步启动生成任务，不等待它完成，直接进入下一次循环等待新请求
        task = asyncio.create_task(handle_request(engine, prompt, sampling_params, request_id))
        tasks.append(task)

    # 等待所有已发出的请求处理完毕
    await asyncio.gather(*tasks)

async def handle_request(engine, prompt, sampling_params, request_id):
    # 提交给 AsyncLLMEngine
    results_generator = engine.generate(prompt, sampling_params, request_id)
    
    final_output = None
    async for request_output in results_generator:
        final_output = request_output

    # print("prompt:",prompt)
    # print("output:",final_output.outputs[0].text)
    print("output len:",len(final_output.outputs[0].token_ids))

async def main():
    # 配置引擎（根据你的显存调整）
    # engine_args = AsyncEngineArgs(
    #     model="/hy-tmp/models/Llama-2-13b", 
    #     gpu_memory_utilization=0.8,
    #     manager_type='pre_swap_out'
    # )
    engine_args = AsyncEngineArgs(
        model="/hy-tmp/models/gemma-2-27b",
        gpu_memory_utilization=0.8,
        manager_type='pre_swap_out'
    )

    engine = AsyncLLMEngine.from_engine_args(engine_args)

    # 准备 Prompt 列表
    all_prompts,sampling_params = get_prompts_from_sharegpt()
    # all_prompts,sampling_params = get_prompts_from_tldr()
    # all_prompts,sampling_params = get_prompts_from_writingprompts()
    
    test_prompts = all_prompts[:10000]

    # 启动泊松流量测试
    arrival_rate = 13
    wait_list = get_wait_time(len(test_prompts), arrival_rate)
    await run_poisson_requests(engine, prompt_list=test_prompts, wait_list=wait_list, sampling_params=sampling_params)

if __name__ == "__main__":
    s_t = time.time()
    asyncio.run(main())
    e_t = time.time()
    print(f"main time: {(e_t-s_t)*1000}\n")
```

# vllm

```
vllm0.5.3/CMakeLists.txt
      cutlass
      SOURCE_DIR /hy-tmp/vllm0.5.3/cutlass-3.5.0
```

```
transformers==4.42.4
```

```
AsyncLLMEngine
	RequestTracker异步请求队列
	_AsyncLLMEngine(LLMEngine)
		input_processor	
		Scheduler调度算法和请求队列
		MultiprocessingGPUExecutorAsync(MultiprocessingGPUExecutor单卡、多卡
			Worker(LocalOrDistributedWorkerBase)一个进程/一张 GPU
            	CacheEngine显存管理
				model_runner模型Forward
					CUDAGraphRunner
					model
		output_processor
```

```
vllm\config.py
class CacheConfig:
    def __init__(
        self.page_size = 8
        self.block_size = block_size
        self.gpu_memory_utilization = gpu_memory_utilization#model+cache
        self.swap_space_bytes = swap_space * _GB
```

## core

```
vllm\core\scheduler.py
class Scheduler:
    def _schedule(self) -> SchedulerOutputs:
        """Schedule queued requests."""
        if self.scheduler_config.chunked_prefill_enabled:
            return self._schedule_chunked_prefill()
        else:
            return self._schedule_default()
    def _schedule_default(self) -> SchedulerOutputs:
    def has_unfinished_seqs(self) -> bool:
        return len(self.waiting) != 0 or len(self.running) != 0 or len(
            self.swapped) != 0
    def schedule(self) -> Tuple[List[SequenceGroupMetadata], SchedulerOutputs]:
        scheduler_outputs = self._schedule()
        print("blocks_to_copy",scheduler_outputs.blocks_to_copy)
        print("blocks_to_swap_out",scheduler_outputs.blocks_to_swap_out)
```

## engine

```
vllm0.5.3/vllm/engine/arg_utils.py
@dataclass
class EngineArgs:
    tensor_parallel_size: int = 1
    swap_space: int = 64  # GiB 单进程的资源数量？
    def create_engine_config(self, ) -> EngineConfig:
        cache_config = CacheConfig(
            block_size=self.block_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            swap_space=self.swap_space,
    def create_engine_config(self, ) -> EngineConfig:
        cache_config = CacheConfig(
        return EngineConfig(
@dataclass
class AsyncEngineArgs(EngineArgs):
```

```
vllm0.5.3/vllm/engine/async_llm_engine.py
class _AsyncLLMEngine(LLMEngine):
    """Extension of LLMEngine to add async methods."""
    async def add_request_async(
        processed_inputs = await self.process_model_inputs_async(    
        self._add_processed_request(    
    async def process_model_inputs_async(
        return self.input_processor(llm_inputs)
    async def step_async(
        seq_group_metadata_list, scheduler_outputs = self.scheduler[
        with open("schedule.log", "a") as f:
        if not scheduler_outputs.is_empty():        
            output = await self.model_executor.execute_model_async(
        request_outputs = self._process_model_outputs(        
    async def stop_remote_worker_execution_loop_async(self) -> None:
        """Stop the remote worker execution loop."""
        await self.model_executor.stop_remote_worker_execution_loop_async()
        
class AsyncLLMEngine:
    _engine_class: Type[_AsyncLLMEngine] = _AsyncLLMEngine
    def __init__(self,
        self.engine = self._init_engine(*args, **kwargs)
    def _init_engine(self, *args,
        if not self.engine_use_ray:
            engine_class = self._engine_class
        return engine_class(*args, **kwargs)
    async def _process_request(
    #一个请求一个队列，每次取一个结果，空就挂起协程
        stream = await self.add_request(
        try:
            async for request_output in stream:
                yield request_output
    async def add_request(#这里为啥要写成协程
        if not self.is_running:
            if self.start_engine_loop:
                self.start_background_loop()
        stream = self._request_tracker.add_request(
        return stream
    @property
    def is_running(self) -> bool:
        return (self.background_loop is not None
                and self._background_loop_unshielded is not None
                and not self._background_loop_unshielded.done())
    async def engine_step(self, virtual_engine: int) -> bool:
        new_requests, finished_requests = (
            self._request_tracker.get_new_and_finished_requests())                     for new_request in new_requests:
                    await self.engine.add_request_async(**new_request)
        else:                    
            request_outputs = await self.engine.step_async(virtual_engine)
        for request_output in request_outputs:
            self._request_tracker.process_request_output(   
    def from_engine_args(
        engine_config = engine_args.create_engine_config()
        engine = cls(
    async def generate(
    #一个请求一个队列，每次取一个结果，空就挂起协程
        async for output in self._process_request(
            yield LLMEngine.validate_output(output, RequestOutput)
    async def run_engine_loop(self):
        has_requests_in_progress = [False] * pipeline_parallel_size
        while True:
            if not any(has_requests_in_progress):
                    await self.engine.stop_remote_worker_execution_loop_async()????
                await self._request_tracker.wait_for_new_requests()        
                requests_in_progress = [
                    asyncio.create_task(self.engine_step(ve))
                    for ve in range(pipeline_parallel_size)
                ]
                has_requests_in_progress = [True] * pipeline_parallel_size
            try:
                async with asyncio_timeout(ENGINE_ITERATION_TIMEOUT_S):
                    done, _ = await asyncio.wait(
                        requests_in_progress,
                        return_when=asyncio.FIRST_COMPLETED)
                    for _ in range(pipeline_parallel_size):
                        await asyncio.sleep(0)
                for task in done:
                    result = task.result()
                    virtual_engine = requests_in_progress.index(task)      
                        has_unfinished_requests = (
                            self.engine.
                            has_unfinished_requests_for_virtual_engine(
                                virtual_engine))
                    if result or has_unfinished_requests:
                        requests_in_progress[virtual_engine] = (
                            asyncio.create_task(
                                self.engine_step(virtual_engine)))
                        has_requests_in_progress[virtual_engine] = True
                    else:
                        has_requests_in_progress[virtual_engine] = False                             await asyncio.sleep(0)   
    def start_background_loop(self) -> None:
        self._request_tracker = RequestTracker()
        self._background_loop_unshielded = asyncio.get_event_loop(
        ).create_task(self.run_engine_loop())

class AsyncStream:
#一个请求一个队列，每次取一个结果，空就挂起协程
    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        self._queue: asyncio.Queue = asyncio.Queue()
        self._finished = False
    def put(self, item: Union[RequestOutput, EmbeddingRequestOutput,
                              Exception]) -> None:
        if self._finished:
            return
        self._queue.put_nowait(item)
    async def __anext__(self) -> Union[RequestOutput, EmbeddingRequestOutput]:
        result = await self._queue.get()
        if isinstance(result, Exception):
            raise result
        return result

class RequestTracker:
    def __init__(self) -> None:
        self._new_requests: asyncio.Queue[Tuple[AsyncStream,
        self.new_requests_event = asyncio.Event()
    def add_request(self,
        stream = AsyncStream(request_id)
        self._new_requests.put_nowait((stream, {
            "request_id": request_id,
        self.new_requests_event.set()
        return stream
```

```
vllm0.5.3/vllm/engine/llm_engine.py
class LLMEngine:
    def __init__(
    	self.cache_config = cache_config
        self.model_executor = executor_class(
            self._initialize_kv_caches()
    def _add_processed_request(
        min_cost_scheduler = self.scheduler[costs.index(min(costs))]
        min_cost_scheduler.add_seq_group(seq_group)
    def _initialize_kv_caches(self) -> None:
        num_gpu_blocks, num_cpu_blocks = (
            self.model_executor.determine_num_available_blocks())
        self.model_executor.initialize_cache(num_gpu_blocks, num_cpu_blocks)
    def _process_model_outputs(
            self.output_processor.process_prompt_logprob(seq_group, outputs)
        for scheduler in self.scheduler:
            scheduler.free_finished_seq_groups()   
    def has_unfinished_requests_for_virtual_engine(
        return self.scheduler[virtual_engine].has_unfinished_seqs()
    def step(self) -> List[Union[RequestOutput, EmbeddingRequestOutput]]:
            output = self.model_executor.execute_model(
```

## executor

```
vllm0.5.3/vllm/executor/executor_base.py
class ExecutorBase(ABC):
    def __init__(
    	self.cache_config = cache_config
        self._init_executor()
```

```
vllm0.5.3/vllm/executor/distributed_gpu_executor.py
class DistributedGPUExecutor(GPUExecutor):
    def initialize_cache(self, num_gpu_blocks: int,
        self._run_workers("initialize_cache",
```

```
vllm0.5.3/vllm/executor/gpu_executor.py
def create_worker(worker_module_name, worker_class_name, **kwargs):
    wrapper = WorkerWrapperBase(
    wrapper.init_worker(**kwargs)
    return wrapper.worker
class GPUExecutor(ExecutorBase):
    def _create_worker(self,
        return create_worker(**self._get_create_worker_kwargs(
    def _init_executor(self) -> None:
        self.driver_worker = self._create_worker()
        self.driver_worker.init_device()
        self.driver_worker.load_model()
    def determine_num_available_blocks(self) -> Tuple[int, int]:
        return self.driver_worker.determine_num_available_blocks()
    def execute_model(
        output = self.driver_worker.execute_model(execute_model_req)
    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks) -> None:
        self.driver_worker.initialize_cache(num_gpu_blocks, num_cpu_blocks)
        
class GPUExecutorAsync(GPUExecutor, ExecutorAsyncBase):
    async def execute_model_async(
        output = await make_async(self.driver_worker.execute_model
        #Take a blocking function, and run it on in an executor thread.
```

```
vllm0.5.3/vllm/executor/multiproc_gpu_executor.py
class MultiprocessingGPUExecutor(DistributedGPUExecutor):
    def _run_workers(
        worker_outputs = [
            worker.execute_method(method, *args, **kwargs)
            for worker in self.workers
        ]
        driver_worker_method = getattr(self.driver_worker, method)
        driver_worker_output = driver_worker_method(*args, **kwargs)
class MultiprocessingGPUExecutorAsync(MultiprocessingGPUExecutor,
    def __init__(self, *args, **kwargs):
        self.driver_exec_model = make_async(self.driver_worker.execute_model)
```

```
vllm0.5.3/vllm/executor/multiproc_worker_utils.py
class ProcessWorkerWrapper:
    def execute_method(self, method: str, *args, **kwargs):
        self._enqueue_task(future, method, args, kwargs)
```

## worker

```
vllm0.5.3/vllm/worker/cache_engine.py
class CacheEngine:
    def __init__(
        self.gpu_cache = self._allocate_kv_cache(
        self.cpu_cache = self._allocate_kv_cache(self.num_cpu_blocks, "cpu")
    def _allocate_kv_cache(
        kv_cache_shape = self.attn_backend.get_kv_cache_shape(
            num_blocks, self.block_size, self.num_kv_heads, self.head_size)
        src_kv_cache = torch.zeros(kv_cache_shape,
        for i in range(self.num_attention_layers):
            kv_cache.append(src_kv_cache[i])
```

```
vllm0.5.3/vllm/worker/model_runner.py
class ModelRunner(GPUModelRunnerBase[ModelInputForGPUWithSamplingMetadata]):
    def execute_model(
    	# Currently cuda graph is only supported by the decode phase.
        if prefill_meta is None and decode_meta.use_cuda_graph:
            assert model_input.input_tokens is not None
            graph_batch_size = model_input.input_tokens.shape[0]
            model_executable = self.graph_runners[virtual_engine][
                graph_batch_size]
        else:
            model_executable = self.model
        hidden_or_intermediate_states = model_executable(
        with open("decode.log", "a") as f:
            if is_prefill_state:
                f.write(f"prefill Time: {start_event.elapsed_time(end_event)}\n")
            else:
                f.write(f"decode Time: {start_event.elapsed_time(end_event)}\n")
        logits = self.model.compute_logits(hidden_or_intermediate_states,
        output: SamplerOutput = self.model.sample(
class GPUModelRunnerBase(ModelRunnerBase[TModelInputForGPU]):
    def profile_run(self) -> None:
        self.execute_model(model_input, kv_caches, intermediate_tensors)
        torch.cuda.synchronize()
```

```
vllm0.5.3/vllm/worker/worker.py
class Worker(LocalOrDistributedWorkerBase):
    def __init__(
        self.model_runner: GPUModelRunnerBase = ModelRunnerClass(
    def _init_cache_engine(self):
        self.cache_engine = [
            CacheEngine(self.cache_config, self.model_config,
                        self.parallel_config, self.device_config)
            for _ in range(self.parallel_config.pipeline_parallel_size)
        ]
        self.gpu_cache = [
            self.cache_engine[ve].gpu_cache
            for ve in range(self.parallel_config.pipeline_parallel_size)
        ]
    def determine_num_available_blocks(self) -> Tuple[int, int]:
        self.model_runner.profile_run()
        free_gpu_memory, total_gpu_memory = torch.cuda.mem_get_info()
        peak_memory = self.init_gpu_memory - free_gpu_memory
        num_gpu_blocks = int(
            (total_gpu_memory * self.cache_config.gpu_memory_utilization -
             peak_memory) // cache_block_size)
        num_cpu_blocks = int(self.cache_config.swap_space_bytes //
                             cache_block_size)
        num_gpu_blocks = max(num_gpu_blocks, 0)
        num_cpu_blocks = max(num_cpu_blocks, 0)
    def execute_worker(self, worker_input: WorkerInput) -> None:
            self.cache_engine[virtual_engine].swap_in(
            self.cache_engine[virtual_engine].swap_out(
            self.cache_engine[virtual_engine].copy(worker_input.blocks_to_copy)           
            with open("defrag.log", "a") as f:
            self.cache_engine[virtual_engine].
            pre_swap_out(worker_input.pages_to_pre_swap_out)
    def initialize_cache(self, num_gpu_blocks: int,
        raise_if_cache_size_invalid(num_gpu_blocks,
        self._init_cache_engine()
        self._warm_up_model()
```

```
vllm\worker\worker_base.py
class LocalOrDistributedWorkerBase(WorkerBase):
    def execute_model(
        if self.is_driver_worker:
            worker_input: WorkerInput = self.prepare_worker_input(
                self.model_runner.prepare_model_input(
        self.execute_worker(worker_input)
        with open("memory.log", "a") as f:    
        self.execute_worker(worker_input)
        output = self.model_runner.execute_model(
            model_input, self.kv_cache[worker_input.virtual_engine]
```



