# install

```
https://docs.vllm.ai/projects/ascend/en/v0.11.0/installation.html
安装toolkit kernels NNAL vllm
CANN == 8.3.RC2
torch_npu
Version: 2.7.1
```

```
pip install modelscope
# Load model from ModelScope to speed up download
export VLLM_USE_MODELSCOPE=True
export MODELSCOPE_CACHE=/root/autodl-tmp/modelscope_cache

# Set `max_split_size_mb` to reduce memory fragmentation and avoid out of memory
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256
```

# code

```
from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8)

# Create an LLM.
llm = LLM(model="Qwen/Qwen3-30B-A3B", enforce_eager=True)
# 关闭torch.compile和cudagraph

# Generate texts from the prompts.
outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

# import pdb
# import socket

# def pre_pdb(local_rank, port, addr='127.0.0.1'):
#     port += local_rank
#     handle_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
#     handle_socket.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,True)
#     handle_socket.bind((addr,port))
#     handle_socket.listen(1)
#     print("pdb %d %s:%d" % (local_rank,addr,port))
#     (client,address) = handle_socket.accept()
#     handle = client.makefile('rw')
#     return handle

# handle = pre_pdb(0,16890)
# pdb.Pdb(stdin=handle,stdout=handle).set_trace()
```

```
LLM->LLMEngine->EngineCoreClient->CoreEngineProcManager
EngineCoreProc->EngineCore->
UniProcExecutor
Scheduler
```

```
UniProcExecutor->WorkerWrapperBase->NPUWorker->
register_ascend_customop
NPUModelRunner->model
```

```
Scheduler
```

```
class LLM:
	def __init__:
		self.llm_engine = LLMEngine.from_engine_args(
    def generate(
        self._validate_and_add_requests(
            prompts=prompts,
            params=sampling_params,
        outputs = self._run_engine(use_tqdm=use_tqdm)
        return self.engine_class.validate_outputs(outputs, RequestOutput)
    def _validate_and_add_requests(
        it = prompts
        for i, prompt in enumerate(it):
            self._add_request(
    def _add_request(
        request_id = str(next(self.request_counter))
        self.llm_engine.add_request(
```

```
class LLMEngine:
    def from_engine_args(
    	executor_class = Executor.get_class(vllm_config)
    	return cls(vllm_config=vllm_config,
	def __init__:
		# Processor (convert Inputs --> EngineCoreRequests)
    	self.processor = Processor(vllm_config=vllm_config,
        self.output_processor = OutputProcessor(self.tokenizer,
		self.engine_core = EngineCoreClient.make_client(
    def add_request(
        prompt_str, request = self.processor.process_inputs(
        self.output_processor.add_request(request, prompt_str, None, 0)
        self.engine_core.add_request(request)
```

## EngineCoreClient

```
/root/autodl-tmp/vllm-ascend-env/lib/python3.10/site-packages/vllm/v1/engine/core_client.py
class EngineCoreClient:
	def make_client:
        if multiprocess_mode and not asyncio_mode:
            return SyncMPClient(vllm_config, executor_class, log_stats)
class MPClient(EngineCoreClient):
	def __init__:
		with launch_core_engines(vllm_config, executor_class,
        	self.resources.engine_manager = engine_manager
		(input_address, ) = addresses.inputs
        (output_address, ) = addresses.outputs
        self.input_socket = self.resources.input_socket = make_zmq_socket(
        self.ctx, input_address, zmq.ROUTER, bind=True)
        self.resources.output_socket = make_zmq_socket(
                self.ctx, output_address, zmq.PULL)
class SyncMPClient(MPClient):
    def add_request(self, request: EngineCoreRequest) -> None:
        self._send_input(EngineCoreRequestType.ADD, request)
    def _send_input(self, request_type: EngineCoreRequestType, request: Any):
    	self.input_socket.send_multipart(msg, copy=False)
```

```
def launch_core_engines(
    # Set up input and output addresses.
    addresses = EngineZmqAddresses(
    handshake_address = get_engine_client_zmq_addr(
    local_handshake_address = handshake_address
    client_handshake_address = None
    with zmq_socket_ctx(local_handshake_address, zmq.ROUTER,
                        bind=True) as handshake_socket:
            local_engine_manager = CoreEngineProcManager(
                EngineCoreProc.run_engine_core,
                vllm_config=vllm_config,
                executor_class=executor_class,
                log_stats=log_stats,
                handshake_address=handshake_address,
                client_handshake_address=client_handshake_address,
        yield local_engine_manager, coordinator, addresses
```

```
local_handshake_address:EngineCore → 前端
handshake_address:前端 → EngineCore
```

```
class CoreEngineProcManager:
    def __init__(
    	context = get_mp_context()
        common_kwargs = {
            "vllm_config": vllm_config,
            "local_client": local_client,
            "handshake_address": handshake_address,
            "executor_class": executor_class,
            "log_stats": log_stats,
        }
        context.Process(target=target_fn,
            name=f"EngineCore_DP{global_index}",
            kwargs=common_kwargs | {
        proc.start()
```

## EngineCoreProc

```
class EngineCoreProc(EngineCore):
    def run_engine_core(*args,
		engine_core = EngineCoreProc(*args, **kwargs)
		engine_core.run_busy_loop()
    def __init__(
    	self.engines_running = False
        with self._perform_handshakes(handshake_address, identity,
                                      local_client, vllm_config,
                                      client_handshake_address) as addresses:
            super().__init__(vllm_config, executor_class, log_stats,
                             executor_fail_callback)
            input_thread = threading.Thread(target=self.process_input_sockets,
			input_thread.start()
            self.output_thread = threading.Thread(
            self.output_thread.start()
    def run_busy_loop(self):
        while True:
            self._process_input_queue()
            self._process_engine_step()
    def _process_input_queue(self):
        while not self.engines_running and not self.scheduler.has_requests() \
            req = self.input_queue.get()
            self._handle_client_request(*req)
    def _handle_client_request(self, request_type: EngineCoreRequestType,
        if request_type == EngineCoreRequestType.ADD:
            req, request_wave = request
            self.add_request(req, request_wave)
    def _process_engine_step(self) -> bool:
        outputs, model_executed = self.step_fn()
    def process_input_sockets(self, input_addresses: list[str],
            while True:
                for input_socket, _ in poller.poll():
                    type_frame, *data_frames = input_socket.recv_multipart(
                    self.input_queue.put_nowait((request_type, request))
/root/autodl-tmp/vllm-ascend-env/lib/python3.10/site-packages/vllm/v1/engine/core.py
class EngineCore:
    def __init__(self,
		self.model_executor = executor_class(vllm_config)
        num_gpu_blocks, num_cpu_blocks, kv_cache_config = \
            self._initialize_kv_caches(vllm_config)
        self.collective_rpc("initialize_cache",
        self.scheduler: SchedulerInterface = Scheduler(
        self.step_fn = (self.step if self.batch_queue is None else
                        self.step_with_batch_queue)
    def collective_rpc(self,
		return self.model_executor.collective_rpc(method, timeout, args,
    def add_request(self, request: Request, request_wave: int = 0):
        self.scheduler.add_request(request)
    def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
        scheduler_output = self.scheduler.schedule()
        model_output = self.execute_model_with_error_logging(
            self.model_executor.execute_model,  # type: ignore
            scheduler_output)
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output)  # type: ignore
	def execute_model_with_error_logging(
		return model_fn(scheduler_output)
```

## UniProcExecutor

```
class ExecutorBase(ABC):
    def __init__(
		self._init_executor()
    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks) -> None:
        self.collective_rpc("initialize_cache",        
class Executor(ExecutorBase):
    def execute_model(
        output = self.collective_rpc("execute_model",
/root/autodl-tmp/vllm-ascend-env/lib/python3.10/site-packages/vllm/executor/uniproc_executor.py
class UniProcExecutor(ExecutorBase):
    def _init_executor(self) -> None:
        self.driver_worker = WorkerWrapperBase(vllm_config=self.vllm_config,
        self.collective_rpc("init_worker", args=([kwargs], ))
        self.collective_rpc("init_device")
        self.collective_rpc("load_model")
    def collective_rpc(self,
        if not non_block:
            return [run_method(self.driver_worker, method, args, kwargs)]
```

## NPUWorker

```
/root/autodl-tmp/vllm-ascend-env/lib/python3.10/site-packages/vllm/worker/worker_base.py
class WorkerBase:
    def __init__(
/root/autodl-tmp/vllm-ascend-env/lib/python3.10/site-packages/vllm/v1/worker/worker_base.py
class WorkerBase(WorkerBaseV0):
	def __init__(
        super().__init__(vllm_config=vllm_config)
class WorkerWrapperBase:
    def init_worker(self, all_kwargs: List[Dict[str, Any]]) -> None:
    	#'vllm_ascend.worker.worker_v1.NPUWorker'
        if isinstance(self.vllm_config.parallel_config.worker_cls, str):
            worker_class = resolve_obj_by_qualname(
                self.vllm_config.parallel_config.worker_cls)
        with set_current_vllm_config(self.vllm_config):
            # To make vLLM config available during worker initialization
            self.worker = worker_class(**kwargs)
    def init_device(self):
        with set_current_vllm_config(self.vllm_config):
            self.worker.init_device()
```

```
/root/autodl-tmp/vllm-ascend-env/lib/python3.10/site-packages/vllm_ascend/worker/worker_v1.py
class NPUWorker(WorkerBase):
	def __init__:
    	register_ascend_customop(vllm_config)
        super().__init__(vllm_config=vllm_config,
    def init_device(self):
        device = self._init_device()
        self.model_runner = NPUModelRunner(self.vllm_config, device)
    def _init_device(self):
        device = torch.device(f"npu:{self.local_rank}")
    def load_model(self) -> None:
            self.model_runner.load_model()
    def execute_model(
        output = self.model_runner.execute_model(scheduler_output,
        
/root/autodl-tmp/vllm-ascend-env/lib/python3.10/site-packages/vllm_ascend/utils.py
#将torch原生CustomOp替换成ascend_customop（替换网络层）
def register_ascend_customop(vllm_config: Optional[VllmConfig] = None):
    for name, op_cls in REGISTERED_ASCEND_OPS.items():
        CustomOp.register_oot(_decorated_op_cls=op_cls, name=name)
```

```
#注册函数torch.ops.vllm.
/root/autodl-tmp/vllm-ascend-env/lib/python3.10/site-packages/vllm/model_executor/layers/fused_moe/layer.py
def moe_forward(
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    return self.forward_impl(hidden_states, router_logits)
direct_register_custom_op(
    op_name="moe_forward",
    op_func=moe_forward,
    mutates_args=["hidden_states"],
    fake_impl=moe_forward_fake,
    tags=(torch.Tag.needs_fixed_stride_order, ),
)

vllm_lib = Library("vllm", "FRAGMENT")  # noqa
def direct_register_custom_op(
    if dispatch_key is None:
        from vllm.platforms import current_platform
        dispatch_key = current_platform.dispatch_key
    import torch.library
    if hasattr(torch.library, "infer_schema"):
        schema_str = torch.library.infer_schema(op_func,
                                                mutates_args=mutates_args)
    else:
        # for pytorch 2.4
        import torch._custom_op.impl
        schema_str = torch._custom_op.impl.infer_schema(op_func, mutates_args)
    my_lib = target_lib or vllm_lib
    my_lib.define(op_name + schema_str, tags=tags)
    my_lib.impl(op_name, op_func, dispatch_key=dispatch_key)
    if fake_impl is not None:
        my_lib._register_fake(op_name, fake_impl)
```

```
select_experts-token_dispatch-mlp-token_combine
```

```
/root/autodl-tmp/vllm-ascend-env/lib/python3.10/site-packages/vllm_ascend/ops/common_fused_moe.py
class AscendFusedMoE(FusedMoE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quant_method = AscendUnquantizedFusedMoEMethod(
        self.quant_method.create_weights(layer=self, **moe_quant_params)
    def forward_impl(self, hidden_states: torch.Tensor,
        hidden_states, router_logits = forward_context.moe_comm_method.prepare(
            hidden_states=hidden_states,
            router_logits=router_logits,
        final_hidden_states = self.quant_method.apply(
        final_hidden_states = forward_context.moe_comm_method.finalize(

class AscendUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):
    def __init__(self, moe: FusedMoEConfig = None):
        super().__init__(moe=moe)
    def apply(self,
        topk_weights, topk_ids = select_experts(
        topk_weights: router weights of shape (num_tokens, top_k).
        topk_ids: selected expert IDs of shape (num_tokens, top_k).
        return moe_comm_method.fused_experts(

/root/autodl-tmp/vllm-ascend-env/lib/python3.10/site-packages/vllm_ascend/ops/moe/moe_comm_method.py
class MoECommMethod(ABC):
    def fused_experts(
        results = self.token_dispatcher.token_dispatch(
        mlp_output = unified_apply_mlp(hidden_states=permuted_hidden_states,
        final_hidden_states = self.token_dispatcher.token_combine(
        
/root/autodl-tmp/vllm-ascend-env/lib/python3.10/site-packages/vllm_ascend/ops/moe/moe_mlp.py
def unified_apply_mlp(hidden_states: torch.Tensor,
        return unquant_apply_mlp(hidden_states=hidden_states,
def unquant_apply_mlp(hidden_states: torch.Tensor,
    gate_up_out = torch_npu.npu_grouped_matmul(
    gate_up_out = torch_npu.npu_swiglu(gate_up_out)
    hidden_states = torch_npu.npu_grouped_matmul(
```

## NPUModelRunner

```
/root/autodl-tmp/vllm-ascend-env/lib/python3.10/site-packages/vllm_ascend/worker/model_runner_v1.py
class NPUModelRunner(LoRAModelRunnerMixin):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        self.attn_backend = get_attn_backend(0,
        self.use_aclgraph = self._use_aclgraph()
        self.aclgraph_batch_sizes = list(
            reversed(self.compilation_config.cudagraph_capture_sizes))
        self.aclgraph_dispatcher = CudagraphDispatcher(self.vllm_config)
    def load_model(self) -> None:
            self.model = get_model(vllm_config=self.vllm_config)
    def execute_model(
        self._update_states(scheduler_output)
        max_query_len) = (self._prepare_inputs(scheduler_output,
        uniform_decode = (max_query_len == self.uniform_decode_query_len) and (
            scheduler_output.total_num_scheduled_tokens
            == self.input_batch.num_reqs * max_query_len)
        aclgraph_runtime_mode, batch_descriptor = \
            self.aclgraph_dispatcher.dispatch(batch_descriptor)
        hidden_states = self._generate_process_reqs_hidden_states(
        model_runner_output = ModelRunnerOutput(

    def _update_states(self, scheduler_output: "SchedulerOutput") -> None:
        # Remove finished requests from the cached states.
        # Remove the finished requests from the persistent batch.
        # Add new requests to the cached states.
        # Add the new or resumed requests to the persistent batch.
	def _prepare_inputs(
        # OPTIMIZATION: Start copying the block table first.
        self.input_batch.block_table.commit_block_table(num_reqs)
		if (self.use_aclgraph and total_num_scheduled_tokens
                <= self.aclgraph_batch_sizes[-1]):
            # Add padding to the batch size.
            num_input_tokens = self.vllm_config.pad_for_cudagraph(
                total_num_scheduled_tokens)
        else:
            # Eager mode.
            num_input_tokens = total_num_scheduled_tokens
        attn_state = self._build_attn_state(num_reqs, num_scheduled_tokens,
        self._prepare_input_ids(total_num_scheduled_tokens, cu_num_tokens)
        self.positions[:num_input_tokens].copy_(
    def _build_attn_state(self, num_reqs, num_scheduled_tokens,
        elif not ascend_config.ascend_scheduler_config.enabled or self.chunked_prefill_enabled:
            attn_state = AscendAttentionState.ChunkedPrefill
    def _generate_process_reqs_hidden_states(self, attn_metadata, with_prefill,
        hidden_states = self.model(

def get_model(*,
    loader = get_model_loader(vllm_config.load_config)
    return loader.load_model(vllm_config=vllm_config,
```

## CudagraphDispatcher

```
class CudagraphDispatcher:
    def dispatch(
        if batch_descriptor in self.cudagraph_keys[CUDAGraphMode.FULL]:
            return CUDAGraphMode.FULL, batch_descriptor
        if non_uniform_key in self.cudagraph_keys[CUDAGraphMode.FULL]:
            return CUDAGraphMode.FULL, non_uniform_key
        if non_uniform_key in self.cudagraph_keys[CUDAGraphMode.PIECEWISE]:
            return CUDAGraphMode.PIECEWISE, non_uniform_key
        return CUDAGraphMode.NONE, None
```

## DefaultModelLoader

```
/root/miniconda3/lib/python3.10/site-packages/vllm/model_executor/model_loader/base_loader.py
class BaseModelLoader(ABC):
    def load_model(self, vllm_config: VllmConfig,
        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                model = initialize_model(vllm_config=vllm_config,
                                         model_config=model_config)
            self.load_weights(model, model_config)
            process_weights_after_loading(model, model_config, target_device)
        return model.eval()
/root/miniconda3/lib/python3.10/site-packages/vllm/model_executor/model_loader/default_loader.py
class DefaultModelLoader(BaseModelLoader):
    def load_weights(self, model: nn.Module,
    	#网络参数
        weights_to_load = {name for name, _ in model.named_parameters()}
        loaded_weights = model.load_weights(
            self.get_all_weights(model_config, model))
    
    def get_all_weights(
    #从磁盘读所有参数
        primary_weights = DefaultModelLoader.Source(
        yield from self._get_weights_iterator(primary_weights)
        
def initialize_model(
	model_class, _ = get_model_architecture(model_config)
    return model_class(vllm_config=vllm_config, prefix=prefix)
```

## Qwen3MoeForCausalLM

```
/root/miniconda3/lib/python3.10/site-packages/vllm/model_executor/models/qwen3_moe.py
#vllm有自己的模型文件，和huggingface独立
class Qwen3MoeForCausalLM(nn.Module, SupportsPP, SupportsLoRA,
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_text_config
        self.model = Qwen3MoeModel(vllm_config=vllm_config,
                                   prefix=maybe_prefix(prefix, "model"))
    def load_weights(self, weights: Iterable[tuple[str,
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
@support_torch_compile
class Qwen3MoeModel(nn.Module):
    def load_weights(self, weights: Iterable[tuple[str,
    	params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
        	# cyy
            if name.startswith("layers.") and int(name.split(".")[1])>=len(self.layers):
                continue
    		for (param_name, weight_name, shard_id) in stacked_params_mapping:
    		else:
                is_expert_weight = False
                for mapping in expert_params_mapping:
                    param = params_dict[name_mapped]
                else:
					param = params_dict[name]
class Qwen3MoeDecoderLayer(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
            self.mlp = Qwen3MoeSparseMoeBlock(vllm_config=vllm_config,
class Qwen3MoeSparseMoeBlock(nn.Module):
    def __init__(
        self.experts = FusedMoE(num_experts=self.n_routed_experts,
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        router_logits, _ = self.gate(hidden_states)
        final_hidden_states = self.experts(hidden_states=hidden_states,
                                           router_logits=router_logits)
```

## CustomOp

```
forward-_forward_method-forward_oot-forward_native-torch.ops.vllm.moe_forward-
						forward_cuda-forward_native
forward_impl
```

```
/root/miniconda3/lib/python3.10/site-packages/vllm/model_executor/layers/fused_moe/layer.py
@CustomOp.register("fused_moe")
class FusedMoE(CustomOp):
    def __init__(
    	compilation_config.static_forward_context[prefix] = self
        quant_method = (UnquantizedFusedMoEMethod(moe) if quant_config is None
        self.quant_method.create_weights(layer=self, **moe_quant_params)
    def create_weights(self, layer: torch.nn.Module, num_experts: int,
        w13_weight = torch.nn.Parameter(torch.empty(
        layer.register_parameter("w13_weight", w13_weight)
    def forward_native(
		fused_output = torch.ops.vllm.moe_forward(
                    hidden_states, router_logits, self.layer_name)
@CustomOp.register("unquantized_fused_moe")
class UnquantizedFusedMoEMethod(FusedMoEMethodBase, CustomOp):
    def __init__(self, moe: FusedMoEConfig):
        self.flashinfer_cutlass_moe_enabled = (
        if self.flashinfer_cutlass_moe_enabled:
        else:
			self.flashinfer_cutlass_moe = None
```

```
class CustomOp(nn.Module):
    def register_oot(cls, _decorated_op_cls=None, name: Optional[str] = None):
        def decorator(op_cls):
            reg_name = name if name is not None else cls.__name__
            assert reg_name not in cls.op_registry_oot, \
                f"Duplicate op name: {reg_name}"
            op_cls.name = reg_name
            cls.op_registry_oot[reg_name] = op_cls
            return op_cls
        return decorator(_decorated_op_cls)
    def __new__(cls, *args, **kwargs):
        if op_name not in cls.op_registry_oot:
            op_cls_to_instantiate = cls
        else:
            op_cls_to_instantiate = cls.op_registry_oot[op_name]
            logger.debug("Instantiating custom op: %s using %s", op_name,
                         str(op_cls_to_instantiate))
        return super().__new__(op_cls_to_instantiate)
    def __init__(self):
        self._forward_method = self.dispatch_forward()
    def forward(self, *args, **kwargs):
        return self._forward_method(*args, **kwargs)
    def dispatch_forward(self):
        enabled = self.enabled()
        if enabled:
            compilation_config.enabled_custom_ops.update([self.__class__.name])
        ...
        elif current_platform.is_out_of_tree():
            return self.forward_oot
    def enabled(cls) -> bool:
        custom_ops = compilation_config.custom_ops
        #['all']
        enabled = f"+{cls.name}" in custom_ops
        disabled = f"-{cls.name}" in custom_ops
        return (CustomOp.default_on() or enabled) and not disabled
    def default_on() -> bool:
        default_on = (compilation_config.level < CompilationLevel.PIECEWISE
                      or not compilation_config.use_inductor)
        count_none = compilation_config.custom_ops.count("none")
        count_all = compilation_config.custom_ops.count("all")
        return default_on and not count_none > 0 or count_all > 0
    def forward_oot(self, *args, **kwargs):
        # By default, we assume that OOT ops are compatible with the
        # PyTorch-native implementation.
        return self.forward_native(*args, **kwargs)
```

## compile

```
/root/autodl-tmp/vllm-ascend-env/lib/python3.10/site-packages/vllm/compilation/decorators.py
def support_torch_compile(
            for k, v in sig.parameters.items():
                if v.annotation in [
                        torch.Tensor, Optional[torch.Tensor],
                        IntermediateTensors, Optional[IntermediateTensors]
                ]:
                    inferred_dynamic_arg_dims[k] = 0
        return _support_torch_compile(cls, inferred_dynamic_arg_dims,
def _support_torch_compile(
	#增加父类
    cls.__bases__ = cls.__bases__ + (TorchCompileWrapperWithCustomDispatcher, )
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = '', **kwargs):
    	old_init(self, vllm_config=vllm_config, prefix=prefix, **kwargs)
        enable_compile = enable_if is None or enable_if(vllm_config)
        TorchCompileWrapperWithCustomDispatcher.__init__(
            self, compilation_level=vllm_config.compilation_config.level)
    cls.__init__ = __init__
    def __call__(self, *args, **kwargs):
    cls.__call__ = __call__
```

```
class CompilationLevel:
    # constants for the levels of the compilation process
    NO_COMPILATION = 0
    DYNAMO_AS_IS = 1
    DYNAMO_ONCE = 2
    PIECEWISE = 3
```

## AutoWeightsLoader

```
/root/miniconda3/lib/python3.10/site-packages/vllm/model_executor/models/utils.py
class AutoWeightsLoader:
    def __init__(
        self.module = module
    def load_weights(
        weights = ((name, weight) for name, weight in weights
                   if not self._can_skip(name))
		autoloaded_weights = set(self._load_module("", self.module, weights))
    def _load_module(
        if module != self.module:
            module_load_weights = getattr(module, "load_weights", None)
            if callable(module_load_weights):    
            	loaded_params = module_load_weights(weights)
        for child_prefix, child_weights in self._groupby_prefix(weights):
		#把前缀提取出来
        	if child_prefix in child_modules:
				yield from self._load_module(prefix,
			elif child_prefix in child_params:
				yield from self._load_param(prefix, child_params[child_prefix],
    def _groupby_prefix(
        weights_by_parts = ((weight_name.split(".", 1), weight_data)
                            for weight_name, weight_data in weights)
        for prefix, group in itertools.groupby(weights_by_parts,
                                               key=lambda x: x[0][0]):
```

## Scheduler

```
class Scheduler(SchedulerInterface):
    def add_request(self, request: Request) -> None:
        self.waiting.add_request(request)
        self.requests[request.request_id] = request
```

