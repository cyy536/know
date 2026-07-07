[vllm-project/llm-compressor: Transformers-compatible library for applying various compression algorithms to LLMs for optimized deployment with vLLM](https://github.com/vllm-project/llm-compressor/tree/main)

```
docker run -itd --ipc host --net host --gpus all --name cyy_accuracy -v ~/qua:/qua pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel bash
```

# quantization

只能有一种量化格式，可以先smooth再gptq，独立

代码很丑，而且是静态图

```
SmoothQuantModifier.init
	找到smooth_layer负责smooth激活，balance_layers负责承载波动
	
GPTQModifier.init
	给量化层注册参数scale和zero

IndependentPipeline:分别处理两个SequentialPipeline

SequentialPipeline
	有参数的设置hook: move param to meta, move buffer to cuda, pre_forward, post_forward
		作用？
	切分成子图，每次处理一个子图，应该是为了降低显存？
	SmoothQuantModifier/GPTQModifier
	
SmoothQuantModifier
    start hook: 记录layernorm.out最大最小值
    for(datalodader):获取activation_scales
    SEQUENTIAL_EPOCH_END:compute scales, smooth(weight)
    for(datalodader):用量化参数计算?
    end:remove_hooks
    
GPTQModifier...
	start hook
	for(datalodader):scales
	SEQUENTIAL_EPOCH_END:update weight
	for(datalodader):用量化参数计算?
	end:remove_hooks
```

```
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor import oneshot

# Select quantization algorithm. In this case, we:
#   * apply SmoothQuant to make the activations easier to quantize
#   * quantize the weights to int8 with GPTQ (static per channel)
#   * quantize the activations to int8 (dynamic per token)
recipe = [
    SmoothQuantModifier(smoothing_strength=0.8),
    GPTQModifier(scheme="W8A8", targets="Linear", ignore=["lm_head"]),
]

# Apply quantization using the built in open_platypus dataset.
#   * See examples for demos showing how to pass a custom calibration set
oneshot(
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    dataset="open_platypus",
    recipe=recipe,
    output_dir="TinyLlama-1.1B-Chat-v1.0-INT8",
    max_seq_length=2048,
    num_calibration_samples=512,
)
    one_shot = Oneshot(**local_args, **kwargs)
    one_shot()
    	self.apply_recipe_modifiers(
            calibration_dataloader=calibration_dataloader,
            recipe_stage=self.recipe_args.stage,
        )
        	session.initialize(
				mod_data = self._lifecycle.initialize(
            		self.recipe = Recipe.create_instance(
                    path_or_modifiers=recipe, target_stage=recipe_stage
                    )
                        return cls.from_modifiers(
                            modifiers=path_or_modifiers, modifier_group_name=
                            modifier_group_name
                        )
                            recipe = cls()
                            recipe.stage = group_name
                            recipe.modifiers = modifiers
                    for mod in self.recipe.modifiers:
            			data = mod.initialize(state=self.state, **kwargs)
                        	self.initialized_ = self.on_initialize(state=state, **kwargs)
            user_pipeline = self.dataset_args.pipeline
			pipeline = CalibrationPipeline.from_modifiers(modifiers, user=user_pipeline)
			#IndependentPipeline
			pipeline(
                self.model,
                calibration_dataloader,
                self.dataset_args,
            )
            session.finalize()
        post_process(
            model_args=self.model_args,
            recipe_args=self.recipe_args,
            output_dir=self.output_dir,
        )
        #Saves the model and tokenizer/processor to the output directory
```

```
_global_session = CompressionSession()
_local_storage = threading.local()
_local_storage.session = _global_session
class CompressionSession:
    def __init__(self):
        self._lifecycle = CompressionLifecycle()
	def event(
		mod_data = self._lifecycle.event(
            event_type=event_type, batch_data=batch_data, loss=loss, **kwargs
        )
        
class CompressionLifecycle:
    def event(
        if global_step is not None:
            self.global_step = global_step
            event = Event(type_=event_type)
            mod_data = []
            for mod in self.recipe.modifiers:
                data = mod.update_event(state=self.state, event=event, **kwargs)
                
class Recipe(BaseModel):

class Modifier(ModifierInterface, HooksMixin):
    def update_event(self, state: State, event: Event, **kwargs):
        self.on_event(state, event, **kwargs)
            
class SmoothQuantModifier(Modifier):
	#找到smooth_layer负责smooth激活，balance_layers负责承载波动
    def on_initialize(self, state: State, **kwargs) -> bool:
        self.mappings = self._infer_mappings_from_model(state.model)
            get_layer_mappings_from_architecture(
                architecture=model.__class__.__name__
            )
            	MAPPINGS_REGISTRY.get(architecture, DEFAULT_SMOOTHQUANT_MAPPINGS)
                	DEFAULT_SMOOTHQUANT_MAPPINGS: List[LayerMap] = [
                        LayerMap(
                            balance_layers=["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"],
                            smooth_layers="re:.*input_layernorm",
                        ),
                        LayerMap(
                            balance_layers=["re:.*gate_proj", "re:.*up_proj"],
                            smooth_layers="re:.*post_attention_layernorm",
                        ),
                    ]
        self.resolved_mappings_ = self._resolve_mappings(state.model)
            for to_balance, to_smooth in self.mappings:
                to_smooth_layers = get_layers(to_smooth, model)
                for layer_name, smooth_layer in to_smooth_layers.items():
                	for balance_suffix in to_balance:
                		_, balance_layer = get_matching_layer(
                            balance_suffix, layer_name, model
                        )
                    # each mapping can contain multiple layers to balance, but only
                    # one layer to smooth
                    mapping = SmoothQuantMapping(
                        layer_name, smooth_layer, balance_layers
                    )
                    #model.layers.0.input_layernorm
                    #smooth_layer负责smooth激活
                    #balance_layers负责承载波动
        self.scales_ = {}
        
    def on_event(self, state: State, event: Event, **kwargs):
        if event.type_ == EventType.CALIBRATION_EPOCH_START:
            if not self.started_:
                self.on_start(state, None)
        if event.type_ == EventType.SEQUENTIAL_EPOCH_END:
            self._apply_smoothing(state.model)
        if event.type_ == EventType.CALIBRATION_EPOCH_END:
            self._apply_smoothing(state.model)
            #continue
            if not self.ended_:
                self.on_end(state, None)  
                	self.remove_hooks()
                	
	#hook: 记录layernorm.out最大最小值
    def on_start(self, state: State, event: Event, **kwargs):
        self.started_ = True
        self._setup_scale_hooks()
        	def create_hook_fn(layer_name):
                def hook_fn(module, inp, out):
                    if isinstance(out, tuple):
                        out = out[0]
                    hidden_dim = out.shape[-1]
                    out = out.view(-1, hidden_dim)
                    latest_mins = torch.min(out, dim=0)[0]
                    latest_maxes = torch.max(out, dim=0)[0]
                    if layer_name in self.scales_:
                        self.scales_[layer_name].min_channel_vals = torch.minimum(
                            self.scales_[layer_name].min_channel_vals, latest_mins
                        )
                        self.scales_[layer_name].max_channel_vals = torch.maximum(
                            self.scales_[layer_name].max_channel_vals, latest_maxes
                        )
                    else:
                        self.scales_[layer_name] = SmoothQuantScale(
                            min_channel_vals=latest_mins, max_channel_vals=latest_maxes
                        )
            for mapping in self.resolved_mappings_:
                name = mapping.smooth_name
                layer = mapping.smooth_layer
                self.register_hook(layer, create_hook_fn(name), "forward")
    
    def _apply_smoothing(self, model: Module):
        for mapping in self.resolved_mappings_:
            if mapping.smooth_name not in self.scales_:
                    continue
            activation_scales = (  # get dynamic range for each activation channel
                self.scales_[mapping.smooth_name].max_channel_vals
                - self.scales_[mapping.smooth_name].min_channel_vals
            )
            smooth_layer = mapping.smooth_layer
            balance_layers = mapping.balance_layers
            scales = self._calculate_smoothing_scales(balance_layers, activation_scales)
           	#s_j = max(|X_j|)^alpha / max(|W_j|)^(1-alpha)
            scales = torch.maximum(
                scales, torch.Tensor([MINIMUM_SMOOTHING_SCALE]).to(scales.device)
            )
            def smooth(module):
                with align_module_device(module):
                    if module in balance_layers:
                        module.weight.mul_(scales.view(1, -1))
                    elif module == smooth_layer:
                        if module.weight.ndim == 1:
                            module.weight.div_(scales)
                        else:
                            module.weight.div_(scales.view(-1, 1))
                        if hasattr(module, "bias") and module.bias is not None:
                            module.bias.div_(scales)
            for layer in balance_layers:
                smooth(layer)
            smooth(smooth_layer)
            del self.scales_[mapping.smooth_name]
            
class QuantizationMixin(HooksMixin):
	#给量化层注册参数scale和zero
    def initialize_quantization(self, model: torch.nn.Module):
        config = self.resolve_quantization_config()
        apply_quantization_config(model, config)
		#给量化层注册scale和zero
        model.apply(self._initialize_observers)
            initialize_observer(module, base_name="weight")
            	arg_name = "weights" if base_name == "weight" else f"{base_name}_activations"
                quantization_args = getattr(quantization_scheme, arg_name, None)
				observer = Observer.load_from_registry(
					retrieved_value = _REGISTRY[parent_class].get(name)
                module.register_module(f"{base_name}_observer", observer)
        model.apply(disable_quantization)
        	module.quantization_enabled = False
        
    def resolve_quantization_config(self) -> QuantizationConfig:
			scheme = {scheme: targets}
			#scheme="W8A8", targets="Linear"
			for idx, key in enumerate(scheme.keys()):
                if is_preset_scheme(key):
                    scheme = preset_name_to_scheme(key, scheme[key])
                    	scheme_args = deepcopy(PRESET_SCHEMES[name])
                    		INT8_W8A8 = dict(
                                weights=QuantizationArgs(
                                    num_bits=8,
                                    type=QuantizationType.INT,
                                    strategy=QuantizationStrategy.CHANNEL,
                                    symmetric=True,
                                    dynamic=False,
                                ),
                                input_activations=QuantizationArgs(
                                    num_bits=8,
                                    type=QuantizationType.INT,
                                    strategy=QuantizationStrategy.TOKEN,
                                    symmetric=True,
                                    dynamic=True,
                                    observer=None,
                                ),
                            )
                        return QuantizationScheme(
                            targets=targets,
                            **scheme_args,
                        )
                group_name = f"group_{idx}"
                config_groups[group_name] = scheme
            return QuantizationConfig(
                config_groups=config_groups,
                kv_cache_scheme=kv_cache_scheme,
                quantization_status=QuantizationStatus.INITIALIZED,
                ignore=ignore,
            )
            
def apply_quantization_config(
#给量化层注册参数scale和zero
    for scheme in config.config_groups.values():
        for target in scheme.targets:
            target_to_scheme[target] = scheme
    for name, submodule in model.named_modules():
        targets = find_name_or_class_matches(name, submodule, target_to_scheme)
            targets = sorted(targets, key=lambda x: ("re:" in x, x))
            if isinstance(targets, Iterable):
                matches = _find_matches(name, targets) + _find_matches(
                    module.__class__.__name__, targets, check_contains
                )
                	elif target == value:
                    	matches.append(target)
		if targets:
			scheme = _scheme_from_targets(target_to_scheme, targets, name)
				target_to_scheme[targets[0]]
			submodule.quantization_scheme = scheme
            names_to_scheme[name] = submodule.quantization_scheme
	apply_quantization_status(model, config.quantization_status)
	    current_status = infer_quantization_status(model)
        if status >= QuantizationStatus.INITIALIZED > current_status:
            model.apply(
                lambda module: initialize_module_for_quantization(
                    module, force_zero_point=force_zero_point_init, scale_dtype=scale_dtype
                )
            )
            	scheme = scheme or getattr(module, "quantization_scheme", None)
            	if scheme.input_activations is not None:
                    _initialize_scale_zero_point(
                        module,
                        "input",
                        scheme.input_activations,
                        force_zero_point=force_zero_point,
                        scale_dtype=scale_dtype,
                    )
                        if quantization_args.dynamic is True:
                            return
                        if base_name == "weight" and weight_shape is not None:
                            if quantization_args.strategy == QuantizationStrategy.CHANNEL:
                                # (output_channels, 1) - only for weights
                                expected_shape = (weight_shape[0], 1)
                        init_scale = Parameter(
                            torch.empty(expected_shape, dtype=scale_dtype, device=device),
                            requires_grad=False,
                        )
                        register_offload_parameter(module, f"{base_name}_scale", init_scale)
                            module.register_parameter(name, parameter)
                        init_zero_point = Parameter(
                            torch.zeros(expected_shape, device=device, dtype=zp_dtype),
                            requires_grad=False,
                        )
                        register_offload_parameter(module, f"{base_name}_zero_point", 
                    											init_zero_point)
        		if scheme.weights is not None:
                    if hasattr(module, "weight"):
                    weight_shape = None
                    if isinstance(module, torch.nn.Linear):
                        weight_shape = module.weight.shape
                    _initialize_scale_zero_point(
                        module,
                        "weight",
                        scheme.weights,
                        weight_shape=weight_shape,
                        force_zero_point=force_zero_point,
                        scale_dtype=scale_dtype,
                    )
                module.quantization_status = QuantizationStatus.INITIALIZED
                wrap_module_forward_quantized(module, scheme)
                    def wrapped_forward(self, *args, **kwargs):
                	setattr(module, "forward", bound_wrapped_forward)
        if current_status < status >= QuantizationStatus.COMPRESSED > current_status:
            model.apply(compress_quantized_weights)
   
class GPTQModifier(Modifier, QuantizationMixin):
    def on_initialize(self, state: State, **kwargs) -> bool:
    	QuantizationMixin.initialize_quantization(self, state.model)
    def resolve_quantization_config(self) -> QuantizationConfig:
        config = super().resolve_quantization_config()

class CalibrationPipeline(ABC, RegistryMixin):
    def from_modifiers(
        user = standardize_lookup_name(user) if user else None
        inferred = standardize_lookup_name(cls._infer_pipeline(modifiers))
    	independent = standardize_lookup_name("independent")
        if user == independent:
            inferred = independent
		return cls.load_from_registry(pipeline)
    def _infer_pipeline(modifiers: List[Modifier]) -> str:
        return "sequential"

class IndependentPipeline(CalibrationPipeline):
    def __call__(
        with patch_attr(session.lifecycle.recipe, "modifiers", None):
        #Patch the value of an object attribute. Original value is restored upon exit
            for modifier in modifiers:
                session.lifecycle.recipe.modifiers = [modifier]
                pipeline = CalibrationPipeline.from_modifiers([modifier])
                #SequentialPipeline
                pipeline(model, dataloader, dataset_args)

class SequentialPipeline(CalibrationPipeline):
    def __call__(
            dispatch_for_sequential(model)
                remove_dispatch(model)
                    module.to("cpu")
                offloaded_dispatch(model, execution_device=torch.device("cuda:0"))
                #有参数的设置hook: move param to meta, move buffer to cuda, pre_forward, post_forward
                    state_dict = module.state_dict()
                    weights_map = OffloadedWeightsLoader(state_dict=state_dict, device=offload_device)
                    tied_params = find_tied_parameters(module)
                        all_named_parameters = {name: param for name, param in model.named_parameters(remove_duplicate=False)}
                        no_duplicate_named_parameters = {name: param for name, param in model.named_parameters(remove_duplicate=True)}
                        //id相同的只出现一次
                        tied_param_names = set(all_named_parameters.keys()) - set(no_duplicate_named_parameters.keys())
                        //多于1次的名字
                        tied_param_groups = {}
                        for tied_param_name in tied_param_names:
                            tied_param = all_named_parameters[tied_param_name]
                            for param_name, param in no_duplicate_named_parameters.items():
                                if param is tied_param:
                                    if param_name not in tied_param_groups:
                                        tied_param_groups[param_name] = []
                                    tied_param_groups[param_name].append(tied_param_name)
                        //把id相同的name放在一组
                        return FindTiedParametersResult([sorted([weight] + list(set(tied))) for weight, tied in tied_param_groups.items()])
                    tied_params_map = {}
                    for group in tied_params:
                        for param_name in group:
                            data_ptr = attrgetter(param_name)(module).data_ptr()
                            tied_params_map[data_ptr] = {}
                    //ptr:{}
                    attach_align_device_hook(
                        module,
                        execution_device=execution_device,
                        offload=True,#offloaded after the forward pass.
                        weights_map=weights_map,
                        tied_params_map=tied_params_map,
                    )
                        directs = named_module_tensors(module)
                            include_buffers: bool = True, recurse: bool = False
                            remove_non_persistent: bool = False
                            yield from module.named_parameters(recurse=recurse)
                            if include_buffers:
                                non_persistent_buffers = set()
                                if remove_non_persistent:
                                    non_persistent_buffers = get_non_persistent_buffers(module, recurse=recurse)
                                for named_buffer in module.named_buffers(recurse=recurse):
                                    name, _ = named_buffer
                                    if name not in non_persistent_buffers:
                                        yield named_buffer
                        if len(list(directs)) > 0 or full_offload:
                            if weights_map is not None:
                                prefix = f"{module_name}." if len(module_name) > 0 else ""
                                prefixed_weights_map = PrefixedDataset(weights_map, prefix)
                            hook = AlignDevicesHook(
                                execution_device=execution_device,
                                offload=offload,
                                weights_map=prefixed_weights_map,
                                offload_buffers=offload_buffers,
                                place_submodules=full_offload,
                                skip_keys=skip_keys,
                                tied_params_map=tied_params_map,
                            )
                            add_hook_to_module(module, hook, append=True)
                            #move param to meta, move buffer to cuda, pre_forward, post_forward
                                module = hook.init_hook(module)
                                module._hf_hook = hook
                                def new_forward(module, *args, **kwargs):
                                    args, kwargs = module._hf_hook.pre_forward(module, *args, **kwargs)
                                    if module._hf_hook.no_grad:
                                        with torch.no_grad():
                                            output = module._old_forward(*args, **kwargs)
                                    else:
                                        output = module._old_forward(*args, **kwargs)
                                    return module._hf_hook.post_forward(module, output)
                                module.forward = functools.update_wrapper(functools.partial(new_forward, module), old_forward)

                        for child_name, child in module.named_children():
                            child_name = f"{module_name}.{child_name}" if len(module_name) > 0 else child_name
                            attach_align_device_hook(
                                child,
            model_device = get_execution_device(model)  
            	return cast_to_device(submodule._hf_hook.execution_device)
           	modifiers = session.lifecycle.recipe.modifiers
        	sequential_targets = get_sequential_targets(modifiers, model, dataset_args)
                if sequential_targets is None:
                    get_no_split_params(model)
                    	model = maybe_get_wrapped(model)
                    	no_split_modules = model._get_no_split_modules("auto")
                    	#Get the modules of the model that should not be spit when using device_map
            #['LlamaDecoderLayer']
            sample_input = next(iter(dataloader))
        	subgraphs = trace_subgraphs(model, sample_input, sequential_targets, ignore)
        		targets = match_modules(model, sequential_targets)
        		ancestors = get_sequential_ancestors(model, targets)
        		#LlamaForCausalLM,LlamaModel,ModuleList
        		    ancestors = set()
                    def is_ancestor(module: Module) -> bool:
                        if module in ancestors or module in targets:
                            return True
                        _is_ancestor = any([is_ancestor(child) for child in module.children()])
                        if _is_ancestor:
                            ancestors.add(module)
                        return _is_ancestor
                    is_ancestor(model)
                    return ancestors
                offloaded = set(m for m in model.modules() if has_offloaded_params(m))
                tracer = SequentialTracer(ancestors, offloaded)
                concrete_args = populate_concrete_args(model, sample_input)
                #{'position_ids': None, 'past_key_values': None, 'inputs_embeds': None, 'labels': None, 'use_cache': False, 'cache_position': None, 'logits_to_keep': 0, 'kwargs': {}}
                autowrap_forwards(ancestors, ignore)
                    for module in modules:
                        if not isinstance(module, (torch.nn.ModuleList, torch.nn.ModuleDict)):
                            stack.enter_context(autowrap_forward(module, ignore))
                            #Replace the `forward` method of the given module with a recompiled version
                graph = GraphModule(
                    model,
                    tracer.trace(
                partitions = topological_partition(graph, targets)
                	target_nodes = find_target_nodes(graph, targets)
                    queue = deque(
                        node
                        for node in graph.graph.nodes
                        if remaining_indegrees[node] == 0 and node.op != "get_attr"
                    )
                    while len(queue) > 0:
                        node = queue.popleft()
                        partitions[partition_index].append(node)
                        if node in target_nodes:#不同LlamaDecoderLayer放在不同组
                            partition_index += 1
                            partitions.append([])
                        for user in node.users:
                            remaining_indegrees[user] -= 1
                            if remaining_indegrees[user] == 0:
                                queue.append(user)
                    #[[input_ids, attention_mask, wrapped_0, wrapped_1, getitem, wrapped_3, getitem_1, getitem_2, wrapped_4, getitem_3, wrapped_5, model_rotary_emb, model_layers_0], [model_layers_1], [model_layers_2], [model_layers_3], [model_layers_4], [model_layers_5], [model_layers_6], [model_layers_7], [model_layers_8], [model_layers_9], [model_layers_10], [model_layers_11], [model_layers_12], [model_layers_13], [model_layers_14], [model_layers_15], [model_layers_16], [model_layers_17], [model_layers_18], [model_layers_19], [model_layers_20], [model_layers_21], [model_norm, getitem_4, lm_head]]
                subgraphs = partition_graph(model, partitions)
                #Convert each partition into a Subgraph
			LifecycleCallbacks.calibration_epoch_start()
			disable_qac = any(
                type(mod).__name__ in ["GPTQModifier", "AWQModifier"]
                for mod in session.lifecycle.recipe.modifiers
            )
            activations = IntermediatesCache.from_dataloader(dataloader, model_device)
			for subgraph_index, subgraph in enumerate(subgraphs):            
            	for batch_idx in tqdm(range(len(dataloader)), desc=calib_desc):
                    inputs = activations.fetch(batch_idx, subgraph.input_names)
                    subgraph.forward(model, **inputs)
                #获取scale, zero的大小
                LifecycleCallbacks.sequential_epoch_end()
                with HooksMixin.disable_hooks():
                    for batch_idx in tqdm(range(len(dataloader)), desc=prop_desc):
                    inputs = activations.fetch(batch_idx, subgraph.input_names)
                    output = subgraph.forward(model, **inputs)
                    if subgraph_index < num_subgraphs - 1:
                        activations.update(batch_idx, output)
                        activations.delete(batch_idx, subgraph.consumed_names)
                 #量化参数
            LifecycleCallbacks.calibration_epoch_end()

class AlignDevicesHook(ModelHook):
	#move param to meta, move buffer to cuda
    def init_hook(self, module):
            for name, _ in named_module_tensors(
                module, include_buffers=self.offload_buffers, recurse=self.place_submodules, remove_non_persistent=True
            ):
            	set_module_tensor_to_device(module, name, "meta")
            		old_value = getattr(module, tensor_name)
                    param = module._parameters[tensor_name] if tensor_name in module._parameters else None
        			param_cls = type(param)
    				new_value = old_value.to(device, non_blocking=non_blocking)
    				if is_buffer:
            			module._buffers[tensor_name] = new_value
                    elif value is not None or not check_device_same(torch.device(device), module._parameters[tensor_name].device):
						new_value = param_cls(new_value, requires_grad=old_value.
                                    requires_grad).to(
                            device, non_blocking=non_blocking
                        )
                        module._parameters[tensor_name] = new_value
                    if clear_cache and device != "cpu":
        				clear_device_cache()
        					torch.cuda.empty_cache()
            for name, _ in module.named_buffers(recurse=self.place_submodules):
                set_module_tensor_to_device(
                module, name, self.execution_device, tied_params_map=self.tied_params_map
                )
```

# inference

```
from vllm import LLM
model = LLM("TinyLlama-1.1B-Chat-v1.0-INT8")
output = model.generate("My name is")
```

# lm_eval

[EleutherAI/lm-evaluation-harness: A framework for few-shot evaluation of language models.](https://github.com/EleutherAI/lm-evaluation-harness)

```
lm_eval --model vllm \
  --model_args pretrained="./Meta-Llama-3-8B-Instruct-W8A8-Dynamic-Per-Token",add_bos_token=true \
  --tasks gsm8k \
  --num_fewshot 5 \
  --limit 250 \
  --batch_size 'auto'
```

```
from lm_eval import tasks, simple_evaluate
from lm_eval.models.huggingface import HFLM
lm = HFLM(
    pretrained=model,
    backend="causal",
    device="cuda",
    batch_size=args.batch_size,
    tokenizer=tokenizer,
    max_lengt=max_length,
)
t_results = simple_evaluate(
    lm,
    tasks=task_names,
    num_fewshot=args.num_fewshot,
    batch_size=args.batch_size,
)
	task_dict = get_task_dict(tasks, task_manager)
		task_name_from_string_dict = task_manager.load_task_or_group(
            string_task_name_list
        )
            all_loaded_tasks = dict(
                collections.ChainMap(*map(self._load_individual_task_or_group, task_list))
            )
            	task_config = self._get_config(name_or_config)
            	return _load_task(task_config, task=name_or_config)
            		task_object = ConfigurableTask(config=config)
            			self.download(self.config.dataset_kwargs)
            				#hellaswag None {'trust_remote_code': True}
            				self.dataset = datasets.load_dataset(
                                path=self.DATASET_PATH,
                                name=self.DATASET_NAME,
                                **dataset_kwargs if dataset_kwargs is not None else {},
                            )
            		return {task: task_object}
	task_dict = _adjust_config(task_dict)
    results = evaluate(
        lm=lm,
        task_dict=task_dict,
```

