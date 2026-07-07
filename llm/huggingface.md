# dataset

```、
def load_dataset(
    path: str,
    name: Optional[str] = None,
    data_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
```

```
path is a dataset repository on the HF hub, a local directory
name, data_dir: 子集
cache_dir: Defaults to `"~/.cache/huggingface/datasets"`
```

```
HF_HOME=~/.cache/huggingface/

模型和数据集原始文件都放在 HF_HUB_CACHE，没有会重新下载
HF_HUB_CACHE=~/.cache/huggingface/hub/

HF_DATASETS_CACHE 缓存数据集处理后的中间结果，没有会重新处理
HF_DATASETS_CACHE=~/.cache/huggingface/datasets/

export HF_TOKEN="你的_huggingface_token"

wget --header="Authorization: Bearer hf_xxx"
```

# tokenizer

指定tokenizer_type->tokenizer_class

tokenizer_config.json.tokenizer_class: transformers原生支持或仓库python文件（需要tokenizer_config.json.auto_map）

```
from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs)
	tokenizer_type = kwargs.pop("tokenizer_type", None)
	if tokenizer_type is not None:
		tokenizer_class_tuple = TOKENIZER_MAPPING_NAMES.get(tokenizer_type, None)
		tokenizer_class_name, tokenizer_fast_class_name = tokenizer_class_tuple
		if tokenizer_fast_class_name is not None:
        	tokenizer_class = tokenizer_class_from_name(tokenizer_fast_class_name)
		if tokenizer_class is None:
        	tokenizer_class = tokenizer_class_from_name(tokenizer_class_name)
		return tokenizer_class.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)	
	tokenizer_config = get_tokenizer_config(pretrained_model_name_or_path, **kwargs)
        resolved_config_file = cached_file(
            pretrained_model_name_or_path,
            TOKENIZER_CONFIG_FILE,#"tokenizer_config.json"
        #Tries to locate a file in a local folder and repo, downloads and cache it if necessary.
        with open(resolved_config_file, encoding="utf-8") as reader:
            result = json.load(reader)
	config_tokenizer_class = tokenizer_config.get("tokenizer_class")
	tokenizer_auto_map = None
    if "auto_map" in tokenizer_config:
        if isinstance(tokenizer_config["auto_map"], (tuple, list)):
            # Legacy format for dynamic tokenizers
            tokenizer_auto_map = tokenizer_config["auto_map"]
        else:
        	tokenizer_auto_map = tokenizer_config["auto_map"].get("AutoTokenizer", None)
	has_remote_code = tokenizer_auto_map is not None
	has_local_code = type(config) in TOKENIZER_MAPPING or (
	#这个 tokenizer 已经被 Transformers 原生支持，无需加载远程代码
	if has_remote_code and trust_remote_code:
    	if use_fast and tokenizer_auto_map[1] is not None:
        	class_ref = tokenizer_auto_map[1]
        else:
        	class_ref = tokenizer_auto_map[0]
        tokenizer_class = get_class_from_dynamic_module(class_ref, pretrained_model_name_or_path, **kwargs)
        return tokenizer_class.from_pretrained(
	elif config_tokenizer_class is not None:
		return tokenizer_class.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
```

```
class BaichuanTokenizer(PreTrainedTokenizer):
class PreTrainedTokenizer(PreTrainedTokenizerBase):
#    Base class for all slow tokenizers.
class PreTrainedTokenizerBase(SpecialTokensMixin, PushToHubMixin):
#    Base class for [`PreTrainedTokenizer`] and [`PreTrainedTokenizerFast`].
    def __call__(
    #Main method to tokenize and prepare for the model one or several sequence(s)
    #{'input_ids':tensor,'attention_mask':tensor}
    def apply_chat_template(
    #Converts a list of dictionaries with `"role"` and `"content"` keys to a list of token ids. This method is intended for use with chat models,
    #应该只能处理一个句子
    messages = [
        {"role": "user", "content": "你好！"},
        {"role": "assistant", "content": "你好！很高兴见到你。"},
        {"role": "user", "content": "你能做什么？"}
    ]
```

初始化参数：

model_max_length，声明模型支持的最大序列长度，通常由模型架构决定

padding_side，应该选left，因为自回归只考虑最后一个token

If models don’t have a padding token set one can use the EOS token as padding token

调用参数：

huggingface要求不同seq长度相同

padding

```
True or 'longest': Pad to the longest sequence in the batch (or no padding if only a single sequence is provided).
'max_length': Pad to a maximum length specified with the argument max_length or to the maximum acceptable input length for the model if that argument is not provided.
```

# model

4.43.1

所有 AutoModel* 类都继承自同一个基类 _BaseAutoModelClass

config.json.model_type: transformers原生支持或仓库python文件（需要config.json.auto_map）

config_class.from_dict：config.json，参数

model_class: model_mapping[type(config)]==config.json.architectures或config.auto_map

get_checkpoint_shard_files

torch.set_default_dtype(torch_dtype)

model = cls(config,

restore default dtype

_load_pretrained_model(device_map)

model.eval()

load_generation_config

```
class AutoModelForCausalLM(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_CAUSAL_LM_MAPPING

AutoModelForCausalLM = auto_class_update(AutoModelForCausalLM, head_doc="causal language modeling")

class _BaseAutoModelClass:
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
		config, kwargs = AutoConfig.from_pretrained(
        	config_dict, unused_kwargs = PretrainedConfig.get_config_dict(pretrained_model_name_or_path, **kwargs)
        		config_dict, kwargs = cls._get_config_dict(pretrained_model_name_or_path, **kwargs)
        			resolved_config_file = cached_file(
                        pretrained_model_name_or_path,
                        configuration_file,
					config_dict = cls._dict_from_json_file(resolved_config_file)
                        with open(json_file, "r", encoding="utf-8") as reader:
            has_remote_code = "auto_map" in config_dict and "AutoConfig" in config_dict["auto_map"]
            has_local_code = "model_type" in config_dict and config_dict["model_type"] in CONFIG_MAPPING
            if has_remote_code and trust_remote_code:
                class_ref = config_dict["auto_map"]["AutoConfig"]
                config_class = get_class_from_dynamic_module(
                return config_class.from_pretrained(
            elif "model_type" in config_dict:
                config_class = CONFIG_MAPPING[config_dict["model_type"]]  
                #transformers.models.qwen2.configuration_qwen2.Qwen2Config
                return config_class.from_dict(config_dict, **unused_kwargs)
				#config.json，参数
        has_remote_code = hasattr(config, "auto_map") and cls.__name__ in config.auto_map
        has_local_code = type(config) in cls._model_mapping.keys()
        if has_remote_code and trust_remote_code:
            class_ref = config.auto_map[cls.__name__]
            return model_class.from_pretrained(
		elif type(config) in cls._model_mapping.keys():
            model_class = _get_model_class(config, cls._model_mapping)
				supported_models = model_mapping[type(config)]
				#transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM
				name_to_model = {model.__name__: model for model in supported_models}
				#Qwen2ForCausalLM
				architectures = getattr(config, "architectures", [])
                for arch in architectures:
                    if arch in name_to_model:
                        return name_to_model[arch]
			return model_class.from_pretrained(
                pretrained_model_name_or_path, *model_args, config=config, **hub_kwargs, **kwargs
            )
            #{'torch_dtype': torch.float16, 'device_map': 'cuda:0'}
            
    def from_config(cls, config, **kwargs):
		model_class = _get_model_class(config, cls._model_mapping)
		return model_class._from_config(config, **kwargs)
```

```
class Qwen2Config(PretrainedConfig):
    def __init__(
        self.use_cache = use_cache
        super().__init__(
        
class PretrainedConfig(PushToHubMixin):
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs) -> "PretrainedConfig":
        config_dict["attn_implementation"] = kwargs.pop("attn_implementation", None)
        config = cls(**config_dict)
        for key, value in kwargs.items():
        	setattr(config, key, value)
        	if key != "torch_dtype":
            	to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)
        if return_unused_kwargs:
            return config, kwargs
		#{'torch_dtype': torch.float16, 'device_map': 'cuda:0'}
		
    def __init__(self, **kwargs):
        self.torch_dtype = kwargs.pop("torch_dtype", None)  # Only used by PyTorch models
		self.torch_dtype = getattr(torch, self.torch_dtype)
		self._attn_implementation_internal = kwargs.pop("attn_implementation", None)
		for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def _attn_implementation(self):
        # This property is made private for now (as it cannot be changed and a PreTrainedModel.use_attn_implementation method needs to be implemented.)
        if hasattr(self, "_attn_implementation_internal"):
            if self._attn_implementation_internal is None:
                # `config.attn_implementation` should never be None, for backward compatibility.
                return "eager"
            else:
                return self._attn_implementation_internal
        else:
            return "eager"
```

```
class ModuleUtilsMixin:
	@property
    def device(self) -> torch.device:
        """
        `torch.device`: The device on which the module is (assuming that all the module parameters are on the same
        device).
        """
        return get_parameter_device(self)
        	return next(parameter.parameters()).device

class PreTrainedModel(nn.Module, ModuleUtilsMixin, GenerationMixin, PushToHubMixin, PeftAdapterMixin):
    @wraps(torch.nn.Module.to)
    def to(self, *args, **kwargs):
    
    def from_pretrained(
    	torch_dtype = kwargs.pop("torch_dtype", None)
    	device_map = kwargs.pop("device_map", None)
        if isinstance(device_map, torch.device):
            device_map = {"": device_map}
        elif isinstance(device_map, str) and device_map not in ["auto", "balanced", "balanced_low_0", "sequential"]:
            device_map = {"": torch.device(device_map)}				
		is_local = os.path.isdir(pretrained_model_name_or_path)
        if is_local:
        	elif use_safetensors is not False and os.path.isfile(
				# Load from a sharded safetensors checkpoint
                archive_file = os.path.join(
                    pretrained_model_name_or_path, subfolder, _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant)
                )
                '/.../model.safetensors.index.json'
                is_sharded = True
		if is_local:
        	resolved_archive_file = archive_file
        if is_sharded:
        #a list of files
            resolved_archive_file, sharded_metadata = get_checkpoint_shard_files(
                pretrained_model_name_or_path,
                resolved_archive_file,
        dtype_orig = cls._set_default_torch_dtype(torch_dtype)   
        	dtype_orig = torch.get_default_dtype()
            torch.set_default_dtype(dtype)
            return dtype_orig
		if is_sharded:
        	loaded_state_dict_keys = sharded_metadata["all_checkpoint_keys"]
        init_contexts = [no_init_weights(_enable=_fast_init)]
        init_contexts.append(init_empty_weights())
        with ContextManagers(init_contexts):
        	model = cls(config, *model_args, **model_kwargs)
		#device(type='meta')
		if dtype_orig is not None:
        	torch.set_default_dtype(dtype_orig)
		(model, ) = cls._load_pretrained_model(
			model,
            loaded_state_dict_keys,  
            resolved_archive_file,
			device_map=device_map,
        #device(type='cuda', index=0)
        model.eval()
		model.generation_config = GenerationConfig.from_pretrained(
			pretrained_model_name_or_path,
		if device_map is not None:
			device_map_kwargs = {
                "device_map": device_map,
            if not is_fsdp_enabled() and not is_deepspeed_zero3_enabled():
                dispatch_model(model, **device_map_kwargs)
                    model.hf_device_map = dict(device_map)
                    
    def _from_config(cls, config, **kwargs):
        config = copy.deepcopy(config)  # We do not want to modify the config inplace in _from_config.
        config._attn_implementation = kwargs.pop("attn_implementation", None)
        # config被kwargs覆盖了
        config = cls._autoset_attn_implementation(
		model = cls(config, **kwargs)
```

```
class Qwen2PreTrainedModel(PreTrainedModel):
    def _init_weights(self, module):

class LlamaForCausalLM(LlamaPreTrainedModel):
    def __init__(self, config):
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
		#lm_head一般和任务相关
    def prepare_inputs_for_generation(
    #产生position_ids
		input_ids = input_ids[:, cache_position]
		if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]
		model_inputs = {"input_ids": input_ids.contiguous()}
        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        
class LlamaModel(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        #position_ids转成position_embeddings可以延后至Attention，例如qwen2,baichuan2
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
		if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )
        	if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
				if AttentionMaskConverter._ignore_causal_mask_sdpa(
        			return None
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)
        
class LlamaAttention(nn.Module):
    def forward(
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
```

```
LLAMA_ATTENTION_CLASSES = {
    "eager": LlamaAttention,
    "flash_attention_2": LlamaFlashAttention2,
    "sdpa": LlamaSdpaAttention,
}
```

```
class GenerationMixin:
    @torch.no_grad()
    def generate(
        generation_config, model_kwargs = self._prepare_generation_config
            (generation_config, **kwargs)
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
        #'input_ids'
        #{'attention_mask': tensor}
        model_kwargs["use_cache"] = generation_config.use_cache
        cache_name = "past_key_values"
        past = model_kwargs.get(cache_name, None)
        if past is None:
            model_kwargs[cache_name] = (DynamicCache()
        generation_mode = generation_config.get_generation_mode(assistant_model)
        #<GenerationMode.GREEDY_SEARCH: 'greedy_search'>
        result = self._sample(
            this_peer_finished = False
            unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
            model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)
                cur_len = input_ids.shape[-1]
                model_kwargs["cache_position"] = torch.arange(past_length, cur_len, device=input_ids.device)
            while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
                model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
                outputs = self(**model_inputs, return_dict=True)
                next_token_logits = outputs.logits[:, -1, :].clone()
                model_kwargs = self._update_model_kwargs_for_generation(
                unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
                this_peer_finished = unfinished_sequences.max() == 0
```

max_length=输入+输出

max_new_tokens=输出

# .

```
vae = AutoencoderDC.from_pretrained("/dynamic_batch/SANA1.5_1.6B_1024px_diffusers", subfolder="vae", torch_dtype=torch.bfloat16)
```

# safetensors

```
https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/tools/safetensors_filter.py
可用于缩小模型参数层数
```

