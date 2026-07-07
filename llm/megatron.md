nvidia

Megatron Core: TP, PP, DP, EP, CP

Megatron Bridge - Bidirectional converter for interoperability between Hugging Face and Megatron checkpoints

apex: A PyTorch Extension: Tools for easy mixed precision and distributed training in Pytorch

```
            args.cuda_graph_impl = "transformer_engine"
```

```
    from transformer_engine.pytorch.optimizers import FusedAdam as Adam
    from transformer_engine.pytorch.optimizers import FusedSGD as SGD

    USING_PYTORCH_OPTIMIZER = False
except ImportError:
    try:
        from apex.optimizers import FusedAdam as Adam
        from apex.optimizers import FusedSGD as SGD

        USING_PYTORCH_OPTIMIZER = False
    except ImportError:
        warnings.warn(
            f'Transformer Engine and Apex are not installed. Falling back to Torch optimizers.'
        )

        # Apex's FusedAdam is a drop-in replacement for torch's AdamW.
        # pylint: disable-next=line-too-long.
        # See https://github.com/NVIDIA/apex/blob/7b73b12361068a10b0f44844534613f252a5ea75/apex/optimizers/fused_adam.py#L16.
        from torch.optim import SGD
        from torch.optim import AdamW as Adam

```

```
class TELayerNormColumnParallelLinear(te.pytorch.LayerNormLinear):
```



```
cuda_12.8.r12.8/compiler.35583870_0

git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
pip install pybind11
修改cu版本："transformer-engine[pytorch,core_cu12]>=2.9.0a0,<2.12.0",
pip install --no-build-isolation -e .[mlm,dev]
(pyproject.toml去掉emerging_optimizers)pip 找不到

Name: torch
Version: 2.10.0+cu128
Name: megatron-core
Version: 0.16.0rc0
```

# train_llama3_8b_h100_fp8.sh

```
--no-gradient-accumulation-fusion
```

# pretrain_gpt.py///

```
def forward_step(data_iterator, model: GPTModel, return_schedule_plan: bool = False):
        tokens, labels, loss_mask, attention_mask, position_ids, packed_seq_params = get_batch(data_iterator, vp_stage)
                output_tensor = model(
DistributedDataParallel(
  (module): Float16Module(
    (module): GPTModel(
```

```
def get_batch(data_iterator, vp_stage: Optional[int] = None):

```



```
def get_embedding_ranks(pp_ranks: List[int]):
    """Get the embedding ranks."""
    embedding_ranks = [pp_ranks[0]]
    if len(pp_ranks) > 1:
        args = get_args()
        if not args.untie_embeddings_and_output_weights:
            embedding_ranks.append(pp_ranks[-1])
        config = core_transformer_config_from_args(args)
        mtp_ranks = get_mtp_ranks(pp_ranks, config)
        embedding_ranks.extend(mtp_ranks)
    embedding_ranks = list(set(embedding_ranks))
    embedding_ranks = sorted(embedding_ranks)
    return embedding_ranks
```

```
from model_provider import model_provider
if __name__ == "__main__":
    train_valid_test_datasets_provider.is_distributed = True
    pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)
    pretrain(
        train_valid_test_datasets_provider,
        partial(model_provider, gpt_builder),
        ModelType.encoder_or_decoder,
        forward_step,
```

```
def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):

```

# core

## distributed

data_parallel_base.py

```
class _BaseDataParallel(MegatronModule):
    def __init__(self, config: TransformerConfig, module: torch.nn.Module):
        super().__init__(config=config)
        self.module = module
    def forward(self, *inputs, **kwargs):
        """
        Calls the wrapped module's forward() method.
        """
        return self.module(*inputs, **kwargs)
```

### distributed_data_parallel.py

```
self.buffers, self.bucket_groups
self.expert_parallel_buffers, self.expert_parallel_bucket_groups 
```

一种数据类型一个buffer

为每个gradient accumulator function注册hook

为每个module注册forward_pre_hook

```
class DistributedDataParallel(_BaseDataParallel):
    def __init__(
        super().__init__(config=config, module=module)
        process_group_dict = ProcessGroupCollection.setup_process_groups_for_ddp(
        self.dp_group = process_group_dict['dp_group']
        for name, param in self.module.named_parameters():
            if not param.requires_grad:
                continue

            # Track params with grad to enable direct setting
            # of param.grad_added_to_main_grad
            self.params_with_grad.append(param)

            param.grad_added_to_main_grad = False
            param_to_name[param] = name

            if getattr(param, 'allreduce', True):
                dense_params.append(param)
            else:
                expert_parallel_params.append(param)

        def _allocate_buffers_for_parameters(
            for param in input_params:
                params = param_and_grad_dtype_to_params.get((param_dtype, grad_dtype), [])
                params.append(param)
                param_and_grad_dtype_to_params[(param_dtype, grad_dtype)] = params
                offset = param_and_grad_dtype_to_offsets.get((param.dtype, grad_dtype), 0)
                param_and_grad_dtype_to_offsets[(param.dtype, grad_dtype)] = offset + 1
                #bug，都是从0开始的连续数
                indices = param_and_grad_dtype_to_indices.get((param_dtype, grad_dtype), [])
                indices.append(offset)
                param_and_grad_dtype_to_indices[(param_dtype, grad_dtype)] = indices
            for (param_dtype, grad_dtype), params in param_and_grad_dtype_to_params.items():
                buffers.append(
                    _ParamAndGradBuffer(
            bucket_groups = partition_buckets(buffers, force_single_bucket_group=disable_bucketing)
            if self.ddp_config.use_distributed_optimizer and self.ddp_config.overlap_param_gather:
                num_bucket_groups = len(bucket_groups)
                for i in range(1, num_bucket_groups):
                    bucket_groups[num_bucket_groups - i].next_param_gather_bucket_group = (
                        bucket_groups[num_bucket_groups - i - 1]
                    )
            for bucket_group in bucket_groups:
                for bucket in bucket_group.buckets:
                    for param in bucket.params_list:
                        self.param_to_bucket_group[param] = bucket_group
		# Allocate the param+grad buffers for dense params' grads.
        self.buffers, self.bucket_groups = _allocate_buffers_for_parameters(
            dense_params, self.intra_dp_cp_group, gradient_scaling_factor=gradient_scaling_factor
        )

        # Allocate separate param+grad buffers for expert parallel params' grads.
        self.expert_parallel_buffers, self.expert_parallel_bucket_groups = (
            _allocate_buffers_for_parameters(
                expert_parallel_params,
                self.intra_expt_dp_group,
                gradient_scaling_factor=expert_gradient_scaling_factor,
            )
        )
        for param in self.module.parameters():
            if param.requires_grad:
                    # Expand so we get access to grad_fn.
                    param_tmp = param.expand_as(param)
                    # Get the gradient accumulator function.
                    grad_acc = param_tmp.grad_fn.next_functions[0][0]
                    grad_acc.register_hook(self._make_backward_post_hook(param))
                    self.grad_accs.append(grad_acc)
        self.use_forward_hook = (
            self.ddp_config.use_distributed_optimizer and self.ddp_config.overlap_param_gather
        )
        if self.use_forward_hook:
            self.enable_forward_pre_hook()
    def _make_backward_post_hook(self, param: torch.nn.Parameter):
        def hook(*unused):
                if param.grad is not None and (
                    not param.grad_added_to_main_grad or getattr(param, 'zero_out_wgrad', False)
                ):
                    param.main_grad.add_(param.grad.data)
                param.grad = None
                if self.ddp_config.overlap_grad_reduce:
                    self.param_to_bucket_group[param].register_grad_ready(
                        param, self.force_all_reduce
                    )
    def enable_forward_pre_hook(self):
        for module in self.module.modules():
            self.remove_forward_pre_hook_handles[module] = module.register_forward_pre_hook(
                self._make_forward_pre_hook()
            )
    def _make_forward_pre_hook(self):
        def hook(module, *unused):
            for param in module.parameters(recurse=False):
                self.param_to_bucket_group[param].finish_param_sync(
                    skip_next_bucket_dispatch=skip_next_bucket_dispatch
                )
    def zero_grad_buffer(self):
            for param in self.params_with_grad:
                param.grad_added_to_main_grad = False
        for buffer in self.buffers + self.expert_parallel_buffers:
            buffer.reset()
        for bucket_group in self.bucket_groups + self.expert_parallel_bucket_groups:
            bucket_group.reset()
    @contextmanager
    def no_sync(self):
        for bucket_group in self.bucket_groups + self.expert_parallel_bucket_groups:
            bucket_group.is_last_microbatch = False
        try:
            yield
        finally:
            for bucket_group in self.bucket_groups + self.expert_parallel_bucket_groups:
                bucket_group.is_last_microbatch = True
```

### fsdp

mcore_fsdp_adapter.py

```
class FullyShardedDataParallel(_BaseDataParallel):
```

### param_and_grad_buffer.py

```
class _ParamAndGradBucket:
    def __init__(
        self.offset = offset
		#在buffer的起点
        for param in params:
            global_start, global_end, _ = param_index_map[param]
            self.param_to_index[param] = (global_start - offset, global_end - offset)
			#param在bucket的起点和终点
```

通过hook管理通信的启动

```
class _ParamAndGradBucketGroup:
    def register_grad_ready(
        if self.is_last_microbatch:
            self.per_param_grad_ready_counts[param] += 1
            if not self.is_first_batch:
                if self.per_param_grad_ready_counts == self.golden_per_param_grad_ready_counts:
                    self.start_grad_sync(force_all_reduce=force_all_reduce)
    def start_grad_sync(self, force_all_reduce: Optional[bool] = False):
        async_op = (
            self.ddp_config.overlap_grad_reduce
            and self.ddp_config.num_distributed_optimizer_instances == 1
        )
        if self.ddp_config.use_distributed_optimizer:
            communication_group = self.intra_distributed_optimizer_instance_group
            ....
    def finish_param_sync(self, skip_next_bucket_dispatch: bool = False):
...
    def reset(self):
        if self.is_first_batch and len(self.per_param_grad_ready_counts) > 0:
            # Record golden per_param_grad_ready_counts.
            assert len(self.per_param_grad_ready_counts) == len(self.params)
            self.golden_per_param_grad_ready_counts = self.per_param_grad_ready_counts
            self.is_first_batch = False
        self.per_param_grad_ready_counts = {}
        self.is_last_microbatch = True
```

把所有参数展成一维，按元素数量从后往前切分成buckets

创建param和grad两个buffer，替换原来的param.data

```
class _ParamAndGradBuffer:
    def __init__(
        def _pad_end_of_bucket_if_needed(bucket_end_index: int) -> int:
            Pads end index of bucket if using distributed optimizer (to ensure uniform sharding).
        def _pad_start_of_param_if_needed(param_start_index: int) -> int:
            Pads start index of param if using distributed optimizer (to ensure "good" alignment).
        def _update_bucket_metadata(param_end_index: int) -> int:
            per_bucket_numel_unpadded.append(param_end_index - bucket_start_index)
            bucket_end_index = _pad_end_of_bucket_if_needed(param_end_index)
            self.bucket_indices.append((bucket_start_index, bucket_end_index))
            bucket_start_index = bucket_end_index
            bucket_params = set()
            bucket_id += 1
        def _does_param_require_new_bucket(param):
            return (
                getattr(param, "shared_embedding", False)
                and self.ddp_config.use_distributed_optimizer
            )
            #???????
        for param in params[::-1]:
            this_numel = param.data.nelement()
            param_start_index = _pad_start_of_param_if_needed(param_start_index)
            if _does_param_require_new_bucket(param) and len(bucket_params) > 0:
                param_start_index = _update_bucket_metadata(param_start_index)
            self.param_index_map[param] = (param_start_index, param_end_index, bucket_id)
            bucket_params.add(param)
            if (
                bucket_size is not None and (param_end_index - bucket_start_index) >= bucket_size
            ) or _does_param_require_new_bucket(param):
                bucket_end_index = _update_bucket_metadata(param_end_index)
                param_start_index = bucket_end_index
            else:
                param_start_index = param_end_index
        if len(bucket_params) > 0:
            bucket_end_index = _update_bucket_metadata(param_end_index)
        self.numel = bucket_end_index
        self.numel_unpadded = sum(per_bucket_numel_unpadded)
                if self.ddp_config.use_distributed_optimizer:
                    self.param_data = torch.zeros(
                        self.numel,
                        dtype=self.param_dtype,
                        device=torch.cuda.current_device(),
                        requires_grad=False,
                    )
                self.grad_data = torch.zeros(
                    self.numel,
                    dtype=self.grad_dtype,
                    device=torch.cuda.current_device(),
                    requires_grad=False,
                )
        for param in params[::-1]:
                if self.param_data is not None:
                    new_param_data = self._get(
                        param.data.shape, param_start_index, buffer_type=BufferType.PARAM
                    )
                        old_param_data = param.data
                        param.data = new_param_data
                        param.data.detach().copy_(old_param_data)
                        del old_param_data   
            param.main_grad = self._get(
                param.data.shape, param_start_index, buffer_type=BufferType.GRAD
            )
            if bucket_id != cur_bucket_id:
                bucket_end_index = _pad_end_of_bucket_if_needed(param_start_index)
                self.buckets.append(
                    self._new_bucket(
                        bucket_params=bucket_params,
                        start_index=bucket_start_index,
                        end_index=bucket_end_index,
                        numel_unpadded=per_bucket_numel_unpadded[cur_bucket_id],
                        bucket_id=cur_bucket_id,
                    )
                )
                bucket_start_index = bucket_end_index
                bucket_params = []
                cur_bucket_id = bucket_id
            bucket_params.append(param)
        if len(bucket_params) > 0:
            bucket_end_index = _pad_end_of_bucket_if_needed(param_end_index)
            self.buckets.append(
                self._new_bucket(
    def _get(self, shape: torch.Size, start_index: int, buffer_type: BufferType) -> torch.Tensor:
        if buffer_type == BufferType.PARAM:
            buffer_tensor = self.param_data[start_index:end_index]
        elif buffer_type == BufferType.GRAD:
            buffer_tensor = self.grad_data[start_index:end_index]
        buffer_tensor = buffer_tensor.view(shape)
    def _new_bucket(
            bucketed_param_data = self._get(
                torch.Size([end_index - start_index]), start_index, buffer_type=BufferType.PARAM
            )
        bucketed_grad_data = self._get(
            torch.Size([end_index - start_index]), start_index, buffer_type=BufferType.GRAD
        )
        bucket = _ParamAndGradBucket(
            params=bucket_params,
            param_data=bucketed_param_data,
            grad_data=bucketed_grad_data,
            offset=start_index,
        for bucket_param in bucket_params:
            assert bucket_param not in self.param_to_bucket
            self.param_to_bucket[bucket_param] = bucket
    def reset(self):
        """
        Zero out the underlying grad_buffer.
        """
        self.grad_data.zero_()
```

把buffers里的buckets重新组织成bucketgroups

```
def partition_buckets(
    1. If force_single_bucket_group is True, put all buckets across all buffers into a single
       bucket group.
    2. If force_single_bucket_group is False, when there is no fp8 buffer in the input buffers,
       let each bucket group have only one bucket.
    3. If force_single_bucket_group is False, when using fp8 params, merge all non-fp8 buckets
       into the last fp8 bucket group.
    if torch.uint8 not in dtype_to_buffer_map:
        for buffer in buffers:
            for bucket in buffer.buckets:
                bucket_groups.append(
                    _ParamAndGradBucketGroup(
```

### torch_fully_sharded_data_parallel.py

```
class TorchFullyShardedDataParallel(_BaseDataParallel):
```

## extensions

### transformer_engine.py

```
class TELayerNormColumnParallelLinear(te.pytorch.LayerNormLinear):
    def __init__(
```

### transformer_engine_spec_provider.py

```
from megatron.core.extensions.transformer_engine import (
class TESpecProvider(BackendSpecProvider):
    def row_parallel_linear(self) -> type:
        """Which row parallel linear module TE backend uses"""
        return TERowParallelLinear
```

## fusions

fused_bias_dropout.py

```
@jit_fuser
def bias_dropout_add_fused_train(
    x_with_bias: Tuple[torch.Tensor, Optional[torch.Tensor]], residual: torch.Tensor, prob: float
) -> torch.Tensor:
    return _bias_dropout_add_func(x_with_bias, residual, prob, True)
```

fused_bias_swiglu.py

```
from megatron.core.jit import jit_fuser

@jit_fuser
def bias_swiglu(y, bias):
    y = y + bias
    return swiglu(y)

@jit_fuser
def swiglu(y):
    y_1, y_2 = torch.chunk(y, 2, -1)
    return F.silu(y_1) * y_2
```

## jit.py

```
def enable_jit_fuser():
    '''Enable the JIT fuser'''
    global jit_fuser
    try:
        if is_torch_min_version("2.2.0a0"):
            jit_fuser = torch.compile
    except ImportError:

        jit_fuser = noop_decorator
enable_jit_fuser()
```



## model_parallel_config.py

```
class ModelParallelConfig:
```

## models

### common

embeddings/language_model_embedding.py

```
class LanguageModelEmbedding(MegatronModule):
    def __init__(
        self.word_embeddings = tensor_parallel.VocabParallelEmbedding(
```

language_module/language_module.py

```
class LanguageModule(MegatronModule):
```

### gpt

#### gpt_layer_specs.py

mlp的实现

```
def get_mlp_module_spec_for_backend(
    linear_fc2 = backend.row_parallel_linear()
    activation_func = backend.activation_func() if use_te_activation_func else None
    if num_experts is None:
        module = TEFusedMLP if use_te_op_fuser else MLP
            linear_fc1 = backend.column_parallel_layer_norm_linear()
        return ModuleSpec(
            module=module,
            submodules=MLPSubmodules(
                linear_fc1=linear_fc1, linear_fc2=linear_fc2, activation_func=activation_func
            ),
    else:
        # Mixture of experts with modules in megatron core.
        return get_moe_module_spec_for_backend(
```

```
@copy_signature(get_gpt_layer_with_transformer_engine_submodules)
def get_gpt_layer_with_transformer_engine_spec(*args, **kwargs) -> ModuleSpec:
    """Use this spec to use lower-level Transformer Engine modules (required for fp8 training)."""
    return ModuleSpec(
        module=TransformerLayer,
        submodules=get_gpt_layer_with_transformer_engine_submodules(*args, **kwargs),
    )
```

TransformerLayer的实现

```
def get_gpt_layer_with_transformer_engine_submodules(
        backend = TESpecProvider()
    mlp = get_mlp_module_spec_for_backend(
        backend=backend,
        return TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=backend.column_parallel_layer_norm_linear(),
            mlp=mlp,
```

#### gpt_model.py

```
class GPTModel(LanguageModule):
    def __init__(
        if self.pre_process or self.mtp_process:
            self.embedding = LanguageModelEmbedding(
        self.decoder = TransformerBlock(
            config=self.config,
            spec=transformer_layer_spec,
        if self.post_process:
            self.output_layer = tensor_parallel.ColumnParallelLinear(
    def forward(
        hidden_states = self.decoder(

```

## optimizer

### __init__.py

把参数组给basic optimizer

```
    from transformer_engine.pytorch.optimizers import FusedAdam as Adam#####
    from transformer_engine.pytorch.optimizers import FusedSGD as SGD
        from apex.optimizers import FusedAdam as Adam
        from apex.optimizers import FusedSGD as SGD
        from torch.optim import SGD
        from torch.optim import AdamW as Adam
def _get_megatron_optimizer_based_on_param_groups(
            kwargs = {
                "params": param_groups,
                "lr": config.lr,
                "weight_decay": config.weight_decay,
                "betas": (config.adam_beta1, config.adam_beta2),
                "eps": config.adam_eps,
            }
            if USING_PYTORCH_OPTIMIZER:
                adam_cls = torch.optim.AdamW if config.decoupled_weight_decay else torch.optim.Adam
            else:
                kwargs["adam_w_mode"] = config.decoupled_weight_decay
                adam_cls = Adam
            optimizer = adam_cls(**kwargs)
        optimizer_args = [optimizer, config, grad_scaler, init_state_fn]
        if config.use_distributed_optimizer:
            optimizer = DistributedOptimizer(
                *optimizer_args,
                model_chunks=model_chunks,
                per_model_buffers=per_model_buffers,
    return ChainedOptimizer(optimizers)
```

params in model_chunks根据优化器参数分组

```
def _get_param_groups(
    for model_chunk in model_chunks:#vpp
        for name, param in model_chunk.named_parameters():
            param_overrides_list: list[ParamGroupOverride] = []
            if config_overrides is not None:
                for param_key, param_override in config_overrides.items():
                    if param_key.matches(param, name):
                        param_overrides_list.append(param_override)
                        #param需要特殊处理
            if param_overrides_list:
                param_override: ParamGroupOverride | None = combine_param_group_overrides(
                    param_overrides_list
                )
            is_expert_parallel = not getattr(param, 'allreduce', True)
            key = (param_override_tuple, is_expert_parallel)
            if key not in params_map:
                params_map[key] = []
            params_map[key].append(param)
    params_key = list(params_map.keys())
    gathered_params_key = [None for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather_object(gathered_params_key, params_key)
    for keys in gathered_params_key:
        for key in keys:
            if key not in params_key:
                params_key.append(key)
    for key in sorted(params_key, key=lambda x: (x[0] is not None, x[0])):
        param_override_tuple, is_expert_parallel = key
        params = params_map[key] if key in params_map else []
        param_group = {
            'params': params,
            'is_expert_parallel': is_expert_parallel,
            'default_config': uses_default_lr_schedule,
            **default_config,
            **param_override,  # keep **param_override last so that users can override other fields.
        }
        param_groups.append(param_group)
```

```
def _get_param_groups_and_buffers(
    param_groups = _get_param_groups(model_chunks, config, config_overrides)
    param_groups = list(filter(filter_fn, param_groups))
    buffers = {}
    for model_chunk_idx, model_chunk in enumerate(model_chunks):
        if hasattr(model_chunk, buffer_name):
            buffers[model_chunk_idx + model_chunk_offset] = getattr(model_chunk, buffer_name)
{0: [<megatron.core.distributed.param_and_grad_buffer._ParamAndGradBuffer object at 0x1519667cb990>]}
    return param_groups, buffers
```

分专家和dense优化器，dense vpps又分成两个优化器

```
def get_megatron_optimizer(
    if config.overlap_param_gather_with_optimizer_step:
        all_dense_model_chunks = [[model_chunks[0]], model_chunks[1:]]
        overlap_param_gather_with_optimizer_step_flags = [True, False]
    else:
        all_dense_model_chunks = [model_chunks]
        overlap_param_gather_with_optimizer_step_flags = [False]
    if get_pg_size(dp_cp_group) > get_pg_size(intra_dp_cp_group):
        inter_dist_opt_group = process_groups_dict['inter_dist_opt_group']
        distributed_optimizer_instance_id = get_pg_rank(inter_dist_opt_group)
    else:
        distributed_optimizer_instance_id = 0
    for dense_model_chunks, overlap_param_gather_with_optimizer_step in zip(
        all_dense_model_chunks, overlap_param_gather_with_optimizer_step_flags
    ):
        param_groups, buffers = _get_param_groups_and_buffers(
            dense_model_chunks,
            model_chunk_offset=model_chunk_offset,
            config=config,
            config_overrides=config_overrides,
            filter_fn=lambda g: not g['is_expert_parallel'],
            buffer_name='buffers',
        )
        #非专家的opt参数s和非专家buffers
       optimizers.append(
            _get_megatron_optimizer_based_on_param_groups(
                config=config,
                model_chunks=dense_model_chunks,
                param_groups=param_groups,
                per_model_buffers=buffers,
                model_parallel_group=mp_group,
                data_parallel_group=intra_dp_cp_group,
                data_parallel_group_gloo=intra_dp_cp_group_gloo,
                data_parallel_group_idx=model_parallel_rank,
                intra_dist_opt_group=intra_dist_opt_group,
                distributed_optimizer_instance_id=distributed_optimizer_instance_id,
                pg_collection=pg_collection,
            )
        )
        model_chunk_offset += 1
    moe_param_groups, moe_buffers = _get_param_groups_and_buffers(
        model_chunks,
        model_chunk_offset=0,
        config=config,
        config_overrides=config_overrides,
        filter_fn=lambda g: g['is_expert_parallel'],
        buffer_name='expert_parallel_buffers',
    )
        optimizers.append(
            _get_megatron_optimizer_based_on_param_groups(
                config=config,
                model_chunks=model_chunks,
                param_groups=moe_param_groups,
                per_model_buffers=moe_buffers,
                model_parallel_group=expt_tp_pp_group,
                data_parallel_group=intra_expt_dp_group,
                data_parallel_group_gloo=expt_data_parallel_group_gloo,
                data_parallel_group_idx=expt_model_parallel_rank,
                intra_dist_opt_group=intra_dist_opt_group,
                distributed_optimizer_instance_id=distributed_optimizer_instance_id,
```

### distrib_optimizer.py

model_chunks[model,model]

model[buffer,buffer]

buffer[bucket,bucket]

bucket[subbucket,subbucket]

生成shard_main_params替换basic optimizer’s params

```
class DistributedOptimizer(MixedPrecisionOptimizer):
    def __init__(
        self.buffers = list(itertools.chain(*per_model_buffers.values()))
        for model_idx, buffers in self.per_model_buffers.items():
            for _ in buffers:
                self.gbuf_idx_to_model_idx_map[gbuf_idx] = model_idx
                gbuf_idx += 1
                #展出一维
        for model_idx, buffers in self.per_model_buffers.items():
            self.per_model_bucket_groups[model_idx] = partition_buckets(buffers)
        for buffer in self.buffers:
            self.gbuf_ranges.append(self._build_gbuf_range_map(buffer))
        self.model_param_gbuf_map = self._build_model_param_gbuf_map(self.gbuf_ranges)
        #计算自己负责更新的参数（所有models）的位置
        #basic optimizer的所有参数（并非所有models）不是一个rank更新的，有多个basic optimizer
        #forward,backward始终用到所有model参数
        (self.model_param_group_index_map, self.opt_group_ranges) = (
            self._build_optimizer_group_ranges(self.optimizer.param_groups, self.gbuf_ranges)
        )
#{自己负责的param:在原始param_groups的位置}[{"params":自己负责的param,"orig_group":原始param_groups[idx]},]
        ) = self._build_model_and_main_param_groups(
            self.gbuf_ranges, self.model_param_gbuf_map, self.opt_group_ranges, config
        )
#生成shard_main_params替换basic optimizer’s params
            self.optimizer.param_groups = [g["orig_group"] for g in self.opt_group_ranges]
            self.optimizer.load_state_dict(self.optimizer.state_dict())
#bucket被dp切分，记录自己负责的param的位置
{dtyoe:[{"param_map":{param:{}}},]}
    def _build_gbuf_range_map(cls, param_and_grad_buffer: _ParamAndGradBuffer):
        return {
            (param_and_grad_buffer.param_dtype, param_and_grad_buffer.grad_dtype): [
                cls._build_model_gbuf_range(param_and_grad_buffer, bucket_index)
                for bucket_index in range(len(param_and_grad_buffer.buckets))
            ]
        }
#bucket被dp切分，记录自己负责的params的位置
{"param_map":{param:{}}}
    def _build_model_gbuf_range(cls, param_and_grad_buffer: _ParamAndGradBuffer, bucket_index: int):
        data_parallel_world_size = param_and_grad_buffer.data_parallel_group.size()
        gbuf_size = bucket.grad_data.numel()
        max_gbuf_range_size = gbuf_size // data_parallel_world_size
        for r in range(data_parallel_world_size):
            gbuf_world_start = r * max_gbuf_range_size
            gbuf_world_end = min(gbuf_size, gbuf_world_start + max_gbuf_range_size)
            gbuf_world_range = Range(
                gbuf_world_start + bucket.offset, gbuf_world_end + bucket.offset
            )
            gbuf_world_all_ranges.append(gbuf_world_range)
        gbuf_world_range = gbuf_world_all_ranges[data_parallel_rank]
        #每个bucket被dp均分
        param_range_map = cls._build_model_gbuf_param_range_map(
            param_and_grad_buffer.param_index_map, gbuf_world_range, bucket.offset
        )
        data = {"param_map": param_range_map}
buffer[bucket,bucket]
bucket[subbucket,subbucket]
subbucket[param,param]
落在subbucket中的param
world:相较于buffer0点
local:相较于subbucket点
in_bucket:相较于bucket0点
sub_param:相较于param0点，0表示起点在subbucket内
{param:{}}
    def _build_model_gbuf_param_range_map(
        for param, param_world_indexes in param_world_index_map.items():
            param_world_start, param_world_end, _ = param_world_indexes
            param_local_start = max(0, param_world_start - gbuf_world_range.start)
            param_local_end = min(gbuf_world_range.size, param_world_end - gbuf_world_range.start)
            if param_local_end > param_local_start:
                param_local_range = Range(param_local_start, param_local_end)
                param_world_range = param_local_range.normalize(
                    param_local_start + gbuf_world_range.start
                )
                param_world_range_in_bucket = Range(
                    param_world_range.start - bucket_offset, param_world_range.end - bucket_offset
                )
                sub_param_start = max(0, gbuf_world_range.start - param_world_start)
                sub_param_range = param_local_range.normalize(sub_param_start)
                param_range_map[param] = {
                    "gbuf_world": param_world_range,
                    "gbuf_world_in_bucket": param_world_range_in_bucket,
                    "gbuf_local": param_local_range,
                    "param": sub_param_range,
                }
[{dtyoe:[{"param_map":{param:{}}},]},]
计算自己负责的param的位置
    def _build_model_param_gbuf_map(
        for gbuf_index, gbuf_range_map in enumerate(gbuf_ranges):
            for dtype, gbuf_range_map_for_all_buckets in gbuf_range_map.items():
                for bucket_index, gbuf_range_map in enumerate(gbuf_range_map_for_all_buckets):
                    for param, _ in gbuf_range_map["param_map"].items():
                        param_gbuf_map[param] = (gbuf_index, dtype, bucket_index)
    def _build_optimizer_group_ranges(cls, param_groups: List[Dict], gbuf_ranges: List[Dict]):
        for group_index, group in enumerate(param_groups):
            for param in group["params"]:
                world_param_group_map[param] = group_index
        local_param_group_map = {}
        group_ranges = [{"params": []} for _ in param_groups]
        #记录basic optimizer中自己负责的param
        for gbuf_range_map in gbuf_ranges:
            for dtype, gbuf_range_map_for_all_buckets in gbuf_range_map.items():
                for gbuf_range_map in gbuf_range_map_for_all_buckets:
                    for param in gbuf_range_map["param_map"]:
                        group_index = world_param_group_map[param]
                        group_range = group_ranges[group_index]
                        group_range["params"].append(param)
                        local_param_group_map[param] = (group_index, len(group_range["params"]) - 1)
        for group_index, group_range in enumerate(group_ranges):
            group_range["orig_group"] = param_groups[group_index]
            group_range["orig_group_idx"] = param_groups[group_index]
        return local_param_group_map, group_ranges
gbuf_ranges[{dtyoe:[{"param_map":{param:{"param": sub_param_range}}},]},]
opt_group_ranges[{"params":自己负责的param,"orig_group":原始param_groups[idx]}]
生成shard_main_params替换basic optimizer’s params
    def _build_model_and_main_param_groups(
        for group_range in opt_group_ranges:
            for model_param in group_range["params"]:
                gbuf_index, dtype, bucket_index = param_gbuf_map[model_param]
                gbuf_range = gbuf_ranges[gbuf_index][dtype][bucket_index]
                #自己负责的param的范围
                param_range = gbuf_range["param_map"][model_param]["param"]
                        shard_model_param = model_param.detach().view(-1)[
                            param_range.start : param_range.end
                        ]
                            shard_main_param = shard_model_param.clone().float()
                    model_param.main_param = shard_main_param
                    model_param.main_param_sharded = True
                    model_float16_params_this_group.append(model_param)
                    shard_float16_params_this_group.append(shard_model_param)
                    shard_fp32_from_float16_params_this_group.append(shard_main_param)
            if not config.use_precision_aware_optimizer_no_fp8_or_ds_fp8:
                group_range["orig_group"]["params"] = [
                    *shard_fp32_params_this_group,
                    *shard_fp32_from_float16_params_this_group,
                ]
```

### optimizer.py

```
class ChainedOptimizer(MegatronOptimizer):
```

```
class MegatronOptimizer(ABC):
    def __init__(
        self.optimizer = optimizer
```

```
class MixedPrecisionOptimizer(MegatronOptimizer):
    def __init__(
```

## parallel_state.py

5种并行方式，attn和expert可以不同的并行方式，cp和ep互斥

opt的范围可以比dp*cp少

```
def create_group(
    ranks=None,
    group = torch.distributed.new_group(**kwargs)
    global _global_process_group_list
    if _global_process_group_list is None:
        # None stands for the default process group
        _global_process_group_list = [None]
    if torch.distributed.get_rank() in ranks:
        _global_process_group_list.append(group)
```

```
def default_position_embedding_ranks(pp_ranks):
    """Return the default ranks that constitute the stages on which the position embeddings live.
    For most models, this is only the first pipeline stage."""
    return [pp_ranks[0]]
```

```
def initialize_model_parallel(
    if get_position_embedding_ranks is None:
        get_position_embedding_ranks = default_position_embedding_ranks
    model_size = tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size
    data_parallel_size: int = world_size // model_size
    decoder_rank_generator = RankGenerator(
    if expert_tensor_parallel_size is None:
        expert_tensor_parallel_size = tensor_model_parallel_size
    expert_tensor_model_pipeline_parallel_size = (
        expert_tensor_parallel_size * expert_model_parallel_size * pipeline_model_parallel_size
    )
    expert_data_parallel_size = world_size // expert_tensor_model_pipeline_parallel_size
    expert_decoder_rank_generator = RankGenerator(
        tp=expert_tensor_parallel_size,
        ep=expert_model_parallel_size,
        dp=expert_data_parallel_size,
        pp=pipeline_model_parallel_size,
        cp=1,
        order=order,
        rank_offset=0,
    )
    intra_partial_data_parallel_size = (
        data_parallel_size * context_parallel_size
    ) // num_distributed_optimizer_instances
    for ranks_with_cp in decoder_rank_generator.get_ranks('dp-cp'):
        group_with_cp = create_group(
            ranks_with_cp,
            timeout=timeout,
            pg_options=get_nccl_options("dp_cp", nccl_comm_cfgs),
            group_desc="DATA_PARALLEL_GROUP_WITH_CP",
        )
        if rank in ranks_with_cp:
            _DATA_PARALLEL_GROUP_WITH_CP = group_with_cp
        if num_distributed_optimizer_instances > 1:
            # Create groups for intra-partial DP domain
            for i in range(num_distributed_optimizer_instances):
                intra_partial_dp_ranks_with_cp = ranks_with_cp[
                    (i * intra_partial_data_parallel_size) : (
                        (i + 1) * intra_partial_data_parallel_size
                    )
                ]
                intra_partial_dp_group_with_cp = create_group(
                    intra_partial_dp_ranks_with_cp,
                if rank in intra_partial_dp_ranks_with_cp:
                    _INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP = intra_partial_dp_group_with_cp
        else:
            _INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP = _DATA_PARALLEL_GROUP_WITH_CP
    if hybrid_context_parallel:
        global _HYBRID_DP_CP_GROUPS
    for ranks in decoder_rank_generator.get_ranks('dp'):
    for ranks in decoder_rank_generator.get_ranks('cp'):
        if hierarchical_context_parallel_sizes:
            assert np.prod(hierarchical_context_parallel_sizes) == context_parallel_size
    for ranks in decoder_rank_generator.get_ranks('tp-pp'):
    for ranks in decoder_rank_generator.get_ranks('tp'):
    for ranks in decoder_rank_generator.get_ranks('pp'):
        embedding_ranks = get_embedding_ranks(ranks)#pp中的embeding
        group = create_group(
        position_embedding_ranks = get_position_embedding_ranks(ranks)
        group = create_group(
    for ranks in decoder_rank_generator.get_ranks('tp-dp-cp'):
    for ranks in decoder_rank_generator.get_ranks('tp-dp'):
    for ranks in decoder_rank_generator.get_ranks('tp-cp'):
    for ranks in expert_decoder_rank_generator.get_ranks('ep'):
    for ranks in expert_decoder_rank_generator.get_ranks('tp'):
    for ranks in expert_decoder_rank_generator.get_ranks('tp-ep'):
    for ranks in expert_decoder_rank_generator.get_ranks('tp-ep-pp'):
    intra_partial_expert_data_parallel_size = (
        expert_data_parallel_size // num_distributed_optimizer_instances
    )
    for ranks in expert_decoder_rank_generator.get_ranks('dp'):
    for ranks in expert_decoder_rank_generator.get_ranks('tp-ep-pp'):
        model_parallel_group_id += 1
        intra_dist_opt_ranks.extend(ranks)
        if model_parallel_group_id % intra_partial_expert_data_parallel_size == 0:
            intra_dist_opt_instance_group = create_group(
                intra_dist_opt_ranks,
                timeout=timeout,
                pg_options=get_nccl_options("intra_dist_opt_instance", nccl_comm_cfgs),
                group_desc="INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP",
            )
            if rank in intra_dist_opt_ranks:
                _INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP = intra_dist_opt_instance_group
            intra_dist_opt_ranks = []
```

```
def generate_masked_orthogonal_rank_groups(
    masked_shape = [s for s, m in zip(parallel_size, mask) if m]
    unmasked_shape = [s for s, m in zip(parallel_size, mask) if not m]
    def prefix_product(a: List[int], init=1) -> List[int]:
        r = [init]
        for v in a:
            init = init * v
            r.append(init)
        return r
    global_stride = prefix_product(parallel_size)
    masked_stride = [d for d, m in zip(global_stride, mask) if m]
    unmasked_stride = [d for d, m in zip(global_stride, mask) if not m]
    group_size = prefix_product(masked_shape)[-1]
    num_of_group = world_size // group_size
    def inner_product(a: List[int], b: List[int]) -> int:
        return sum([x * y for x, y in zip(a, b)])
    def decompose(index, shape, stride=None):
    #index = sum(idx[i] * stride[i])
        if stride is None:
            stride = prefix_product(shape)
        idx = [(index // d) % s for s, d in zip(shape, stride)]
        return idx
    for group_index in range(num_of_group):
        decomposed_group_idx = decompose(group_index, unmasked_shape)
        rank = []
        #unmasked_idx相同
        for rank_in_group in range(group_size):
        	#masked_idx不同
            decomposed_rank_idx = decompose(rank_in_group, masked_shape)
            rank.append(
                inner_product(decomposed_rank_idx, masked_stride)
                + inner_product(decomposed_group_idx, unmasked_stride)
            )
            #global_rank
        ranks.append(rank)
```

```
class RankGenerator(object):
    def __init__(
        assert (
            ep == 1 or cp == 1
    def get_mask(self, order: str, token: str):
	#'tp-cp-ep-dp-pp'
        ordered_token = order.split("-")
        token_list = token.split("-")
        mask = [False] * len(ordered_token)
        for t in token_list:
            mask[ordered_token.index(t)] = True
    def get_ranks(self, token):
        mask = self.get_mask(self.order, token)
        ranks = generate_masked_orthogonal_rank_groups(self.world_size, self.ordered_size, mask)
		#分组，组内unmasked_idx相同
```

## pipeline_parallel

### schedules.py

micro_batch=global_batch/dp

```
def forward_backward_no_pipelining(
    config = get_model_config(model)
    no_sync_func = config.no_sync_func
        with no_sync_func():
            for i in range(num_microbatches - 1):
                output_tensor, num_tokens = forward_step(
                    forward_step_func,
        output_tensor, num_tokens = forward_step(
```

```
def forward_step(
    set_input_tensor = get_attr_wrapped_model(model, "set_input_tensor")
    set_input_tensor(input_tensor)
            output_tensor, loss_func = forward_step_func(data_iterator, model)
```

三种调度

```
def get_forward_backward_func(pp_size: Optional[int] = None, vp_size: Optional[int] = None):
        pp_size (Optional[int]): Pipeline model parallel size to use.
        vp_size (Optional[int]): Virtual pipeline model parallel size to use.
    if pp_size > 1:
        if vp_size is not None:
            forward_backward_func = forward_backward_pipelining_with_interleaving
        else:
            forward_backward_func = forward_backward_pipelining_without_interleaving
    else:
        forward_backward_func = forward_backward_no_pipelining
```

process_groups_config.py

```
@dataclass
class ProcessGroupCollection:
    @staticmethod
    def setup_process_groups_for_ddp(
        if pg_collection is None:
            # Use parallel_state groups
            return {
                'dp_group': parallel_state.get_data_parallel_group(
                    with_context_parallel=False, partial_data_parallel=False
                ),
```

## tensor_parallel

layers.py

```
class VocabParallelEmbedding(torch.nn.Module):
    def __init__(
            self.weight = Parameter(
                torch.empty(
```

## transformer

### mlp.py

```
class MLP(MegatronModule):
    def __init__(
        self.linear_fc1 = submodules.linear_fc1(

```



### module.py

```
class MegatronModule(torch.nn.Module):
```

```
class GraphableMegatronModule(MegatronModule):
    """Megatron module that can be used to capture and replay CUDA graphs.
    def __init__(self, config: TransformerConfig, vp_stage: Optional[int] = None):
        if (
            config.cuda_graph_impl == "local"
```

```
class Float16Module(MegatronModule):
    def __init__(self, config: TransformerConfig, module: torch.nn.Module):
        elif self.bf16:
            self.add_module('module', module.bfloat16())

            def float16_convertor(val):
                return val.bfloat16()
        self.float16_convertor = float16_convertor
    def forward(self, *inputs, fp32_output=True, **kwargs):
        if is_vp_first_stage(self.vp_stage, self.vp_size) and is_pp_first_stage(pp_group):
            inputs = fp32_to_float16(inputs, self.float16_convertor)
        outputs = self.module(*inputs, **kwargs)
        if (
            is_vp_last_stage(self.vp_stage, self.vp_size)
            and is_pp_last_stage(pp_group)
            and fp32_output is True
        ):
            outputs = float16_to_fp32(outputs)
```

### spec_utils.py

```
@dataclass
class ModuleSpec:
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return build_module(self, *args, **kwargs)
def build_module(spec_or_module: Union[ModuleSpec, type], *args, **kwargs):
    elif hasattr(spec_or_module, "module") and isinstance(spec_or_module.module, type):
        module = spec_or_module.module
        return module(
            *args, **spec_or_module.params if hasattr(spec_or_module, "params") else {}, **kwargs
        )
```

### transformer_block.py

block=layers

```
def _get_block_submodules(
    elif isinstance(spec, ModuleSpec):
        if issubclass(spec.module, TransformerBlock):
            return spec.submodules
        elif issubclass(spec.module, BaseTransformerLayer):
            num_layers = get_num_layers_to_build(config, vp_stage, pp_rank)
            return TransformerBlockSubmodules(
                layer_specs=[spec] * num_layers, layer_norm=LayerNormImpl
            )
```

```
class TransformerBlock(GraphableMegatronModule, MegatronModule):
    def __init__(
        self.submodules = _get_block_submodules(config, spec, vp_stage, pp_rank)
        self._build_layers()
    def _build_layers(self):
        def build_layer(layer_spec, layer_number):
                module = build_module(
                    layer_spec,
                    config=layer_config,
                    layer_number=layer_number,
        self.layers = torch.nn.ModuleList(
            [
                build_layer(layer_spec, i + 1)
                for i, layer_spec in enumerate(self.submodules.layer_specs)
```

### transformer_layer.py

```
class TransformerLayer(GraphableMegatronModule, BaseTransformerLayer):
    def __init__(
        self.self_attention = build_module(
            submodules.self_attention,
        self.mlp = build_module(submodules.mlp, config=self.config, **additional_mlp_kwargs)
```

## tokenizers

utils/build_tokenizer.py

```
def build_tokenizer(args, **kwargs):
    tokenizer = MegatronTokenizer.from_pretrained(
        tokenizer_path=tokenizer_path, metadata_path=metadata, **kwargs
    )
```

## utils.py

```
def unwrap_model(model, module_instances=None):
        module_instances = (DDP, torch_FSDP, megatron_FSDP, Float16Module)

    return_list = True
    if not isinstance(model, list):
        model = [model]
        return_list = False
    unwrapped_model = []
    for model_module in model:
        while isinstance(model_module, module_instances):
            model_module = model_module.module
        unwrapped_model.append(model_module)
```

# gpt_builders.py

tp实现在网络层

```
def _get_transformer_layer_spec(use_te, config):
    if use_te:
        return get_gpt_layer_with_transformer_engine_spec(
```

```
def gpt_builder(args, pre_process, post_process, vp_stage=None, config=None, pg_collection=None):
    if config is None:
        if args.yaml_cfg is not None:
            config = core_transformer_config_from_yaml(args, "language_model")
        else:
            config = core_transformer_config_from_args(args)
                transformer_layer_spec = _get_transformer_layer_spec(use_te, config)
        model = GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
```

# model_provider.py

```
def model_provider(
    return model_builder(args, pre_process, post_process, vp_stage, config=config, pg_collection=pg_collection)
```

# training

argument_utils.py

```
class ArgumentGroupFactory:
    def __init__(self, src_cfg_class: type, exclude: Optional[list[str]] = None) -> None:
        self.src_cfg_class = src_cfg_class
        self.field_docstrings = self._get_field_docstrings(src_cfg_class)
    def build_group(self, parser: ArgumentParser, title: Optional[str] = None) -> _ArgumentGroup:
        arg_group = parser.add_argument_group(title=title, description=self.src_cfg_class.__doc__)
        for attr in fields(self.src_cfg_class):
            add_arg_kwargs = self._build_argparse_kwargs_from_field(attr)
            arg_group.add_argument(*arg_names, **add_arg_kwargs)
    def _build_argparse_kwargs_from_field(self, attribute: Field) -> dict[str, Any]:
        argparse_kwargs["arg_names"] = [self._format_arg_name(attribute.name)]
        argparse_kwargs["dest"] = attribute.name
    def _format_arg_name(self, config_attr_name: str, prefix: Optional[str] = None) -> str:
        arg_name = "--" + arg_name.replace("_", "-")
```

## arguments.py

```
def _add_distributed_args(parser):
    dist_init_factory = ArgumentGroupFactory(DistributedInitConfig)
```

```
def _add_training_args(parser):
    train_factory = ArgumentGroupFactory(TrainingConfig)
    group = train_factory.build_group(parser, "training")
    #把dataclass的fields转成--xxx
```

```
def add_megatron_arguments(parser: argparse.ArgumentParser):
    parser = _add_training_args(parser)
    parser = _add_distributed_args(parser)
```

```
def parse_args(extra_args_provider=None, ignore_unknown_args=False):
    parser = argparse.ArgumentParser(description='Megatron-LM Arguments',
    parser = add_megatron_arguments(parser)
    if ignore_unknown_args:
        args, _ = parser.parse_known_args()
    args.rank = int(os.getenv('RANK', '0'))
    args.world_size = int(os.getenv("WORLD_SIZE", '1'))
```

```
def validate_args(args, defaults={}):
    total_model_size = args.tensor_model_parallel_size * args.pipeline_model_parallel_size * args.context_parallel_size
    args.data_parallel_size = args.world_size // total_model_size
    if args.hierarchical_context_parallel_sizes:
        from numpy import prod
        assert args.context_parallel_size == prod(args.hierarchical_context_parallel_sizes)
        if args.enable_cuda_graph:
            print_rank_0(
                '--enable-cuda-graph is deprecated, use --cuda-graph-impl=local instead.', args.rank
            )
            args.cuda_graph_impl = "local"
            del args.enable_cuda_graph
        if args.external_cuda_graph:
            print_rank_0(
                '--external-cuda-graph is deprecated, use --cuda-graph-impl=transformer_engine instead.',
                args.rank,
            )
            args.cuda_graph_impl = "transformer_engine"
            del args.external_cuda_graph
        args.global_batch_size = args.micro_batch_size * args.data_parallel_size
            args.virtual_pipeline_model_parallel_size = args.num_virtual_stages_per_pipeline_rank
        args.kv_channels = args.hidden_size // args.num_attention_heads
    _print_args("arguments", args)
```

## checkpointing.py

```
def _load_base_checkpoint(
    non_persistent_iteration = _get_non_persistent_iteration(
        tracker_filename = get_checkpoint_tracker_filename(load_dir)
        if isfile(tracker_filename):
            iteration, release = read_metadata(tracker_filename)
    set_loaded_iteration(iteration)
    checkpoint_name = get_checkpoint_name(load_dir, iteration, release, return_base_dir=True)
    ckpt_format = _get_checkpoint_format(checkpoint_name, args)
    if ckpt_format == "torch_dist":
        return _load_global_dist_base_checkpoint(
```

```
def _load_global_dist_base_checkpoint(
        checkpoint_name = find_checkpoint_rank_0(load_dir, iteration, release)
        state_dict = dist_checkpointing.load_common_state_dict(checkpoint_name)
```

```
def _get_checkpoint_format(checkpoint_name, args):
        checkpoint_dir = msc.Path(checkpoint_name)
        is_torch_ckpt = any([f.name.startswith("mp_rank_0") for f in checkpoint_dir.iterdir()])
        is_torch_dcp = checkpoint_dir.joinpath(".metadata").exists()
    if dist_checkpointing.check_is_distributed_checkpoint(checkpoint_name):
        ckpt_format = "torch_dist"
```

```
def get_checkpoint_name(checkpoints_path, iteration, release=False,
        directory = 'iter_{:07d}'.format(iteration)
    if return_base_dir:
        common_path = os.path.join(checkpoints_path, directory)
```

```
def get_checkpoint_tracker_filename(checkpoints_path):
    return os.path.join(checkpoints_path, 'latest_checkpointed_iteration.txt')
```

```
def load_checkpoint(ddp_model, optimizer, opt_param_scheduler, load_arg='load', strict=True,
    load_dir = getattr(args, load_arg)
    model = unwrap_model(ddp_model)
        state_dict, checkpoint_name, release, ckpt_type = _load_base_checkpoint(
            load_dir,
        elif ckpt_type in [CheckpointType.LOCAL, CheckpointType.GLOBAL]:
            ckpt_format = "torch_dist"
            ckpt_args = state_dict.get("args")
            sharded_sd_metadata = dist_checkpointing.load_content_metadata(preloaded_state_dict=state_dict)
            gen_sd_optim = optimizer
        optim_sd_kwargs = dict(metadata=sharded_sd_metadata, is_loading=True)
        model_sd_kwargs = dict(metadata=sharded_sd_metadata)
            load_kwargs['sharded_state_dict'] = generate_state_dict(
                args, model, gen_sd_optim, gen_sd_opt_param_scheduler, gen_sd_rng_state,
                optim_sd_kwargs=optim_sd_kwargs, model_sd_kwargs=model_sd_kwargs,
    state_dict, checkpoint_name, release, ckpt_type = _load_base_checkpoint(
            load_model_state_dict(ddp_model[0], state_dict['model'], strict)
                optimizer.load_state_dict(state_dict['optimizer'])
```

## common_config.py

```
class DistributedInitConfig:
    local_rank: int = field(default_factory=lambda: int(os.getenv("LOCAL_RANK", "0")))
```

## initialize.py

```
def _initialize_distributed(get_embedding_ranks, get_position_embedding_ranks, store):
            torch.cuda.set_device(args.local_rank)
            device_id = torch.device(f'cuda:{args.local_rank}')
        if args.cuda_graph_impl == "transformer_engine":
            torch.cuda.set_stream(torch.cuda.Stream())
        torch.distributed.init_process_group(**init_process_group_kwargs)
            mpu.initialize_model_parallel(
```

```
def _warmup_jit_function():
            if args.swiglu:
                output = bias_swiglu(input, bias)
            else:
                output = bias_gelu(bias, input)
            output = bias_dropout_add_fused_train([input, bias], residual, dropout_rate)
```

```
def initialize_megatron(
        args = parse_args(extra_args_provider, ignore_unknown_args)
        validate_args(args, args_defaults)
    set_global_variables(args)
    args = get_args()
    def finish_mpu_init():
        _initialize_distributed(get_embedding_ranks, get_position_embedding_ranks, store)
    else:
        # Megatron's MPU is the master. Complete initialization right away.
        finish_mpu_init()
```

```
def set_jit_fusion_options():
    _warmup_jit_function()
```



## inprocess_restart.py

```
def maybe_wrap_for_inprocess_restart(pretrain):
    args = arguments.parse_args(ignore_unknown_args=True)
```

## global_vars.py

```
def _build_tokenizer(args):
    _GLOBAL_TOKENIZER = build_tokenizer(args)
def get_args():
	return _GLOBAL_ARGS
def set_args(args):
    global _GLOBAL_ARGS
    _GLOBAL_ARGS = args
def set_global_variables(args, build_tokenizer=True):
    set_args(args)
    if build_tokenizer:
        _ = _build_tokenizer(args)
```

## training.py

```
def build_train_valid_test_datasets(build_train_valid_test_datasets_provider, train_valid_test_num_samples=None):
        train_valid_test_num_samples = get_train_valid_test_num_samples()
```

```
def build_train_valid_test_data_iterators(build_train_valid_test_datasets_provider):
    train_dataloader, valid_dataloaders, test_dataloader = build_train_valid_test_data_loaders(
        build_train_valid_test_datasets_provider
    )
```

```
def build_train_valid_test_data_loaders(build_train_valid_test_datasets_provider):
    is_distributed = getattr(build_train_valid_test_datasets_provider, "is_distributed", False)
            train_ds, valid_ds, test_ds = build_train_valid_test_datasets(build_train_valid_test_datasets_provider)
```

以vpp stage为单元创建模型，cuda()，Float16Module()，DDP()

```
def get_model(model_provider_func, model_type=ModelType.encoder_or_decoder, wrap_with_ddp=True, config=None, pg_collection=None):
    def build_model():
        if (
            get_pg_size(pg_collection.pp) > 1
            and args.virtual_pipeline_model_parallel_size is not None
        ):
            vp_size = args.virtual_pipeline_model_parallel_size
            for i in range(vp_size):
                pre_process = is_pp_first_stage(pg_collection.pp) and is_vp_first_stage(
                    vp_stage=i, vp_size=vp_size
                )
                post_process = is_pp_last_stage(pg_collection.pp) and is_vp_last_stage(
                    vp_stage=i, vp_size=vp_size
                )
                this_model = model_provider_func(
                    pre_process=pre_process,
                    post_process=post_process,
                    vp_stage=i,
                this_model.vp_stage = i
                model.append(this_model)
        else:
            pre_process = is_pp_first_stage(pg_collection.pp)
            post_process = is_pp_last_stage(pg_collection.pp)
            model = model_provider_func(
        model = build_model()
    if not isinstance(model, list):
        model = [model]
    for model_module in model:
        for param in model_module.parameters():
            tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)
        for model_module in model:
            model_module.cuda(torch.cuda.current_device())
        model = [Float16Module(config, model_module) for model_module in model]
    if wrap_with_ddp:
        if args.use_torch_fsdp2:
            assert HAVE_FSDP2, "Torch FSDP2 requires torch>=2.4.0"
            DP = torch_FSDP
        elif args.use_megatron_fsdp:
            DP = megatron_FSDP
        else:
            DP = DDP
        with torch.cuda.stream(ddp_stream):
            model = [
                DP(
                )
                for (model_chunk_idx, model_chunk) in enumerate(model)
            ]
```

```
def get_train_valid_test_num_samples():
    args = get_args()
        train_samples = args.train_samples
            eval_iters = (args.train_iters // args.eval_interval + 1) * args.eval_iters
        eval_samples = eval_iters * args.global_batch_size
    test_samples = args.eval_iters * args.global_batch_size
        train_samples_in_current_phase = train_samples
    return (train_samples_in_current_phase, eval_samples, test_samples)
```

计算进程组，use torch.compile to fuse function，model，optimizer，data，train

```
def pretrain(
        model_provider: a function that returns a vanilla version of the
            model. By vanilla we mean a simple model on cpu with no fp16 or ddp.
    initialize_megatron(
    set_jit_fusion_options()
    model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
        model_provider, model_type, checkpointing_context=checkpointing_context
    )
        train_data_iterator, valid_data_iterator, test_data_iterator = (
            build_train_valid_test_data_iterators(train_valid_test_dataset_provider)
        )
            iteration, num_floating_point_operations_so_far = train(
                forward_step_func,
                model,
                optimizer,
```

模型组网，optimizer划分负责的参数，加载checkpoint

```
def setup_model_and_optimizer(
    wrap_with_ddp = not args.skip_train
    model = get_model(model_provider_func, model_type, wrap_with_ddp=wrap_with_ddp)
    unwrapped_model = unwrap_model(model)
            optimizer = get_megatron_optimizer(
                config,
                model,
        args.iteration, args.num_floating_point_operations_so_far = load_checkpoint(
```

global_batch

```
from megatron.core.pipeline_parallel import get_forward_backward_func
def train(
    forward_step_func,
    for model_module in model:
        model_module.train()
	forward_backward_func = get_forward_backward_func()
	#调度函数
    if args.cuda_graph_impl == "local" and CudaGraphScope.full_iteration in args.cuda_graph_scope:
        forward_backward_func = FullCudaGraphWrapper(forward_backward_func, cuda_graph_warmup_steps=args.cuda_graph_warmup_steps)
    start_iteration = iteration
    if should_disable_forward_pre_hook(args):
        disable_forward_pre_hook(model, param_sync=False)
    if args.cuda_graph_impl == "transformer_engine":
        cuda_graph_helper = TECudaGraphHelper(
    while iteration < args.train_iters:
        num_microbatches = get_num_microbatches()
        update_num_microbatches(args.consumed_train_samples, consistency_check=True, verbose=True)
        args.curr_iteration = iteration
        ) = train_step(
            forward_step_func, train_data_iterator, model, optimizer, opt_param_scheduler, config, forward_backward_func, iteration=iteration
        )
```

zero_grad，schedule

```
def train_step(forward_step_func, data_iterator, model, optimizer, opt_param_scheduler, config, forward_backward_func, iteration=None):
    while rerun_state_machine.should_run_forward_backward(data_iterator):
        for model_chunk in model:
            model_chunk.zero_grad_buffer()
        optimizer.zero_grad()
        losses_reduced = forward_backward_func(
            forward_step_func=forward_step_func,
```

## training_config.py

```
@dataclass(kw_only=True)
class TrainingConfig:
    """Configuration settings related to the training loop."""

    train_samples: int | None = None
    """Total number of samples to train over all training runs.
    Note that either train_iters or train_samples should be provided."""
```

