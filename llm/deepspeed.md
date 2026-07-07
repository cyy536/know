```
docker run -itd --ipc host --net host --gpus all --name cyy-ds -v ~/ds:/workspace pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel bash
pip install deepspeed
```

If no hostfile is specified or found, DeepSpeed queries the number of GPUs on the local machine to discover the number of local slots available.

```
deepspeed --hostfile=myhostfile <client_entry.py> <client args> \
  --deepspeed --deepspeed_config ds_config.json
deepspeed --include localhost:0,1 train.py
```

```python
# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import deepspeed
from torch.utils.data import DataLoader, TensorDataset
from deepspeed import comm as dist

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

inputs = torch.randn(24, 10)
labels = torch.randint(0, 2, (24,))

dataset = TensorDataset(inputs, labels)
dataloader = DataLoader(dataset, batch_size=8)

model = SimpleModel()

ds_config = {
    "train_batch_size": 8,
    "gradient_accumulation_steps": 1,
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 0.001
        }
    },
    "fp16": {
        "enabled": True,
        "auto_cast": True
    },
    "zero_optimization": {
        "stage": 2,  
    }
}

model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config=ds_config,
)

for epoch in range(1):
    for batch in dataloader:
        inputs, labels = batch
        inputs = inputs.to(model_engine.local_rank)
        labels = labels.to(model_engine.local_rank)

        outputs = model_engine(inputs)
        loss = nn.CrossEntropyLoss()(outputs, labels)

        # if dist.get_rank()==0:
        #     import pdb
        #     pdb.set_trace()

        model_engine.backward(loss)
        
        model_engine.step()

        print(f"Epoch {epoch}, Loss: {loss.item()}")

```

# Optimizer

optimizer.param_groups==list[dict[str, Any]]

不同参数组可以有不同配置字典

```
class Optimizer:
    def __init__(self, params: ParamsT, defaults: dict[str, Any]) -> None:  # noqa: D107
		self.param_groups: list[dict[str, Any]] = []
		if not isinstance(param_groups[0], dict):
            param_groups = [{"params": param_groups}]
        for param_group in param_groups:
            self.add_param_group(cast(dict, param_group))
            	self.param_groups.append(param_group)
            	
class FusedAdam(torch.optim.Optimizer):
    """Implements Adam algorithm.

    Currently GPU-only.  Requires Apex to be installed via
    ``pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./``.

    This version of fused Adam implements 2 fusions.

      * Fusion of the Adam update's elementwise operations
      * A multi-tensor apply launch that batches the elementwise updates applied to all the model's parameters into one or a few kernel launches.
```

# ZeroOptimizer.init

zero1,2是一样的

完整的16位参数，dp分之一的32位参数，都以存储于GPU一维数组中

```
self.bit16_groups[group_id]
[所有参数]

self.round_robin_bit16_groups[group_id]
(连续的参数分配在不同rank，让相同rank的参数放在一起，加速卸载)

self.bit16_groups_flat[group_id]
展成一维（要2*dp对齐），移到GPU，模型参数=buffer+偏移

self.parallel_partitioned_bit16_groups[group_id]
按dp平分一维Tensor: [Tensor]

self.groups_padding[group_id]
所在dp padding的数量，padding只在一维Tensor尾部，应该只有最后一个rank有padding

self.single_partition_of_fp32_groups[group_id]
创建所在dp的32位参数: Tensor，设为优化器参数

self.partition_size[group_id]
负责的元素数量

self.params_in_partition[group_id]
负责的参数，只有部分落在区间里也算：[Tensor]
self.params_not_in_partition[group_id]

self.first_offset[group_id]
第一个部分tensor起点到本地的距离

self.param_id[id(param)] = count++
self.param_dict[count] = param
self.is_param_in_current_partition[self.get_param_id(param)] = True

self.param_to_partition_ids[group_id][param_id]
param所在partitons:[partition_id]

self.total_grads_in_partition[group_id][partition_id]++
self.remaining_grads_in_partition[group_id][partition_id] = self.total_grads_in_partition[group_id][partition_id]

self.grad_partition_insertion_offset[group_id][partition_id][param_id]
在本地的起点

self.grad_start_offset[group_id][partition_id][param_id]			
起点到本地的距离

self.first_param_index_in_partition[group_id][partition_id]
分区的第一个参数

注册grad_hook

_hp_mapping暂时没有看到使用
hp_frag_address
#在32位分区里的起点>=0，在分区里的元素数量<=tensor.numel()
lp_frag_address
#在参数里的起点
建立16位参数到32位参数的连接
这里有bug，16位参数没有round_robin

self._param_slice_mappings[group_id][lp_name]=lp.hp_frag_address
```

```
class DeepSpeedZeroOptimizer(ZeROOptimizer):
    def __init__(self,
    	self.optimizer = init_optimizer
        self.real_dp_process_group = [dp_process_group for i in range(len(self.optimizer.param_groups))]
        self.partition_count = [dp_size for i in range(len(self.optimizer.param_groups))]
        self.dtype = self.optimizer.param_groups[0]['params'][0].dtype
        if self.dtype != self.gradient_accumulation_dtype:
            self.use_separate_grad_accum = True
        else:
            self.use_separate_grad_accum = False#
        if self.use_separate_grad_accum and not self.partition_gradients:#zero1
            self.use_grad_accum_attribute = True
        else:
            self.use_grad_accum_attribute = False#
        if is_autocast_initialized():
        else:#
            comm_dtypes = {self.communication_data_type}
            self.torch_autocast_gradscaler = None
        self.ipg_buckets: Dict[torch.dtype, IPGBucket] = 
        	{dtype: IPGBucket() for dtype in comm_dtypes}
        for i, param_group in enumerate(self.optimizer.param_groups):
        	partition_id = dist.get_rank(group=self.real_dp_process_group[i])
			trainable_parameters = []
            for param in param_group['params']:
                if param.requires_grad:
                    param.grad_accum = None
                    param.param_idx_in_group = len(trainable_parameters)
                    trainable_parameters.append(param)
            self.bit16_groups.append(trainable_parameters)
            #[[Tensor]]
            orig_group_numel = 0
            for param in self.bit16_groups[i]:
                orig_group_numel += param.numel()
                param.cpu_data = param.data.cpu()
                param.data = torch.empty(1).to(param.device)
            #转成cpu
            round_robin_tensors = self.bit16_groups[i]
            round_robin_indices = list(range(len(self.bit16_groups[i])))
            #连续的优化器状态可以分配在不同rank，让相同rank的放在一起
            #{original_index:new_index}
            self.round_robin_bit16_groups.append(round_robin_tensors)
            self.round_robin_bit16_indices.append(round_robin_indices)
            meta_tensors = []
            for param in round_robin_tensors:
                meta_tensors.append(torch.zeros_like(param.cpu_data, device="meta"))
            self.round_robin_bit16_meta.append(meta_tensors)
            flattened_buffer = self.flatten_dense_tensors_aligned(
                self.round_robin_bit16_groups[i],
                self.nccl_start_alignment_factor * dist.get_world_size(group=self.real_dp_process_group[i]),
            use_cpu_data=True)				
                tensor_list = [param.cpu_data for param in tensor_list] if use_cpu_data else tensor_list
        		return self.flatten(align_dense_tensors(tensor_list, alignment))
        			padded_tensor_list = tensor_list + [pad_tensor]
        			return torch._C._nn.flatten_dense_tensors(tensors)
            #创建CPU一维副本，要2*dp对齐
            for param in self.bit16_groups[i]:
                del param.cpu_data
            #删除 
            self.bit16_groups_flat.append(
            	flattened_buffer.to(get_accelerator().current_device_name()))
            #移到GPU
            self._update_model_bit16_weights(i)
            	updated_params = self.unflatten(self.bit16_groups_flat[group_index], self.round_robin_bit16_meta[group_index])
            	for p, q in zip(self.round_robin_bit16_groups[group_index], updated_params):
          			p.data = q.data
                for param_index, param in enumerate(self.bit16_groups[group_index]):
                    new_index = self.round_robin_bit16_indices[group_index][param_index]
                    param.data = self.round_robin_bit16_groups[group_index][new_index].data
            #模型参数=buffer+偏移
            data_parallel_partitions = self.get_data_parallel_partitions(self.bit16_groups_flat[i], i)
            	base_size = total_num_elements // dp
                partitions.append(tensor.narrow(0, start, partition_size))
            #[Tensor]
            self.parallel_partitioned_bit16_groups.append(data_parallel_partitions)
            if orig_group_numel <= left_boundary:
                padding = curr_partition_size
            elif orig_group_numel < left_boundary + curr_partition_size:
                padding = left_boundary + curr_partition_size - orig_group_numel
            else:
                padding = 0
            self.groups_padding.append(padding)
            #自己负责的部分padding的数量
            if not fp16_master_weights_and_gradients:
            	weights_partition = self.parallel_partitioned_bit16_groups[i]								[partition_id].to(
            		self.device).clone().float().detach()
            #创建对应的32位参数
            self.single_partition_of_fp32_groups.append(weights_partition)
            self.single_partition_of_fp32_groups[
            i].requires_grad = True  
            param_group['params'] = [self.single_partition_of_fp32_groups[i]]
            #单卡优化器只有一个大Tensor(32位参数)
            partition_size = len(self.bit16_groups_flat[i]) / dist.get_world_size(group=self.real_dp_process_group[i])
            #对齐过，是整除的
            params_in_partition, params_not_in_partition, first_offset = 									self.get_partition_info(
            		self.round_robin_bit16_groups[i], partition_size, partition_id)
                for tensor in tensor_list:
                    tensor_size = tensor.numel()
                    if start_index <= current_index < end_index:
                        params_in_partition.append(tensor)
                    elif current_index < start_index < (current_index + tensor_size):
                        params_in_partition.append(tensor)
                        first_offset = start_index - current_index
                        #起点到本地的距离
                    else:
                        params_not_in_partition.append(tensor)
                    current_index = current_index + tensor_size
            #[Tensor],[Tensor],第一个部分tensor起点到本地的距离
            self.partition_size.append(partition_size)
            self.params_in_partition.append(params_in_partition)
            self.params_not_in_partition.append(params_not_in_partition)
            self.first_offset.append(first_offset)
        count = 0
        #在所有组里的id
        for i, params_group in enumerate(self.bit16_groups):
            for param in params_group:
                unique_id = id(param)
                self.param_id[unique_id] = count
                self.param_dict[count] = param
                self.params_already_reduced.append(False)
                if param.numel() > largest_param_numel:
                    largest_param_numel = param.numel()
                count = count + 1
        for param_group in self.params_in_partition:
        	for param in param_group:
                self.is_param_in_current_partition[self.get_param_id(param)] = True
        for param_group in self.params_not_in_partition:
            for param in param_group:
                self.is_param_in_current_partition[self.get_param_id(param)] = False
        self.initialize_gradient_partitioning_data_structures()
        	for i, param_group in enumerate(self.round_robin_bit16_groups):
        		#一个param可以在多个partition
				for partition_id in range(total_partitions):
					self.initialize_gradient_partition(i, param_group, partition_id)
                    	start_index = partition_size * partition_id
        				end_index = partition_size * (partition_id + 1)
                        current_index = 0#在组内起点
                    	for param in param_group:
                        	if start_index <= current_index < end_index:
            				elif current_index < start_index < 
            					(current_index + param_size):#部分在
            					first_offset = start_index - current_index
            					#起点到本地的距离
            					set_key_value_list(self.param_to_partition_ids[i], 											param_id, partition_id)
            					#param所在partiton
                				increment_value(self.total_grads_in_partition[i], 											partition_id)
                				self.grad_partition_insertion_offset[i][partition_id]										[param_id] = 0
                				#在本地的起点
                				self.grad_start_offset[i][partition_id][param_id] = 										first_offset
                				#起点到本地的距离
            				current_index = current_index + param_size
                    self.first_param_index_in_partition[i][partition_id] = 											self.get_first_param_index(
                    		i, param_group, partition_id)
                    	if partition_id in self.param_to_partition_ids[group_id][param_id]:
                    		return index
        self.reset_partition_gradient_structures()
        	self.remaining_grads_in_partition[i][partition_id] = self.total_grads_in_partition[i][partition_id]
        if (self.partition_gradients or self.overlap_comm or 											self.use_grad_accum_attribute
                or self.contiguous_gradients):
        	self.create_gradient_handling_hooks()
                for i, param_group in enumerate(self.bit16_groups):
                    for param in param_group:
                        if param.requires_grad:
                            def wrapper(param, i):
                                def grad_handling_hook(*notneeded):
                                    self.process_gradients(param, i)
                                self._grad_acc_hooks.append(
                                    register_grad_hook(param, grad_handling_hook))
                            wrapper(param, i)
        self.ready_for_gradients = False
        self._link_all_hp_params()
            for i, _ in enumerate(self.optimizer.param_groups):
                partition_id = dist.get_rank(group=self.real_dp_process_group[i])
                partition_size = self.bit16_groups_flat[i].numel() // dist.get_world_size(
                    group=self.real_dp_process_group[i])
                flat_hp_partition = self.single_partition_of_fp32_groups[i]
                link_hp_params(lp_param_list=self.bit16_groups[i],#bug,没有打乱顺序？？？
                               flat_hp_partition=flat_hp_partition,
                               gradient_dict=self.averaged_gradients,#{}
                               offload_gradient_dict=self.offload_gradient_dict,
                               use_offload=self.cpu_offload,
                               param_group_index=i,
                               partition_start=partition_id * partition_size,
                               partition_size=partition_size,
                               dp_group=self.real_dp_process_group[i])
    				local_lp_param_and_offset = _init_lp_to_hp_mapping(lp_param_list, partition_start, partition_size, dp_group)
    					for i, lp_param in enumerate(lp_param_list):
							if current_offset < partition_end and lp_param_end > 																partition_start:
                                param_and_offset_list.append((lp_param, current_offset))
                                lp_param._index_in_param_group = index_in_param_group
                                # Indices for params in this partition/GPU
                                index_in_param_group += 1
                            current_offset += lp_param.numel()
                    #[tensor,startIdx]
                    for lp_param, lp_start in local_lp_param_and_offset:
        				lp_param._hp_mapping = get_hp_fragment_mapping(lp_param, 										lp_start, flat_hp_partition, gradient_dict,
                                offload_gradient_dict, use_offload, param_group_index,
                                                       partition_start, partition_size)
                            hp_frag_address = fragment_address(start=fragment_start - 									hp_start, numel=fragment_numel)
                            #在分区里的起点>=0，在分区里的元素数量<=tensor.numel()
                            hp_fragment_tensor = flat_hp_partition.narrow(0, 											hp_frag_address.start, hp_frag_address.numel)
                            lp_frag_address = fragment_address(start=fragment_start - 									lp_start, numel=fragment_numel)
                            #在参数里的起点
                            lp_fragment_tensor = lp_param.flatten().narrow(0, 											lp_frag_address.start, lp_frag_address.numel)
		#建立16位参数到32位参数的连接
        self._param_slice_mappings = self._create_param_mapping()
        	param_mapping = []
            for i, _ in enumerate(self.optimizer.param_groups):
                param_mapping_per_group = OrderedDict()
                for lp in self.bit16_groups[i]:
                    if lp._hp_mapping is not None:
                        lp_name = self.param_names[lp]
                        param_mapping_per_group[lp_name] = lp._hp_mapping.get_hp_fragment_address()
                param_mapping.append(param_mapping_per_group)
         #[group_id][name]=hp_addr
```

# ZeroOptimizer.backward

```
reduce bucket在反向开始时创建，在反向结束时销毁
梯度先拷到bucket(释放旧tensor)再通信，通信完只拷出自己分区的梯度

zero1考虑什么时候开始允许通信，采用什么原语
zero2考虑采用什么原语
zero1完全可以实现zero2
```

算出梯度后，执行hook：

分配buffer for reduce

zero1+overlap_comm or zero2: 把grad拷贝到bucket，足够数量就allreduce/[reduce]

```
当reduce_scatter=True时(默认值)，理论上应该采用Reduce-Scatter模式，但实际实现中会根据use_multi_rank_bucket_allreduce的设置选择不同的通信路径

当use_multi_rank_bucket_allreduce=True时(默认值)，使用AllReduce操作
当use_multi_rank_bucket_allreduce=False时，使用一系列Reduce操作实现Reduce-Scatter效果

This feature is useful when the model is small and we want to scale it on too many GPUs which therefore reduces the message sizes of each packet.
```

```
    def process_gradients(self, param, i):
        self.backward_prologue()
        	if self.contiguous_gradients:
        		if not self.ready_for_gradients:
                    for _, bucket in self.ipg_buckets.items():
                        bucket.buffer.clear()
                        buf_0 = torch.empty(int(self.reduce_bucket_size),#超参数
                                            dtype=self.dtype,
                                            device=
                                            get_accelerator().current_device_name())
                        bucket.buffer.append(buf_0)
                        bucket.index = 0
                 	self.ready_for_gradients = True
        if self.use_grad_accum_attribute:
            self._fill_param_grad_accum_attribute(param)
        if self.partition_gradients or self.overlap_comm:
            self.reduce_ready_partitions_and_remove_grads(param, i)
            	self.reduce_independent_p_g_buckets_and_remove_grads(param, i)	
```

反向后对所有梯度

zero1+not overlap_comm+is_gradient_accumulation_boundary:

​	allreduce/[reduce] grad

zero1+is_gradient_accumulation_boundary or zero2:

1. reduce_ipg_grads
2. grad连续存储于self.averaged_gradients[group_id]
3. free bucket buffer

```
self.first_offset[group_id]
第一个部分tensor起点到本地的距离
self.averaged_gradients[group_id]
reduce后的grad:[Tensor]

	def reduce_gradients(self, pipeline_parallel=False):#zero1
        if not self.overlap_comm:
            for i, group in enumerate(self.bit16_groups):
                for param in group:
                    grad_reduc = self.get_gradient_for_reduction(param)
                    	if self.use_grad_accum_attribute:
            				return param.grad_accum.to(self.dtype) 
            					if param.grad_accum is not None else None
                        else:#
                            return param.grad
                    if grad_reduc is not None:
                        self.reduce_ready_partitions_and_remove_grads(param, i)
                        	if self.partition_gradients or 																	self.is_gradient_accumulation_boundary 
                        			or self.zenflow:
                                self.reduce_independent_p_g_
                                buckets_and_remove_grads(param, i)
        self.overlapping_partition_gradients_reduce_epilogue()
        	self.reduce_ipg_grads()
            if self.cpu_offload is False:
            	for i, _ in enumerate(self.bit16_groups):
                	if i not in self.averaged_gradients or self.averaged_gradients[i] is None:
                    	self.averaged_gradients[i] = self.get_flat_partition(
                                self.params_in_partition[i],
                                self.first_offset[i],
                                self.partition_size[i],
                                dtype=self.gradient_accumulation_dtype,
                                device=get_accelerator().current_device_name(),
                                return_tensor_list=True)
                            for i, tensor in enumerate(tensor_list):
                            	grad_accum = self.get_param_gradient_attribute(tensor)
                            	tensor = grad_accum
                                if i == 0 and first_offset > 0:
                                    tensor_offset = first_offset
                                    num_elements = num_elements - tensor_offset
                                if tensor_offset > 0 or num_elements < tensor.numel():
                                    flat_tensor_list.append(
                                    tensor.contiguous().view(-1).narrow(
                                    0, int(tensor_offset), int(num_elements)))
                                else:
                                    flat_tensor_list.append(tensor)
                    else:
                        avg_new = self.get_flat_partition()
                        for accumulated_grad, new_avg_grad in zip(
                        		self.averaged_gradients[i], avg_new):
                            accumulated_grad.add_(new_avg_grad)
            self._release_ipg_buffers()
            	if self.contiguous_gradients:
                    for bucket in self.ipg_buckets.values():
                        bucket.buffer.clear()
                    self.grads_in_partition = None
                    self.grads_in_partition_offset = 0
                self.ready_for_gradients = False
            self.zero_grad(set_to_none=True)
            	p.grad = None
```

zero1+overlap_comm hook

zero1+not overlap_comm+is_gradient_accumulation_boundary

zeor2 hook

grad拷贝到一维buffer，grad=buffer[idx]，释放旧tensor

凑够数量就reduce_ipg_grads

```
    def reduce_independent_p_g_buckets_and_remove_grads(self, param, i):
        grad_reduc = self.get_gradient_for_reduction(param)
        bucket = self.ipg_buckets[self.get_param_comm_dtype(param)]
        	return self.communication_data_type
        if bucket.elements + param.numel() > self.reduce_bucket_size:
        	self.reduce_ipg_grads()
        if self.contiguous_gradients:
        	new_grad_tensor = bucket.buffer[bucket.index].narrow(0, bucket.elements, 					param.numel())
            new_grad_tensor.copy_(
                    grad_reduc.view(-1) if not self.zenflow else grad_reduc.permute(
                        *reversed(range(grad_reduc.ndim))).contiguous().view(-1))
            grad_reduc.data = new_grad_tensor.data.view_as(grad_reduc) if (
                    not self.zenflow or grad_reduc.dim() == 1)
        bucket.elements += param.numel()
        if getattr(param, "ds_grad_is_ready", True):
            bucket.grads.append(grad_reduc)
            bucket.params.append((i, param.param_idx_in_group, param_id))
```

reduce_ipg_grads:

allreduce or [reduce] bucket

grad=copy(bucket[idx])，只拷贝自己负责的结果，存储于self.grads_in_partition

reset bucket

```
    def reduce_ipg_grads(self):
        for comm_dtype in sort_dtypes(self.ipg_buckets.keys()):
            bucket = self.ipg_buckets[comm_dtype]
            if self.contiguous_gradients:
            	self.average_tensor(bucket.buffer[bucket.index].
                		narrow(0, 0, bucket.elements), comm_dtype)
		stream = get_accelerator().current_stream()
        with get_accelerator().stream(stream):
            for comm_dtype in sort_dtypes(self.ipg_buckets.keys()):
                bucket = self.ipg_buckets[comm_dtype]

                for group_idx, param_idx_in_group, param_id in bucket.params:
                    param = self.bit16_groups[group_idx][param_idx_in_group]
                    if self.partition_gradients:
                    	if not self.is_param_in_current_partition[param_id]:
                            if self.overlap_comm and self.contiguous_gradients is False:
                            else:
                                self.clear_grad_attribute(param)
                        elif self.contiguous_gradients:
                            self.copy_grads_in_partition(param)
                    else:  # zero stage 1 - partition only optimizer state
                        if self.contiguous_gradients and self.is_param_in_current_partition[param_id]:
                            self.copy_grads_in_partition(param)
                            	if self.grads_in_partition is None:
                                    self.grads_in_partition_offset = 0
                                    total_size = 0
                                    for group in self.params_in_partition:
                                        for param_in_partition in group:
                                            total_size += param_in_partition.numel()
                                    self.grads_in_partition = 																			torch.empty(int(total_size),
                                                dtype=self.dtype,
                                                device=get_accelerator().
                                                current_device_name())
                                grad_reduc = self.get_gradient_for_reduction(param)
                                new_grad_tensor = self.grads_in_partition.view(-1).narrow(0, self.grads_in_partition_offset, param.numel())
                                new_grad_tensor.copy_(grad_reduc.view(-1))
                                grad_reduc.data = new_grad_tensor.data.view_as(grad_reduc)
                                self.grads_in_partition_offset += param.numel()
                bucket.clear()
                    self.params.clear()
                    self.grads.clear()
                    self.elements = 0
                    self.index = 0
                    self.has_moe_params = False
                    #tensor没有free
```

reduce bucket：

bucket划分成不同通信组的任务[(dst,Tensor)]

任务里的Tensor进行flatten，allreduce/[reduce]，只拷贝自己负责的结果

```
self.param_to_partition_ids[group_id][param_id]
param所在partitons:[partition_id]
self.grad_partition_insertion_offset[group_id][partition_id][param_id]
在本地的起点
self.grad_start_offset[group_id][partition_id][param_id]			
起点到本地的距离

    def average_tensor(self, tensor: torch.Tensor, communication_data_type: torch.dtype):
        stream = get_accelerator().current_stream()
        with get_accelerator().stream(stream):
        	prev_id, prev_process_group = -1, None
            for i, param_idx_in_group, param_id in bucket.params:
                partition_ids = self.param_to_partition_ids[i][param_id]
                partition_ids_w_offsets = []
                for partition_id in partition_ids:
                    offset = self.grad_start_offset[i][partition_id][param_id]
                    partition_ids_w_offsets.append((partition_id, offset))
                partition_ids_w_offsets.sort(key=lambda t: t[1])
                #按tensor顺序处理分区
                for idx in range(len(partition_ids_w_offsets)):
                    partition_id, offset = partition_ids_w_offsets[idx]
                    if idx == len(partition_ids_w_offsets) - 1:
                        numel = param.numel() - offset
                    else:
                        numel = partition_ids_w_offsets[idx + 1][1] - offset
                        #在当前分区的大小
                    if partition_id == prev_id and process_group == prev_process_group:
                        prev_pid, prev_size, prev_numel = rank_and_offsets[-1]
                        rank_and_offsets[-1] = (prev_pid, prev_size, prev_numel + numel)
                    else:
                        rank_and_offsets.append((partition_id, curr_size, numel))
                        #在bucket里的起点，len
                        real_dp_process_group.append(process_group)
                    curr_size += numel
                    prev_id, prev_process_group = partition_id, process_group
            tensor.div_(dist.get_world_size(group=self.dp_process_group) / float(self.sequence_parallel_size))
            buckets = {}
            for i, (dst, bucket_offset, numel) in enumerate(rank_and_offsets):
                grad_slice = tensor.narrow(0, int(bucket_offset), int(numel))
                bucket_key = real_dp_process_group[i] if self.use_multi_rank_bucket_allreduce else (
                    dst, real_dp_process_group[i])
                if bucket_key not in buckets:
                    buckets[bucket_key] = []
                if self.use_multi_rank_bucket_allreduce:
                    buckets[bucket_key].append((dst, grad_slice))
                else:
                    buckets[bucket_key].append(grad_slice)
            #不同通信组
            for bucket_key in buckets:
                if self.use_multi_rank_bucket_allreduce:
                    self.allreduce_and_scatter(buckets[bucket_key],
                                               communication_data_type,
                                               numel_per_bucket=self.reduce_bucket_size,
                                               divide=False,
                                               process_group=bucket_key)
                        for i, bucket_elem in enumerate(bucket):
                            rank, tensor = bucket_elem
                            small_bucket.append(tensor)
                            small_bucket_ranks.append(rank)
                            numel = numel + tensor.numel()
                        if len(small_bucket) > 0:
            				self.allreduce_and_copy_with_multiple_ranks(small_bucket,
                                                        communication_data_type,
                                                        log=None,
                                                        divide=divide,
                                                        process_group=process_group,
                                                        bucket_ranks=small_bucket_ranks)
        						allreduced = self.allreduce_bucket(small_bucket,
                                           communication_data_type,
                                           log=log,
                                           divide=divide,
                                           process_group=process_group)
                                	tensor = self.flatten(bucket)
                                	if rank is None:
            							dist.all_reduce(tensor_to_allreduce, 																	group=process_group)
                                for buf, synced, bucket_rank in zip(small_bucket, self.unflatten(allreduced, small_bucket), bucket_ranks):
                                    if dist.get_rank(group=process_group) == bucket_rank:
                                        buf.copy_(synced)
```

# ZeroOptimizer.step

只处理averaged_gradients

check_grad_overflow

calculating L2 norm

分组处理：

1. 16位梯度转成32位，更新32位参数的梯度，释放16位梯度
2. unscale_and_clip_grads
3. self.optimizer.step()，建立16位参数到2*32位优化器状态的连接
4. 释放32位梯度，用32位参数更新16位参数

分区参数allgather到一个一维tensor，allgather_bucket_size没有用到

```
self.parallel_partitioned_bit16_groups[group_id]
按dp平分一维Tensor: [Tensor]

self.single_partition_of_fp32_groups[group_id]
创建所在dp的32位参数: Tensor，设为优化器参数

self.bit16_groups_flat[group_id]
展成一维（要2*dp对齐），移到GPU，模型参数=buffer+偏移

    def step(self, closure=None):
        if self.check_grad_overflow:
            self.check_overflow(partition_gradients=self.partition_gradients)
				self.overflow = self.has_overflow(partition_gradients)
                	overflow = self.local_overflow if self.cpu_offload else 
                			self.has_overflow_partitioned_grads_serial()
                        for i in range(len(self.bit16_groups)):
                            for j, grad in enumerate(self.averaged_gradients[i]):
                                if grad is not None:
                                    invalid_grad_count += self._has_inf_or_nan(grad)
                                    	float_x = x.float()
                                        nan = float_x.isnan()
                                        inf = float_x.isinf()
                                        inf_or_nan = nan.logical_or(inf)
                                        return inf_or_nan.float().max()
                        return invalid_grad_count.bool()
                    self._model_parallel_all_reduce(tensor=overflow_gpu, op=dist.ReduceOp.MAX)
                        if self.model_parallel_group is None or 
                        		self.model_parallel_world_size == 1:
                            pass
                        else:
                            dist.all_reduce(tensor=tensor, op=op, 
                            	group=self.model_parallel_group)
                    overflow = overflow_gpu[0].item()
                    return bool(overflow)
        scaled_global_grad_norm = self.scaled_global_norm()
        	for i, group in enumerate(self.bit16_groups):
                norm_groups.append(self.get_grad_norm_direct(self.averaged_gradients[i], self.params_in_partition[i]))
            		for g, p in zip(gradients, params):
                		if is_model_parallel_parameter(p) or 
                			(self.model_parallel_rank == 0):
                    		all_norms.append(
                        		torch.linalg.vector_norm(g.data.double().detach(),
                                                 ord=norm_type).to(
                                                 get_accelerator().
                                                 current_device_name()))
                    if len(all_norms) > 0:
                		total_norm = torch.stack(all_norms).square().sum().float()
                    dist.all_reduce(total_norm, op=dist.ReduceOp.SUM,
                    	group=self.dp_process_group)
                    self._model_parallel_all_reduce(tensor=total_norm, 
                    	op=dist.ReduceOp.SUM)
            		total_norm = total_norm.pow(1. / norm_type)
                	mask_nan_or_inf_with_val_inplace(total_norm, device=self.device)
            return torch.linalg.vector_norm(torch.stack(norm_groups), ord=norm_type)
        for i, group in enumerate(self.bit16_groups):
			if partition_id == dist.get_world_size(
					group=self.real_dp_process_group[i]) - 1:
            	single_grad_partition = self.flatten_dense_tensors_aligned(
                        self.averaged_gradients[i],
                        int(self.partition_size[i])).
                        to(self.single_partition_of_fp32_groups[i].dtype)
            else:
            	single_grad_partition = self.flatten(self.averaged_gradients[i]).to(
                        self.single_partition_of_fp32_groups[i].dtype)
            self.single_partition_of_fp32_groups[i].grad = single_grad_partition
            self.averaged_gradients[i] = None
            self.unscale_and_clip_grads([single_grad_partition], scaled_global_grad_norm)
            	grad.data.mul_(1. / combined_scale)
            self._optimizer_step(i)
            	original_param_groups = self.optimizer.param_groups
            	self.optimizer.param_groups = [original_param_groups[group_no]]
            	self.optimizer.step()
            	self.optimizer.param_groups = original_param_groups
            	self._lazy_init_hp_params_optimizer_state()
					if not self._hp_optimizer_states_linked:
                        for i, _ in enumerate(self.optimizer.param_groups):
                            lazy_init_hp_params_optimizer_state(self.bit16_groups[i], 
                            		self.single_partition_of_fp32_groups[i],
                                                                self.optimizer.state)
                                for lp in lp_param_list:
                                    if lp._hp_mapping is not None:
                                            lp._hp_mapping.set_optim_state_fragment(
                                            flat_hp_partition, optimizer_state[
                                            flat_hp_partition])
                                        #dict_keys(['step', 'exp_avg', 'exp_avg_sq'])
                                        self.optim_fragment = {
                                            key: value.narrow(0, self.
                                            	hp_fragment_address.start
                                            	, self.hp_fragment_address.numel)
                                                for key, value in optim_fragment.items()
                                                if torch.is_tensor(value) and 
                                                value.shape == flat_hp_partition.shape
                                        	}
                        self._hp_optimizer_states_linked = True
            self.single_partition_of_fp32_groups[i].grad = None
            bit16_partitions = self.parallel_partitioned_bit16_groups[i]
            fp32_partition = self.single_partition_of_fp32_groups[i]
            bit16_partitions[partition_id].data.copy_(fp32_partition.data)
        all_gather_dp_groups(groups_flat=self.bit16_groups_flat,
                             partitioned_param_groups=
                             self.parallel_partitioned_bit16_groups,
                             dp_process_group=self.real_dp_process_group,
                             start_alignment_factor=self.nccl_start_alignment_factor,
                             allgather_bucket_size=self.allgather_bucket_size)
        	#allgather_bucket_size没有用到
        	return all_gather_into_tensor_dp_groups(groups_flat, partitioned_param_groups, dp_process_group)
                for group_id, (group_flat, partitioned_params) in enumerate(zip(groups_flat, partitioned_param_groups)):
                    partition_id = dist.get_rank(group=dp_process_group[group_id])
                    dist.all_gather_into_tensor(group_flat, partitioned_params[partition_id], dp_process_group[group_id])
			#分区allgather到一个tensor
```

# Engine.init

```
def initialize(args=None,
    dist.init_distributed(dist_backend=dist_backend,
    engine = DeepSpeedEngine(args=args,
    return_items = [
        engine,
        engine.optimizer,
        engine.training_dataloader,
        engine.lr_scheduler,
    ]

class DeepSpeedEngine(Module):
    def __init__(self,
		self._configure_distributed_model(model)
			if self.fp16_enabled():
            	self.module.half()
			if not (self.dont_change_device or is_zero_init_model):
            	self.module.to(self.device)
			self.data_parallel_group = groups._get_data_parallel_group()
        self.param_names = {param: name for name, param in model.named_parameters()}
        if has_optimizer:
            # Convert model parameters from generator to list
            if not isinstance(model_parameters, list):
                model_parameters = list(model_parameters)
            self._configure_optimizer(optimizer, model_parameters)
            	basic_optimizer = self._configure_basic_optimizer(model_parameters)
            		optimizer = FusedAdam(
                        model_parameters,
                        **optimizer_parameters,
                        adam_w_mode=effective_adam_w_mode,
                    )
				self.optimizer = self._configure_zero_optimizer(basic_optimizer)
					model_dtype, gradient_accumulation_dtype = self.get_data_types()
						if self.fp16_enabled():
            				model_dtype = torch.float16
                        if self._config.grad_accum_dtype is None:
                            if model_dtype == torch.bfloat16 and not 														self.zero_optimization():
                                grad_accum_dtype = torch.float32
                            else:
                                grad_accum_dtype = model_dtype
                    if self.bfloat16_enabled():
                        check_grad_overflow = 
                        	self._config.bfloat16_config.check_grad_overflow
                    elif self.fp16_enabled():
                        check_grad_overflow = True
                    else:
                        check_grad_overflow = False
                    overlap_comm = self.zero_overlap_comm()#False
                    #overlap the reduction of the gradients with backward computation
                    contiguous_gradients = self.zero_contiguous_gradients()#True
                    #Copies the gradients to a contiguous buffer as they are produced.
                    #Avoids memory fragmentation during backward pass.
					round_robin_gradients = self.zero_round_robin_gradients()#False
					#for CPU offloading that parallelizes gradient copying
					Stage1And2ZeroOptimizer = DeepSpeedZeroOptimizer
					optimizer = Stage1And2ZeroOptimizer(
                        optimizer,
                        self.param_names,
                        dp_process_group=self.seq_data_parallel_group,
                        partition_grads=zero_stage == ZeroStageEnum.gradients,
    @property
    def communication_data_type(self):
        res = self._config.communication_data_type
        if res is not None:
            return res
        if self.fp16_enabled():
            return torch.float16
        if self.bfloat16_enabled():
            return torch.bfloat16
        return torch.float32
```

# Engine.backward

反向后对所有梯度allreduce/[reduce]

zero1+is_gradient_accumulation_boundary

zero2

```
outputs = model_engine(inputs)
	loss = self.module(*inputs, **kwargs)
	
model_engine.backward(loss)
	loss = self._backward_prologue(loss, scale_wrt_gas)
    self._do_optimizer_backward(loss, retain_graph)
        self.optimizer.backward(loss, retain_graph=retain_graph)
            self.loss_scaler.backward(loss.float(), retain_graph=retain_graph)
            self.backward_epilogue()
            	if self.use_grad_accum_attribute:#False
    self._backward_epilogue()
    	self.allreduce_gradients()
    		if self.zero_optimization_partition_gradients()://zero2,3
    			self.optimizer.overlapping_partition_gradients_reduce_epilogue()
            elif self.is_gradient_accumulation_boundary():
                if self.zero_optimization_stage() == ZeroStageEnum.optimizer_states and 						hasattr(self.optimizer, 'reduce_gradients'):
                    self.optimizer.reduce_gradients
                        (pipeline_parallel=self.pipeline_parallelism)
                else:
                    grads = None
                    self.buffered_allreduce_fallback(grads=grads, 												elements_per_buffer=bucket_size)
```

# Engine.step

```
model_engine.step()
	if self.is_gradient_accumulation_boundary():
		self._take_model_step(lr_kwargs)
			self.optimizer.step()
			overflow = False
            if hasattr(self.optimizer, "overflow"):
                overflow = self.optimizer.overflow
            self._step_applied = not overflow
            self.global_steps += 1
	self.micro_steps += 1
```

# ZeroOptimizer3...

# Inference 

支持模型并行性

```
ds_engine = deepspeed.init_inference(model,
```

DeepSpeed-Inference v2 来了，它被称为 DeepSpeed-FastGen

DeepSpeed-FastGen 是 DeepSpeed-MII 和 DeepSpeed-Inference 的协同组合

MII features include blocked KV-caching, continuous batching, Dynamic SplitFuse

# DeepSpeed-HE

DeepSpeed-HE can automatically change the ZeRO-3 training mode to Tensor Parallelism 
