FP8 and FP4 on Hopper, Ada and Blackwell GPUs

Modules provided by TE internally maintain scaling factors and other values needed for FP8 training, greatly simplifying mixed precision training for users.

both training and inference

NVIDIA Megatron-LM、DeepSpeed已集成

deepseek-v3采用fp8混合精度训练

```
import transformer_engine.pytorch as te
te.Linear
te.LayerNorm
te.LayerNormLinear
#fuses LayerNorm + Linear
te.LayerNormMLP
#fuses LayerNorm + MLP

te.DotProductAttention

te.TransformerLayer

with te.autocast(enabled=True, recipe=recipe):
    y = model(x, attention_mask=None)
#Wrap your code within an autocast context manager to enable FP8.

te.optimizers.FusedAdam
```

```
class Linear(TransformerEngineBaseModule):
    Parallelism parameters
    ----------------------
    sequence_parallel : bool, default = False
                       if set to ``True``, uses sequence parallelism.
    tp_group : ProcessGroup, default = None
              tensor parallel process group.

class DotProductAttention(TransformerEngineBaseModule):
    @no_torch_dynamo(recursive=False)
    def forward(
            DotProductAttention supports three backends: 1) FlashAttention which calls
            HazyResearch/Dao-AILab's `flash-attn <https://arxiv.org/pdf/2305.13245.pdf>`_
            PyTorch API, 2) FusedAttention which has multiple fused attention implementations
            based on `cuDNN Graph API
            <https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html#op-fusion>`_
            (see :attr:`FusedAttention` for more details on FusedAttention backends), and 3)
            UnfusedDotProductAttention which is the native PyTorch implementation
            with fused scaled masked softmax.

class TransformerEngineBaseModule(torch.nn.Module, ABC):
```
