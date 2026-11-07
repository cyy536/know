通信库，训练和推理都能用

DeepEP provides high-throughput and low-latency all-to-all GPU kernels

expand 指的是 token 扩展了？

expand_x[N',D],expand_weight[N'],cumsum[E]=dispatch(x[N,D],topk_weight[N,K])

[N,D]=combine([N',D])
```
from deep_ep import ElasticBuffer

recv_x,recv_topk_idx,recv_scores,handle,_event=buffer.dispatch(x)
```

