# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
            # attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
            #     `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
class LlamaMLP(nn.Module):
    def __init__(self):
        self.up_proj=nn.Linear(hidden_size,intermediate_size)
        self.gate_proj=nn.Linear(hidden_size,intermediate_size)
        self.down_proj=nn.Linear(intermediate_size,hidden_size)
        self.act_fn=nn.Swish()
    def forward(self,hidden_states):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_states))*self.up_proj(hidden_states))
class LlamaAttention(nn.Module):
    def __init__(self):
        self.q_proj=nn.linear(hidden_size,head_dim*num_query_heads)
        self.k_proj=nn.linear(hidden_size,head_dim*num_kv_heads)
        self.v_proj=nn.linear(hidden_size,head_dim*num_kv_heads)
        self.o_proj=nn.linear(head_dim*num_query_heads,hidden_size)
    def forward(self,hidden_states):
        Q=self.q_proj(hidden_states).reshape(B,L,-1,D).transpose(1,2)
        K=self.k_proj(hidden_states).reshape(B,L,-1,D).transpose(1,2)
        V=self.v_proj(hidden_states).reshape(B,L,-1,D).transpose(1,2)
        
        Q,K=apply_rope(Q,K)
        
        K=torch.cat([past_key_value[0],K],dim=-2)
        V=torch.cat([past_key_value[1],V],dim=-2)
        past_key_value=[K,V]

        K = K[:, :, None, :, :].expand(B, num_kv_heads, num_query_heads//num_kv_heads, L, D).reshape(B,-1,L,D)
        V = V[:, :, None, :, :].expand(B, num_kv_heads, num_query_heads//num_kv_heads, L, D).reshape(B,-1,L,D)
        # expand 只能扩展为1的维度
        
        S=nn.functional.softmax(torch.matmul(Q,K.transponse(-1,-2))/math.sqrt(D)+MASK,dim=-1)
        O=torch.matmul(S,V).transpose(1,2).reshape(B,L,-1)

        O=self.o_proj(O)

        return O
class transformerBlock(nn.Module):
    def __init__(self):   
        self.self_attn=LlamaAttention()
        self.mlp=LlamaMLP()
        self.ln1=LlamaRMSNorm(hidden_size)
        self.ln2=LlamaRMSNorm(hidden_size)

    def forward(self,hidden_states):
        res=hidden_states
        hidden_states=self.ln1(hidden_states)
        hidden_states=self.self_attn(hidden_states)
        hidden_states=res+hidden_states
        
        res=hidden_states
        hidden_states=self.ln2(hidden_states)
        hidden_states=self.mlp(hidden_states)
        hidden_states=res+hidden_states
        return hidden_states