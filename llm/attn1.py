class Attn(nn.Module):
    def __init__(self,hidden_size):
        self.q_proj=nn.Linear(hidden_size,numQ*H)
        self.k_proj=nn.Linear(hidden_size,numK*H)
        self.v_proj=nn.Linear(hidden_size,numV*H)
        self.o_proj=nn.Linear(numQ*H,hidden_size)
    def forward(self,hidden_states):
        Q=self.q_proj(hidden_states).reshape(B,L,-1,D).transpose(1,2)
        K=self.k_proj(hidden_states).reshape(B,L,-1,D).transpose(1,2)
        V=self.v_proj(hidden_states).reshape(B,L,-1,D).transpose(1,2)

        Q,K=apply_rope(Q,K)

        K=torch.cat([past_key_value[0],K],dim=-2)
        V=torch.cat([past_key_value[1],V],dim=-2)
        past_key_value=[K,V]

# /////////////////////None pos
        K=K[:,:,None,:,:].expand(B,numK,numQ//numK,L,D).reshape(B,-1,L,D)
        V=V[:,:,None,:,:].expand(B,numV,numQ//numV,L,D).reshape(B,-1,L,D)

        S=nn.functional.softmax(torch.matmul(Q,K.transpose(-1,-2))/scale+mask,dim=-1)
        # Q 和 K 的点积结果是两个向量的内积，如果向量的维度 d_k 很大，点积的值可能会非常大
        # softmax 的输入值过大时，注意力权重会变得非常尖锐
        O=torch.matmul(S,V).transpose(1,2).reshape(B,L,-1)
        O=self.o_proj(O)
        return O
class Mlp(nn.Module):
    def __init__(self):
        self.up_proj=nn.Linear(hidden_size,intermediate_size)
        self.gate_proj=nn.Linear(hidden_size,intermediate_size)
        self.down_proj=nn.Linear(intermediate_size,hidden_size)
        self.act=nn.Swish()
    def forward(self,hidden_states):
        return self.down_proj(self.act(self.gate_proj(hidden_states))*self.up_proj(hidden_states))
class Layer(nn.Module):
    def __init__(self):
        self.ln1=RMSNorm(hidden_size)
        self.attn=Attn(hidden_size)
        self.ln2=RMSNorm(hidden_size)
        self.mlp=Mlp(hidden_size)
    def forward(self,hidden_states):
        re=hidden_states
        hidden_states=self.ln1(hidden_states)
        hidden_states=self.attn(hidden_states)
        hidden_states+=re
        
        re=hidden_states
        hidden_states=self.ln2(hidden_states)
        hidden_states=self.mlp(hidden_states)
        hidden_states+=re
        return hidden_states