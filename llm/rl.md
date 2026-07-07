# 梯度下降

```
如果你能知道 J(θ) 对 θ 的导数（梯度），那么沿着梯度方向更新 θ，J(θ) 就会变大
```

# pretrain

AutoModelForCausalLM
$$
L(\theta)=-[\sum_tlog\pi_{\theta}(y_t^{label}|y_{<t})]/T
$$

```
input_ids = batch[:, :-1]   
target_ids = batch[:, 1:]   
logits = model(input_ids) 
loss = cross_entropy(
    input=logits.view(-1, vocab_size),    
    target=target_ids.view(-1),             
)
H(p,q)=-\sum_xp(x)log(q(x))=-log(q(x^label))

probs = softmax(input,dim=-1)  
p_correct = probs[:,target]
loss = -log(p_correct).mean()
```

# SFT

```
input_ids = tokenizer(prompt + response).input_ids
# 只计算 response 部分的损失
labels = input_ids.clone()
labels[:len(prompt_tokens)] = -100
loss = cross_entropy(model(input_ids), labels))
```

lora: x(W+AB)，参数量dk->dr+rk，计算量ndk->ndr+nrk

prompt-tuning: 通过可训练的连续提示优化模型输入

Adapter 在 FFN 层后插入 MLP

# REINFORCE

没有label，只能根据回报来控制正负
$$
L(\theta)=-[\sum_tlog\pi_{\theta}(a_t|s_t)*G_t]/T
$$

```
π_θ(a_t | s_t)：在状态 s_t 下，选择动作 a_t 的概率
G_t：从时间 t 开始到结束的总回报（奖励总和）
混合了“当前动作的贡献” + “未来随机事件的影响”，导致梯度噪声极大。
```

# Actor-Critic

```
G_t = 真实回报 = 动作 a 的贡献 + 环境噪声
V(s_t) = 平均回报 = 平均动作贡献 + 平均噪声
A_t = G_t − V(s_t) (蒙特卡洛) 或 r_t + γV(s_{t+1}) - V(s_t) (TD(0))
排除了噪声
如果我在状态 s 下特意选了动作 a，而不是随机选，我能多得多少分？
```

```
初始化策略参数 θ（Actor），价值网络参数 φ（Critic）
重复：
  1. 用 π_θ 与环境交互，收集一批轨迹（或固定步数）
  2. 对每个时间步 t：
        - 计算折扣回报 G_t
        - 计算优势：Â_t = G_t - V_φ(s_t)
  3. 更新 Actor（策略）：
        θ ← θ + α_θ * Σ_t ∇_θ log π_θ(a_t|s_t) * Â_t
  4. 更新 Critic（价值网络）：
        φ ← φ - α_φ * Σ_t ∇_φ (G_t - V_φ(s_t))²   // 预测 ≈ 真实 → 均方误差（MSE） 
```

# TRPO

如果步长 α 太大，或曲率变化剧烈 → 新策略可能比旧策略更差

每次更新策略时，必须保证新策略和旧策略足够接近，不能跑太远

约束：新旧策略的 KL 散度 ≤ δ

```
Eτ∼π_new[R(τ)]
=Eτ∼π_old[P_new(τ)/P_old(τ)R(τ)]
=Es,a∼π_old[π_θ(a|s)/π_old(a|s)A_old(s,a)]
δt = r_t + γV(s_{t+1}) - V(s_t)
At = δt + (γλ)At+1
λ 越大 → 越接近蒙特卡洛（无偏，高方差）
λ 越小 → 越接近 TD(0)（有偏，低方差）
```

# PPO

用裁剪代替 KL 约束

```
r_t(θ)=π_θ(a|s)/π_old(a|s)
Et[min(r_t(θ)A_t, clip(r_t(θ),1−ϵ,1+ϵ)A_t)]
```

# RLHF

## actor

sft大模型（好的回答）, actor==sft

## reward model

sft奖励模型（区分好坏回答）, critic==reward

reward model=automodel+linear(hidden_size,1)

```
prompt: [11, 22, 33]
chosen_sentence: [11, 22, 33, 44, 55, 66, 0, 0, 0, 0] 
reject_sentence: [11, 22, 33, 40, 50, 0, 0, 0, 0, 0]
#计算response的loss
loss = -torch.log(torch.sigmoid(c_truncated_reward - r_truncated_reward)).mean()
```

模型在预测的时候，使用的reward是最后一个token的打分

## ppo

```
inference:actor.inference->(actor,critic,sft,reward).forward->dataset
train:dataset->(actor,critic)
```

时间步 t 就是生成序列中的第 t 个 token

actor model需要推理t次

计算A需要reward model和critic model做一次前向

在ppo的基础上增加KL 散度约束：锚定原始语言模型

```
L_KL=β⋅KL(πθ(⋅∣s)||πSFT(⋅∣s))
```

```
with torch.no_grad():
    response = actor.generate(prompt)
    log_probs_old = actor.log_prob(prompt + response)  
    values = critic(prompt + response) 
    rewards = torch.zeros(T)
    rewards[-1] = reward_model(prompt + response)
    advantages = compute_gae(rewards, values, gamma=1.0, lam=0.95)
    returns = advantages + values  # G_t = A_t + V(s_t)
    ref_log_probs = sft_model.log_prob(prompt + response)

for k = 1 to K:
	log_probs = actor.log_prob(prompt + response) 
	values = critic(prompt + response) 
    ratio = torch.exp(log_probs - log_probs_old)
    L_actor = -mean( min(ratio * A_t, clip(ratio,1−ε,1+ε) * A_t) )
	L_critic = mean( (values - returns)^2 )
	kl = log_probs - ref_log_probs  # ≈ KL(π_θ || π_SFT)
	L_total = L_actor + c1 * L_critic + beta * kl_div.mean()
	θ ← θ - α_θ * ∇_θ L_total
    φ ← φ - α_φ * ∇_φ L_total
```

多次更新是为了提高样本效率，clip 机制天然限制了更新幅度，即使更新多次，策略也不会偏离“生成数据时的旧策略”太远

# dschat

```
from dschat.rlhf.rlhf_engine import DeepSpeedRLHFEngine
from dschat.rlhf.ppo_trainer import DeepSpeedPPOTrainer
engine = DeepSpeedRLHFEngine(actor_model_name_or_path=actor_model_name_or_path,
                             critic_model_name_or_path=critic_model_name_or_path,
                             tokenizer=tokenizer,
                             args=args)
	self.actor = self._init_actor(
		actor_model = create_hf_model(
		actor_engine, *_ = deepspeed.initialize(model=actor_model,
			engine = DeepSpeedHybridEngine(args=args,
	self.ref = self._init_ref(
		ref_engine, *_ = deepspeed.initialize(model=ref_model,
		
trainer = DeepSpeedPPOTrainer(engine=engine, args=args)

for prompt_batch in prompt_train_dataloader:
    out = trainer.generate_experience(prompt_batch)
    	seq = self._generate_sequence(prompts, mask, step)
    	with torch.no_grad():
            output = self.actor_model(seq, attention_mask=attention_mask)
    exp_dataset = exp_mini_dataset.add(out)
    for ppo_ep in range(args.ppo_epochs):
    	for i, (exp_data, unsup_data) in enumerate(
    		actor_loss, critic_loss = trainer.train_rlhf(out)
    			actor_prob = self.actor_model(**batch, use_cache=False).logits
```

# verl

灵活性：RLHF算法与特定的LLM训练框架紧密耦合，新增算法和并行方式困难

分成两层：ray+训推框架

actor性能：几个副本，是否变化。DeepSpeed-Chat 和 OpenRLHF 采用ZeRO-3进行演员训练，TP用于演员生成。OpenRLHF使用不同设备上的演员模型的不同副本进行训练和生成，导致冗余的内存使用和设备之间频繁的权重同步。DeepSpeed-Chat在同一组设备上维护相同的Actor模型副本用于训练和生成，并在训练和生成之间重新分配模型权重（由于两个阶段中使用的不同并行性），这对于大型模型仍然可能产生大量的内存和通信开销。NeMo-Aligner 在演员训练和生成中使用相同的3D并行配置，经历低生成吞吐量。

```
𝑁𝑎=𝑝×𝑡×𝑑=𝑝𝑔×𝑡𝑔×𝑑𝑔×𝑑
train:[dp,pp,tp]
reduce pp, tp
allgather in dp_i, dp_dpt=[p,t], discard unused, may keep train weight
allgather in dg_i, dg_dpt=[p/pg,t/tg], zero weight redundancy, 减少显存峰值和通行量
```

We store actor model weights for training and generation stages on separate memory buffers, offload generation weights to the CPU memory during training, reload generation weights back to GPU
memory during the transition, and use both buffers in generation. 生成阶段需要的参数多一些，一部分是训练的，一部分通过通信更新（需要recv buffer）

我们在生成后将KVCache卸载到CPU内存，并在下一次迭代中将其重新加载回GPU。

模型放置：DeepSpeed-Chat 全放在所有gpu上，低效并行；OpenRLHF，NeMo-Aligner，百川分开放置，空泡

枚举模型组队情况，枚举组的资源数量，求解每个模型的最快并行
