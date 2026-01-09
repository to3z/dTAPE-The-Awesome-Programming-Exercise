import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AttentionalQMixer(nn.Module):
    def __init__(self, args):
        super(AttentionalQMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        
        # === 关键改进参数 ===
        # 使用 Attention 的 Head 数量作为 QMIX 的中间嵌入维度 (embed_dim)
        # 如果 args 中没有 n_attention_head，默认设为 4
        self.n_heads = getattr(args, 'n_attention_head', 4) 
        self.embed_dim = self.n_heads # 对齐维度：每个 Head 对应一个混合通道
        
        # 尝试获取每个智能体的特征大小，如果 state 中不包含结构化信息，则需要 fallback
        self.agent_own_state_size = getattr(args, 'agent_own_state_size', 0)
        self.u_dim = int(np.prod(self.agent_own_state_size)) if self.agent_own_state_size > 0 else 64

        # Query 网络：把全局 State 映射为 Query
        self.query_embedding_layers = nn.ModuleList()
        # Key 网络：把每个智能体的局部特征映射为 Key
        self.key_embedding_layers = nn.ModuleList()
        
        # 如果没有明确的 agent_own_state_size，我们用全连接层从 global state 提取特征 (Fallback)
        self.use_feature_extraction = (self.agent_own_state_size == 0)
        if self.use_feature_extraction:
            self.feature_extractor = nn.Linear(self.state_dim, self.u_dim * self.n_agents)

        # 构建 Attention Heads (替代原版 QMIX 的 hyper_w_1)
        # 这里对应 qatten.py 的 query/key 结构
        for i in range(self.n_heads):
            # State -> Query
            self.query_embedding_layers.append(nn.Sequential(
                nn.Linear(self.state_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32) # Query dim
            ))
            # Agent Feature -> Key
            self.key_embedding_layers.append(nn.Linear(self.u_dim, 32)) # Key dim

        self.scaled_product_value = np.sqrt(32) #用于缩放 Dot Product

        # === QMIX Layer 1 偏置 (b1) ===
        # 保持原版 QMIX 的结构，用 Linear 生成偏置
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # === QMIX Layer 2 (hyper_w_final) ===
        # 保持原版 QMIX 的结构，用 Linear 生成第二层的权重和偏置
        # 输入是 state，输出是 embed_dim -> 1 的权重
        hypernet_embed = getattr(self.args, "hypernet_embed", 64)
        
        self.hyper_w_final = nn.Sequential(
            nn.Linear(self.state_dim, hypernet_embed),
            nn.ReLU(inplace=True),
            nn.Linear(hypernet_embed, self.embed_dim)
        )
        
        # 最终的 V(s) 偏置
        self.V = nn.Sequential(
            nn.Linear(self.state_dim, hypernet_embed),
            nn.ReLU(inplace=True),
            nn.Linear(hypernet_embed, 1)
        )

    def forward(self, agent_qs, states, dropout=False):
        # agent_qs Shape: [batch, 1, n_agents] (从 learner 传过来通常是这个形状)
        bs = agent_qs.size(0)
        
        if dropout:
            # 兼容代码里的 Dropout 逻辑
             states = states.reshape(states.shape[0], states.shape[1], 1, states.shape[2]).repeat(1, 1, self.n_agents, 1)

        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.reshape(-1, 1, self.n_agents)

        # === 1. 计算 Attention Weights (替代原来的 hyper_w_1) ===
        # 获取智能体特征 us: [batch * n_agents, u_dim]
        us = self._get_us(states, bs) 
        
        w1_heads = []
        for i in range(self.n_heads):
            # Query: 来自 Global State
            state_embedding = self.query_embedding_layers[i](states) # [batch, 32]
            state_embedding = state_embedding.unsqueeze(1) # [batch, 1, 32]
            
            # Key: 来自 Agent Features
            # 此时 us 的形状是 [Batch*Time*N_Agents, U_Dim]
            # 要将其还原为 [Batch*Time, N_Agents, U_Dim]
            # 不要用 bs，因为 bs 只是 episode 的数量，不包含 Time
            u_reshaped = us.reshape(-1, self.n_agents, self.u_dim)
            u_embedding = self.key_embedding_layers[i](u_reshaped) # [batch, n_agents, 32]
            
            # Attention Score: (Q * K^T)
            # [batch, 1, 32] * [batch, 32, n_agents] -> [batch, 1, n_agents]
            raw_scores = th.matmul(state_embedding, u_embedding.transpose(1, 2)) 
            raw_scores = raw_scores / self.scaled_product_value
            
            # Softmax 保证权重为正 (单调性)，且和为1 (这也是一种归一化)
            attention_w = F.softmax(raw_scores, dim=-1) # [batch, 1, n_agents]
            
            w1_heads.append(attention_w)
            
        # 堆叠 Heads
        # w1 Shape: [batch, n_agents, n_heads]
        # 注意: QMIX原本是 [batch, n_agents, embed_dim]。这里 n_heads 就是 embed_dim
        w1 = th.cat(w1_heads, dim=1) 
        w1 = w1.transpose(1, 2) # [batch, n_agents, n_heads]

        # === 2. 第一层混合 (Mixing) ===
        # agent_qs: [batch, 1, n_agents]
        # w1:       [batch, n_agents, n_heads]
        # matmul -> [batch, 1, n_heads]
        # 这里的数学含义是：每个 Head 基于 Attention 对 Agent Qs 进行了一次加权求和
        hidden = th.matmul(agent_qs, w1)
        
        # 加上偏置 b1
        b1 = self.hyper_b_1(states).view(-1, 1, self.embed_dim)
        hidden = hidden + b1
        
        # 激活函数
        hidden = F.elu(hidden)

        # === 3. 第二层混合 (Standard QMIX) ===
        # 生成第二层权重 w_final
        w_final = self.hyper_w_final(states).abs() # 仍然需要 abs 保证单调性
        w_final = w_final.view(-1, self.embed_dim, 1)
        
        # 生成状态价值 V(s)
        v = self.V(states).view(-1, 1, 1)

        # 计算最终 Q_tot
        # [batch, 1, embed_dim] * [batch, embed_dim, 1] -> [batch, 1, 1]
        y = th.matmul(hidden, w_final) + v
        
        q_tot = y.view(bs, -1, 1)
        return q_tot

    def _get_us(self, states, bs):
        # 从 State 中提取每个 Agent 的特征
        # 在 SMAC 中，State 的前部分通常是 Agent 的特征拼接
        
        if self.use_feature_extraction:
            # 如果没有 args 信息，使用 MLP 提取
            extract = self.feature_extractor(states) # [batch, u_dim * n_agents]
            return extract.reshape(-1, self.u_dim)
        else:
            # 按照 qatten 的逻辑进行切片
            agent_own_state_size = self.args.agent_own_state_size
            with th.no_grad():
                # 假设 state 结构是 [agents_feat, enemies_feat, ...]
                us = states[:, :agent_own_state_size * self.n_agents]
                us = us.reshape(-1, agent_own_state_size)
            return us