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
        
        self.agent_own_state_size = getattr(args, 'agent_own_state_size', 0)
        self.u_dim = int(np.prod(self.agent_own_state_size)) if self.agent_own_state_size > 0 else 64

        # Query 网络：把全局 State 映射为 Query
        self.query_embedding_layers = nn.ModuleList()
        # Key 网络：把每个智能体的局部特征映射为 Key
        self.key_embedding_layers = nn.ModuleList()
        
        self.use_feature_extraction = (self.agent_own_state_size == 0)
        if self.use_feature_extraction:
            self.feature_extractor = nn.Linear(self.state_dim, self.u_dim * self.n_agents)

        # 构建 Attention Heads (替代原版 QMIX 的 hyper_w_1)
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
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)
        hypernet_embed = getattr(self.args, "hypernet_embed", 64)
        
        self.hyper_w_final = nn.Sequential(
            nn.Linear(self.state_dim, hypernet_embed),
            nn.ReLU(inplace=True),
            nn.Linear(hypernet_embed, self.embed_dim)
        )
        self.V = nn.Sequential(
            nn.Linear(self.state_dim, hypernet_embed),
            nn.ReLU(inplace=True),
            nn.Linear(hypernet_embed, 1)
        )

    def forward(self, agent_qs, states, dropout=False):
        bs = agent_qs.size(0)
        
        if dropout:
             states = states.reshape(states.shape[0], states.shape[1], 1, states.shape[2]).repeat(1, 1, self.n_agents, 1)

        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.reshape(-1, 1, self.n_agents)

        us = self._get_us(states, bs) 
        
        w1_heads = []
        for i in range(self.n_heads):
            state_embedding = self.query_embedding_layers[i](states) # [batch, 32]
            state_embedding = state_embedding.unsqueeze(1) # [batch, 1, 32]
            
            u_reshaped = us.reshape(-1, self.n_agents, self.u_dim)
            u_embedding = self.key_embedding_layers[i](u_reshaped) # [batch, n_agents, 32]
            
            # [batch, 1, 32] * [batch, 32, n_agents] -> [batch, 1, n_agents]
            raw_scores = th.matmul(state_embedding, u_embedding.transpose(1, 2)) 
            raw_scores = raw_scores / self.scaled_product_value
            
            attention_w = F.softmax(raw_scores, dim=-1) # [batch, 1, n_agents]
            
            w1_heads.append(attention_w)
            
        w1 = th.cat(w1_heads, dim=1) 
        w1 = w1.transpose(1, 2) # [batch, n_agents, n_heads]
        hidden = th.matmul(agent_qs, w1)
        
        b1 = self.hyper_b_1(states).view(-1, 1, self.embed_dim)
        hidden = hidden + b1
        
        hidden = F.elu(hidden)

        w_final = self.hyper_w_final(states).abs() # 仍然需要 abs 保证单调性
        w_final = w_final.view(-1, self.embed_dim, 1)
        
        v = self.V(states).view(-1, 1, 1)
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
            agent_own_state_size = self.args.agent_own_state_size
            with th.no_grad():
                us = states[:, :agent_own_state_size * self.n_agents]
                us = us.reshape(-1, agent_own_state_size)
            return us