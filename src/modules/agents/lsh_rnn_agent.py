import torch
import torch.nn as nn
import torch.nn.functional as F

class LSHRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(LSHRNNAgent, self).__init__()
        self.args = args

        # === 改进 1: 更深的特征提取器 (MLP) ===
        # 原来只有一层 fc1，可以加深
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim) # 新增一层
        
        # === 改进 2: Layer Normalization ===
        self.layer_norm = nn.LayerNorm(args.rnn_hidden_dim)

        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        
        self.fc3 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        # === 改进 3: 参数初始化 ===
        # RNN 对初始化敏感，正交初始化通常表现更好
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.GRUCell):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state=None):
        b, a, e = inputs.size()
        
        x = F.relu(self.fc1(inputs.view(-1, e)), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        
        x = self.layer_norm(x)

        if hidden_state is not None:
            hidden_state = hidden_state.reshape(-1, self.args.rnn_hidden_dim)

        h = self.rnn(x, hidden_state)
        
        logits = self.fc3(h)
        
        # === 关键修正: 移除了 torch.clamp ===
        
        return logits.view(b, a, -1), h.view(b, a, -1)