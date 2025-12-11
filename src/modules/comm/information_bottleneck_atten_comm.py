import torch as th
import torch.nn as nn
import torch.nn.functional as F


class IBAttenComm(nn.Module):
    def __init__(self, input_shape, args):
        super(IBAttenComm, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.att_dim = args.atten_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_shape, args.rnn_hidden_dim),
            nn.ReLU(True),
            nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim),
            nn.ReLU(True)
        )

        self.w_q = nn.Linear(args.rnn_hidden_dim, self.att_dim)
        self.w_k = nn.Linear(args.rnn_hidden_dim, self.att_dim)

        self.mu_gen = nn.Sequential(
            nn.Linear(args.rnn_hidden_dim + self.att_dim, args.rnn_hidden_dim),
            nn.ReLU(True),
            nn.Linear(args.rnn_hidden_dim, args.comm_embed_dim)
        )

        self.inference_model = nn.Sequential(
            nn.Linear(input_shape + args.comm_embed_dim * self.n_agents, 4 * args.comm_embed_dim * self.n_agents),
            nn.ReLU(True),
            nn.Linear(4 * args.comm_embed_dim * self.n_agents, 4 * args.comm_embed_dim * self.n_agents),
            nn.ReLU(True),
            nn.Linear(4 * args.comm_embed_dim * self.n_agents, args.n_actions)
        )

        self.action_predictor = nn.Sequential(
            nn.Linear(args.rnn_hidden_dim + self.att_dim, args.rnn_hidden_dim),
            nn.ReLU(True),
            nn.Linear(args.rnn_hidden_dim, args.n_actions),
            nn.LogSoftmax(dim=-1)
        )