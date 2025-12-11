import torch
import torch.nn as nn
import torch.nn.functional as F

class ICM(nn.Module):
    def __init__(self, scheme, args):
        super(ICM, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = self._get_input_shape(scheme)
        self.action_dim = args.n_actions
        self.hidden_dim = args.icm_feature_dim if hasattr(args, "icm_feature_dim") else 256
        
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU()
        )
        
        # Inverse dynamics model
        self.inverse_model = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim)
        )
        
        # Forward dynamics model
        self.forward_model = nn.Sequential(
            nn.Linear(self.hidden_dim + self.action_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents
        return input_shape

    def _build_inputs(self, batch):
        # return: (t+1, bs*n_agents, input_dim)
        bs = batch.batch_size
        t = batch["obs"].shape[1] - 1
        inputs = []
        inputs.append(batch["obs"])  # (bs, t+1, n_agents, obs_dim)
        if self.args.obs_last_action:
            first_action = torch.zeros_like(batch["actions_onehot"][:, 0]).unsqueeze(1)  # (bs, 1, n_agents, action_dim)
            rem_actions = batch["actions_onehot"][:, :-1]  # (bs, t, n_agents, action_dim)
            inputs.append(torch.cat([first_action, rem_actions], dim=1))  # (bs, t+1, n_agents, action_dim)
        if self.args.obs_agent_id:
            inputs.append(
                torch.eye(self.n_agents, device=batch.device)
                    .reshape(1, 1, self.n_agents, self.n_agents)
                    .expand(bs, t+1, -1, -1)
            )  # (bs, t+1, n_agents, n_agents)

        inputs = torch.cat([x.reshape(bs*(t+1)*self.n_agents, -1) for x in inputs], dim=1)  # (bs*(t+1)*n_agents, input_dim)
        inputs = inputs.reshape(bs, t+1, self.n_agents, -1).permute(1, 0, 2, 3).reshape((t+1)*bs*self.n_agents, -1)  # ((t+1)*bs*n_agents, input_dim)
        return inputs

    def forward(self, ep_batch):
        """
        Compute intrinsic rewards and ICM losses.
        Returns:
            intrinsic_rewards: (bs, t, n_agents)
            inverse_loss: scalar
            forward_loss: scalar
        """
        bs = ep_batch.batch_size
        t = ep_batch["obs"].shape[1] - 1
        obs = self._build_inputs(ep_batch)  # ((t+1)*bs*n_agents, input_dim)
        actions = ep_batch["actions"][:, :-1].squeeze(3).permute(1, 0, 2)  # (t, bs, n_agents)
        features = self.encoder(obs).reshape(t+1, bs*self.n_agents, self.hidden_dim)  # (t+1, bs*n_agents, hidden_dim)

        # inverse model loss
        predicted_actions = self.inverse_model(
            torch.cat([features[:-1], features[1:]], dim=2)  # (t, bs*n_agents, hidden_dim*2)
                 .reshape(t*bs*self.n_agents, -1)  # (t*bs*n_agents, hidden_dim*2)
        )  # (t*bs*n_agents, action_dim)
        inverse_loss = F.cross_entropy(predicted_actions, actions.reshape(t*bs*self.n_agents))

        # forward model loss
        actions_onehot = F.one_hot(actions, num_classes=self.action_dim).float()  # (t, bs, n_agents, action_dim)
        predicted_next_features = self.forward_model(
            torch.cat([features[:-1], actions_onehot.reshape(t, bs*self.n_agents, -1)], dim=2)  # (t, bs*n_agents, hidden_dim + action_dim)
                 .reshape(t*bs*self.n_agents, -1)  # (t*bs*n_agents, hidden_dim + action_dim)
        )  # (t*bs*n_agents, hidden_dim)
        forward_loss = F.mse_loss(
            predicted_next_features,
            features[1:].reshape(t*bs*self.n_agents, -1),
            reduction='none'
        ).mean(dim=1)  # (t*bs*n_agents)

        intrinsic_rewards = forward_loss.detach().reshape(t, bs, self.n_agents).permute(1, 0, 2)  # (bs, t, n_agents)

        return intrinsic_rewards, inverse_loss, forward_loss.mean()
