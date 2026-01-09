from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
from multiprocessing import Pipe, Process
from collections import defaultdict
import numpy as np
import torch as th

class AsyncRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        self.steps = getattr(args, "runner_steps", 8) 

        # Make subprocesses for the envs
        self.parent_conns, self.worker_conns = zip(*[Pipe() for _ in range(self.batch_size)])
        env_fn = env_REGISTRY[self.args.env]
        self.ps = []
        for i, worker_conn in enumerate(self.worker_conns):
            ps = Process(target=env_worker, 
                    args=(worker_conn, CloudpickleWrapper(partial(env_fn, **self.args.env_args))))
            self.ps.append(ps)

        for p in self.ps:
            p.daemon = True
            p.start()

        self.parent_conns[0].send(("get_env_info", None))
        self.env_info = self.parent_conns[0].recv()
        self.episode_limit = self.env_info["episode_limit"]

        self.t_env = 0 

        self.episode_memory = [defaultdict(list) for _ in range(self.batch_size)]
        self.env_t = [0] * self.batch_size 
        
        # [Log] Persistent running returns for each env
        self.episode_returns = [{"reward": 0, "delta_enemy": 0, "delta_deaths": 0, "delta_ally": 0} for _ in range(self.batch_size)]

        # [Log] Stats accumulators
        self.train_returns = {"reward": [], "delta_enemy": [], "delta_deaths": [], "delta_ally": []}
        self.test_returns = {"reward": [], "delta_enemy": [], "delta_deaths": [], "delta_ally": []}
        self.train_stats = {}
        self.test_stats = {}
        self.log_train_stats_t = -100000

        self.batch = None 
        self.last_actions = None

    def setup(self, scheme, groups, preprocess, mac):
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess
        self.mac = mac
        self.n_agents = self.mac.n_agents
        
        self.batch = EpisodeBatch(scheme, groups, self.batch_size, 2, 
                                  preprocess=preprocess, device=self.args.device)

    def get_env_info(self):
        return self.env_info

    def save_replay(self):
        pass

    def close_env(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(("close", None))

    def reset(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(("reset", None))
        
        self.last_actions = th.zeros((self.batch_size, self.n_agents), dtype=th.long, device=self.args.device)

        pre_transition_data = {
            "state": [], "avail_actions": [], "obs": []
        }
        for parent_conn in self.parent_conns:
            data = parent_conn.recv()
            pre_transition_data["state"].append(data["state"])
            pre_transition_data["avail_actions"].append(data["avail_actions"])
            pre_transition_data["obs"].append(data["obs"])
            
        self.batch.update(pre_transition_data, ts=1)
        
        self.episode_memory = [defaultdict(list) for _ in range(self.batch_size)]
        self.env_t = [0] * self.batch_size
        
        # [Log] Reset running returns
        self.episode_returns = [{"reward": 0, "delta_enemy": 0, "delta_deaths": 0, "delta_ally": 0} for _ in range(self.batch_size)]
        
        self.mac.init_hidden(batch_size=self.batch_size)

    def run(self, test_mode=False):
        if test_mode:
            # AsyncRunner does not ensure full episodes in one run
            raise NotImplementedError("AsyncRunner does not support test_mode. For testing, use EpisodeRunner/ParallelRunner.")

        if self.batch is None or self.t_env == 0:
            self.reset()

        completed_episodes = []
        
        ep_length_sum, n_episodes = 0, 0
        
        for _ in range(self.steps):
            # 1. Construct Input
            self.batch.update({"actions": self.last_actions.unsqueeze(-1)}, ts=0)
            
            actions_onehot = self.batch["actions_onehot"]
            reset_indices = [i for i, t in enumerate(self.env_t) if t == 0]
            if reset_indices:
                actions_onehot[reset_indices, 0, :, :] = 0
            
            # 2. Select Actions
            actions = self.mac.select_actions(self.batch, t_ep=1, t_env=self.t_env, test_mode=test_mode)
            
            cpu_actions = actions.to("cpu").numpy()
            self.last_actions = actions
            
            # 3. Step Envs
            for idx, parent_conn in enumerate(self.parent_conns):
                parent_conn.send(("step", cpu_actions[idx]))

            # 4. Collect Data & Handle Termination
            next_pre_transition_data = {"state": [], "avail_actions": [], "obs": []}
            
            # Receive data from all workers
            worker_data = []
            for parent_conn in self.parent_conns:
                worker_data.append(parent_conn.recv())

            for idx, data in enumerate(worker_data):
                # Process Reward
                reward, delta_enemy, delta_deaths, delta_ally, delta_lists = data["reward"]
                if self.args.reward_decompose:
                    if delta_lists is None:
                        reward_item = tuple([0] * self.args.reward_dim)
                    else:
                        delta_ally_list, delta_enemy_list, death_list = delta_lists
                        terminate_reward = 1 if data["info"].get("battle_won", False) else 0
                        reward_item = (reward, *delta_ally_list, *delta_enemy_list, *death_list, terminate_reward)
                else:
                    reward_item = (reward,)

                # [Log] Accumulate running returns
                self.episode_returns[idx]["reward"] += reward
                self.episode_returns[idx]["delta_enemy"] += delta_enemy
                self.episode_returns[idx]["delta_deaths"] += delta_deaths
                self.episode_returns[idx]["delta_ally"] += delta_ally

                # Store transition in memory (Keep on GPU)
                mem = self.episode_memory[idx]
                
                # Data from t (current step)
                mem["state"].append(self.batch["state"][idx, 1])
                mem["obs"].append(self.batch["obs"][idx, 1])
                mem["avail_actions"].append(self.batch["avail_actions"][idx, 1])
                mem["actions"].append(actions[idx].unsqueeze(-1))
                
                # Data resulting from step (t -> t+1)
                mem["reward"].append(th.tensor(reward_item, dtype=th.float32, device=self.args.device))
                mem["terminated"].append(th.tensor([data["terminated"]], dtype=th.uint8, device=self.args.device))
                mem["battle_won"].append(th.tensor([data["info"].get("battle_won", False)], dtype=th.uint8, device=self.args.device))
                mem["filled"].append(th.tensor([1], dtype=th.long, device=self.args.device))
                
                self.env_t[idx] += 1
                self.t_env += 1
                
                if data["terminated"]:
                    # Episode Finished
                    ep_len = self.env_t[idx]
                    ep_length_sum += ep_len
                    n_episodes += 1
                    
                    # --- Handle Final Step (T) ---
                    # We need to store s_T, obs_T, and a dummy action a_T
                    # The data received from worker contains s_T, obs_T
                    
                    s_T = th.from_numpy(data["state"]).to(self.args.device)
                    obs_T = th.tensor(data["obs"], device=self.args.device)
                    avail_T = th.tensor(data["avail_actions"], device=self.args.device)
                    action_T = th.zeros_like(actions[idx].unsqueeze(-1)) # Dummy action for T
                    
                    mem["state"].append(s_T)
                    mem["obs"].append(obs_T)
                    mem["avail_actions"].append(avail_T)
                    mem["actions"].append(action_T)
                    
                    # [Log] Store completed episode stats
                    cur_returns = self.test_returns if test_mode else self.train_returns
                    cur_stats = self.test_stats if test_mode else self.train_stats
                    
                    cur_returns["reward"].append(self.episode_returns[idx]["reward"])
                    cur_returns["delta_enemy"].append(self.episode_returns[idx]["delta_enemy"])
                    cur_returns["delta_deaths"].append(self.episode_returns[idx]["delta_deaths"])
                    cur_returns["delta_ally"].append(self.episode_returns[idx]["delta_ally"])
                    
                    for k in set(cur_stats) | set(data["info"]):
                        cur_stats[k] = cur_stats.get(k, 0) + data["info"].get(k, 0)
                    cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
                    
                    self.episode_returns[idx] = {"reward": 0, "delta_enemy": 0, "delta_deaths": 0, "delta_ally": 0}

                    # --- Create EpisodeBatch ---
                    # Length is ep_len + 1 to accommodate the final state
                    ep_batch = EpisodeBatch(self.scheme, self.groups, 1, ep_len + 1, 
                                            preprocess=self.preprocess, device=self.args.device)
                    
                    # Split data into two groups based on length
                    # Group A (Length T): reward, terminated, battle_won, filled
                    # Group B (Length T+1): state, obs, avail_actions, actions
                    
                    group_a_keys = ["reward", "terminated", "battle_won", "filled"]
                    group_b_keys = ["state", "obs", "avail_actions", "actions"]
                    
                    data_a = {k: th.stack(mem[k]).unsqueeze(0) for k in group_a_keys}
                    data_b = {k: th.stack(mem[k]).unsqueeze(0) for k in group_b_keys}
                    
                    # Update using slices to handle length mismatch
                    ep_batch.update(data_a, ts=slice(0, ep_len))
                    ep_batch.update(data_b, ts=slice(0, ep_len + 1))
                    
                    completed_episodes.append(ep_batch)
                    
                    # --- Reset for Next Episode ---
                    self.episode_memory[idx] = defaultdict(list)
                    self.env_t[idx] = 0
                    
                    if self.mac.hidden_states is not None:
                        self.mac.hidden_states[idx, :, :].zero_()
                    
                    # Send Reset command to worker
                    self.parent_conns[idx].send(("reset", None))
                    reset_data = self.parent_conns[idx].recv()
                    
                    # Prepare for next step (t=0 of new episode)
                    next_pre_transition_data["state"].append(reset_data["state"])
                    next_pre_transition_data["obs"].append(reset_data["obs"])
                    next_pre_transition_data["avail_actions"].append(reset_data["avail_actions"])
                    
                else:
                    # Episode continues
                    next_pre_transition_data["state"].append(data["state"])
                    next_pre_transition_data["obs"].append(data["obs"])
                    next_pre_transition_data["avail_actions"].append(data["avail_actions"])

            # Update self.batch for next step
            self.batch.update(next_pre_transition_data, ts=1)
            
        # [Log] Logging logic
        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""

        if test_mode and (len(self.test_returns["reward"]) >= self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            cur_stats["ep_length"] = ep_length_sum
            cur_stats["n_episodes"] = max(n_episodes, 1)  # Avoid division by zero
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        # Return completed episodes
        if completed_episodes:
            max_len = max(ep.max_seq_length for ep in completed_episodes)
            ret_batch = EpisodeBatch(self.scheme, self.groups, len(completed_episodes), max_len,
                                     preprocess=self.preprocess, device=self.args.device)
            
            for i, ep in enumerate(completed_episodes):
                for k, v in ep.data.transition_data.items():
                    ret_batch.data.transition_data[k][i, :ep.max_seq_length] = v[0]
                for k, v in ep.data.episode_data.items():
                    ret_batch.data.episode_data[k][i] = v[0]
            
            return ret_batch
        else:
            return None

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "reward_mean", np.mean(returns["reward"]), self.t_env)
        self.logger.log_stat(prefix + "reward_std", np.std(returns["reward"]), self.t_env)
        self.logger.log_stat(prefix + "delta_enemy_mean", np.mean(returns["delta_enemy"]), self.t_env)
        self.logger.log_stat(prefix + "delta_deaths_mean", np.mean(returns["delta_deaths"]), self.t_env)
        self.logger.log_stat(prefix + "delta_ally_mean", np.mean(returns["delta_ally"]), self.t_env)

        for k in returns:
            returns[k].clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()

def env_worker(remote, env_fn):
    # Make environment
    env = env_fn.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            actions = data
            # Take a step in the environment
            reward, terminated, env_info = env.step(actions)
            # Return the observations, avail_actions and state to make the next action
            state = env.get_state()
            avail_actions = env.get_avail_actions()
            obs = env.get_obs()
            remote.send({
                # Data for the next timestep needed to pick an action
                "state": state,
                "avail_actions": avail_actions,
                "obs": obs,
                # Rest of the data for the current timestep
                "reward": reward,
                "terminated": terminated,
                "info": env_info
            })
        elif cmd == "reset":
            env.reset()
            remote.send({
                "state": env.get_state(),
                "avail_actions": env.get_avail_actions(),
                "obs": env.get_obs()
            })
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_env_info":
            remote.send(env.get_env_info())
        elif cmd == "get_stats":
            remote.send(env.get_stats())
        else:
            raise NotImplementedError

class CloudpickleWrapper():
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)