import datetime
import os
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer, DualReplayBuffer, PrioritizedReplayBuffer
from components.transforms import OneHot

from smac.env import StarCraft2Env

def get_agent_own_state_size(env_args):
    sc_env = StarCraft2Env(**env_args)
    # qatten parameter setting (only use in qatten)
    return  4 + sc_env.shield_bits_ally + sc_env.unit_type_bits

def run(_run, _config, _log):

    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"
    args.multi_reward = getattr(args, 'multi_reward', False)  # lhc

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(dirname(abspath(__file__)))), "results", "tb_logs",args.env_args['map_name'])
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_func = run_async if getattr(args, "run_async", False) else run_sequential
    run_func(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


def evaluate_sequential(args, runner):

    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()

def run_sequential(args, logger):

    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_enemies = env_info["n_enemies"] # lsh modified, ensured key "n_enemies" exists
    args.agent_feature_names = env_info["agent_features"] # lsh modified
    args.enemy_feature_names = env_info["enemy_features"] # lsh modified
    args.unit_dim_tuple = env_info["unit_dim"] # lsh modified
    args.reward_dim = (args.n_agents + args.n_enemies) * 2 + 2 if getattr(args, 'reward_decompose', False) else 1  # lhc
    args.mispred_rewards = getattr(args, 'mispred_rewards', False)  # lhc

    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.accumulated_episodes = getattr(args, "accumulated_episodes", None)
    args.reward_decompose = getattr(args, "reward_decompose", False)  # lhc

    if getattr(args, 'agent_own_state_size', False):
        args.agent_own_state_size = get_agent_own_state_size(args.env_args)

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "probs": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.float},
        "reward": {"vshape": (args.reward_dim,)},  # lhc
        "terminated": {"vshape": (1,), "dtype": th.uint8},
        "battle_won": {"vshape": (1,), "dtype": th.uint8}, # lhc modified
    }
    groups = {
        "agents": args.n_agents,
        "enemies": args.n_enemies  # lsh modified, don't know what use
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    buffer_type = {
        "replay_buffer": ReplayBuffer,
        "prioritized_replay_buffer": PrioritizedReplayBuffer,
        "dual_replay_buffer": DualReplayBuffer
    }[args.buffer_type if hasattr(args, 'buffer_type') else 'replay_buffer']
    buffer = buffer_type(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)
    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # policy_mac = mac_REGISTRY[args.policy_mac](buffer.scheme, groups, args)

    # Learner
    set_max_reward(args)
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()

    if args.checkpoint_path != "":

        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

        if args.evaluate or args.save_replay:
            evaluate_sequential(args, runner)
            return

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0
    train_counter = 0

    steps_per_train = getattr(args, "steps_per_train", 100) 
    flow_control_thres = getattr(args, "flow_control_thres", 10000) # Allow some buffer

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    while runner.t_env <= args.t_max:

        # Run for a whole episode at a time

        t_run_start = time.time()
        if runner.t_env <= train_counter * steps_per_train + flow_control_thres or train_counter == 0:
            with th.no_grad():
                episode_batch = runner.run(test_mode=False)
                buffer.insert_episode_batch(episode_batch)
        t_run_end = time.time()

        t_train_start = time.time()
        if buffer.can_sample(args.batch_size):
            next_episode = episode + args.batch_size_run
            if args.accumulated_episodes and next_episode % args.accumulated_episodes != 0:
                continue

            # [修改建议] 根据 batch_size_run 的大小循环多次训练
            # 这样可以保证每收集 1 个回合的数据量，平均还是对应 1 次训练
            train_steps = min(args.batch_size_run, (runner.t_env + flow_control_thres) // steps_per_train - train_counter)
            for _ in range(train_steps): 
                episode_sample = buffer.sample(args.batch_size)

                # Truncate batch to only filled timesteps
                max_ep_t = episode_sample.max_t_filled()
                episode_sample = episode_sample[:, :max_ep_t]

                if episode_sample.device != args.device:
                    episode_sample.to(args.device)

                learner.train(episode_sample, runner.t_env, episode)
                train_counter += 1
                del episode_sample
        t_train_end = time.time()

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()

            last_test_T = runner.t_env
            for _ in range(n_test_runs):
                runner.run(test_mode=True)

        if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            model_save_time = runner.t_env
            save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
            #"results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.log_stat("time_runner", t_run_end - t_run_start, runner.t_env)
            logger.log_stat("time_learner", t_train_end - t_train_start, runner.t_env)
            logger.log_stat("train_counter", train_counter, runner.t_env)
            if isinstance(buffer, DualReplayBuffer):
                logger.log_stat("win_rate", buffer.win_rate(), runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    return config


import copy

def run_async(args, logger):
    # --- 1. Init Training Runner (Will be the new N-step runner later) ---
    # For now, we use the registry, assuming args.runner points to the new runner eventually
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_enemies = env_info["n_enemies"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.unit_dim_tuple = env_info["unit_dim"]
    args.agent_feature_names = env_info["agent_features"]
    args.enemy_feature_names = env_info["enemy_features"]
    args.reward_decompose = getattr(args, "reward_decompose", False)
    args.reward_dim = (args.n_agents + args.n_enemies) * 2 + 2 if args.reward_decompose else 1
    args.mispred_rewards = getattr(args, 'mispred_rewards', False)
    args.accumulated_episodes = getattr(args, "accumulated_episodes", None)

    if getattr(args, 'agent_own_state_size', False):
        args.agent_own_state_size = get_agent_own_state_size(args.env_args)

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "probs": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.float},
        "reward": {"vshape": (args.reward_dim,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
        "battle_won": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {
        "agents": args.n_agents,
        "enemies": args.n_enemies
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    buffer_type = {
        "replay_buffer": ReplayBuffer,
        "prioritized_replay_buffer": PrioritizedReplayBuffer,
        "dual_replay_buffer": DualReplayBuffer
    }[args.buffer_type if hasattr(args, 'buffer_type') else 'replay_buffer']
    
    buffer = buffer_type(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)
    
    # --- 2. Setup MACs ---
    # Main MAC for Learner
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)
    
    # Runner MAC (Actor) - Deepcopy for safety
    runner_mac = copy.deepcopy(mac)

    # Setup Training Runner
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=runner_mac)

    # --- 3. Init Test Runner (Dedicated for evaluation) ---
    # Must be EpisodeRunner to run full episodes for evaluation
    test_runner_args = copy.deepcopy(args)
    test_runner_args.batch_size_run = 1
    test_runner = r_REGISTRY["parallel"](args=test_runner_args, logger=logger)
    # Test runner uses the MAIN mac (shared with learner), because:
    # 1. It runs in the learner thread (sequentially), so no race condition.
    # 2. It should evaluate the latest model.
    test_runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=runner_mac)

    # Learner
    set_max_reward(args)
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    
    # Shared variables
    terminate_event = threading.Event()
    episode_counter = 0
    train_counter = 0

    temp_mac = copy.deepcopy(mac)  # For updating runner_mac
    mac_updated_flag = threading.Event()

    # Locks
    buffer_lock = threading.Lock()
    temp_mac_lock = threading.Lock()  # Avoid reading and writing temp_mac simultaneously

    if args.use_cuda:
        learner.cuda()
        runner_mac.cuda()
        temp_mac.cuda()  # For faster copying

    steps_per_train = getattr(args, "steps_per_train", 100) 
    flow_control_thres = getattr(args, "flow_control_thres", 10000) # Allow some buffer

    # --- Runner Thread Function ---
    def runner_worker():
        nonlocal episode_counter
        last_log_T = 0
        last_test_T = -args.test_interval - 1
        total_wait_time = 0
        
        while not terminate_event.is_set() and runner.t_env <= args.t_max:
            if train_counter > 0 and runner.t_env > steps_per_train * train_counter + flow_control_thres:
                # train_counter == 0 means learner is waiting for initial data
                time.sleep(0.05)  # wait for learner to catch up
                total_wait_time += 0.05
                continue

            # 1. Update runner_mac if there is an update
            if mac_updated_flag.is_set():
                with temp_mac_lock:
                    runner_mac.load_state(temp_mac)
                    mac_updated_flag.clear()
    
            # 2. Run Environment
            with th.no_grad():
                episode_batch = runner.run(test_mode=False)
            
            # 3. Insert into Buffer
            with buffer_lock:
                buffer.insert_episode_batch(episode_batch)
            
            # Update counters
            episode_counter += args.batch_size_run

            # 4. Periodic Testing (Using test_runner)
            if (runner.t_env - last_test_T) / args.test_interval >= 1.0:
                logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
                last_test_T = runner.t_env
                test_runner.t_env = runner.t_env  # Sync t_env

                mac_hidden_states = runner_mac.hidden_states  # Save runner_mac hidden states
                
                # Run evaluation
                n_test_runs = max(1, args.test_nepisode // test_runner.batch_size)
                with th.no_grad():
                    for _ in range(n_test_runs):
                        test_runner.run(test_mode=True)
                
                runner_mac.hidden_states = mac_hidden_states  # Restore runner_mac hidden states
            
            # Logging (Runner side)
            if (runner.t_env - last_log_T) >= args.log_interval:
                logger.log_stat("episode", episode_counter, runner.t_env)
                logger.log_stat("train_counter", train_counter, runner.t_env)
                logger.log_stat("runner_wait_time", total_wait_time, runner.t_env)
                if isinstance(buffer, DualReplayBuffer):
                    logger.log_stat("win_rate", buffer.win_rate(), runner.t_env)
                logger.print_recent_stats()
                last_log_T = runner.t_env

    # --- Learner Thread Function ---
    def learner_worker():
        nonlocal train_counter
        model_save_time = 0
        
        # Learner loop
        while not terminate_event.is_set() and runner.t_env <= args.t_max:
            if train_counter * steps_per_train > runner.t_env + flow_control_thres:
                time.sleep(0.05)  # wait for runner to catch up
                continue
            
            # 1. Check if we can sample
            can_sample = False
            with buffer_lock:
                if buffer.can_sample(args.batch_size):
                    can_sample = True
            
            if not can_sample:
                time.sleep(0.05) # Short sleep to prevent busy waiting
                continue
            
            # 2. Sample and Train
            for _ in range(args.batch_size_run):
                with buffer_lock:
                    episode_sample = buffer.sample(args.batch_size)
                
                max_ep_t = episode_sample.max_t_filled()
                episode_sample = episode_sample[:, :max_ep_t]

                if episode_sample.device != args.device:
                    episode_sample.to(args.device)

                # Learner updates 'mac' here. 
                # Since test_runner uses 'mac' and runs in THIS thread, it's safe.
                # runner_mac uses 'mac' via copy under lock, also safe.
                learner.train(episode_sample, runner.t_env, episode_counter)
                train_counter += 1
                del episode_sample

            # 3. Save Model
            if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
                model_save_time = runner.t_env
                save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
                os.makedirs(save_path, exist_ok=True)
                logger.console_logger.info("Saving models to {}".format(save_path))
                learner.save_models(save_path)
            
            # 4. Update Runner MAC
            with temp_mac_lock:
                temp_mac.load_state(mac)
                mac_updated_flag.set()

    # Start Threads
    runner_thread = threading.Thread(target=runner_worker, name="RunnerThread")
    learner_thread = threading.Thread(target=learner_worker, name="LearnerThread")
    
    runner_thread.start()
    learner_thread.start()
    
    runner_thread.join()
    terminate_event.set()
    learner_thread.join()

    runner.close_env()
    test_runner.close_env() # Don't forget to close test env
    logger.console_logger.info("Finished Training")

def set_max_reward(args):
    from envs import REGISTRY as env_REGISTRY
    temp_env = env_REGISTRY[args.env](**args.env_args)
    temp_env.reset()
    # env 创建时会根据 n_enemies, reward_death_value, reward_win 来计算 max_reward
    # reset() 调用 init_units(), init_units() 第一次调用时更新 max_reward
    # 为了获取环境的 max_reward，我们需要调用 reset() 一次
    max_reward = temp_env.max_reward
    temp_env.close()
    args.max_reward = max_reward