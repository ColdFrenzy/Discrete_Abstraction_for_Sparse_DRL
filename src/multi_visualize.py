import os
import numpy as np
from src.environments.multi_agent.ma_cont_uav_env import MultiAgentContinuousUAV
import torch
from src.environments.multi_agent.ma_cont_uav_env_sb3 import MultiAgentContinuousUAVSb3HerWrapper
from src.utils.paths import ROOT_DIR
from src.definitions import RewardType, TransitionMode
from src.environments.maps import MAPS_FREE
from src.environments.maps import MAPS_OBST
from stable_baselines3 import HerReplayBuffer, SAC
from src.environments.single_agent.cont_uav_env_sb3 import ContinuousUAVSb3HerWrapper
from src.utils.heatmap import generate_heatmaps_numbers
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
from scipy.interpolate import interp1d

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_agent_pos(idx: int) -> float:
    """
    idx goes from 0 to map_size (exclusive)
    """
    return ((idx + 1) + idx) * 0.5

def multi_tensordboard_plot(tensorboard_path: str, save_path: str):
    # Initialize figure for plotting
    plt.figure(figsize=(10, 6))
    for algorithm_name, log_dir in tensorboard_path.items():
        
        all_rewards = []
        all_steps = []
        agent_dirs = [os.path.join(log_dir, agent) for agent in os.listdir(log_dir)]
        env_dirs = [os.path.join(agent_dir, run) for agent_dir in agent_dirs for run in os.listdir(agent_dir) ]
        run_dirs = [os.path.join(env_dir, run) for env_dir in env_dirs for run in os.listdir(env_dir) ]
        
        # Read the event files from each run directory
        for run in run_dirs:
            event_acc = event_accumulator.EventAccumulator(run)
            event_acc.Reload()

            # Extract step and reward values
            steps = [event.step for event in event_acc.Scalars('custom/win_rate')]
            reward_values = [event.value for event in event_acc.Scalars('custom/win_rate')]
            
            # steps = [event.step for event in event_acc.Scalars('rollout/ep_len_mean')]
            # reward_values = [event.value for event in event_acc.Scalars('rollout/ep_len_mean')]
            
            # Store rewards of each run
            all_steps.append(np.array(steps))
            all_rewards.append(np.array(reward_values))

        # Find the unique set of steps across all runs
        all_unique_steps = np.unique(np.concatenate(all_steps))

        aligned_rewards = []
        for steps, rewards in zip(all_steps, all_rewards):
            # Create an interpolation function based on the current run's steps
            interp_func = interp1d(steps, rewards, kind='linear', bounds_error=False, fill_value="extrapolate")
            
            # Interpolate the rewards to match the unique steps
            new_rewards = interp_func(all_unique_steps)
            
            aligned_rewards.append(new_rewards)
        # Convert the list of rewards to a NumPy array
        aligned_rewards = np.array(aligned_rewards)

        # Compute the mean and std across all runs (along axis=0 corresponds to different seeds/runs)
        mean_rewards = np.mean(aligned_rewards, axis=0)
        std_rewards = np.std(aligned_rewards, axis=0)
    
        plt.plot(all_unique_steps, mean_rewards, label=f"{algorithm_name}", lw=2)
        plt.fill_between(all_unique_steps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2, )
    
    plt.xlabel('Steps')
    plt.ylabel('Win Rate')
    plt.legend()
    plt.show()

def main(alg="SAC_HR", map_size=10, seed=13):
    model_class = SAC  # works also with SAC, DDPG and TD3
    os.environ["IS_RENDER"] = "True"
    if alg == "SAC_HR":
        env_reward_type = RewardType.model
    elif alg == "SAC_DENSE":
        env_reward_type = RewardType.dense
    else:
        env_reward_type = RewardType.sparse

    is_slippery = False
    map_size = map_size
    MAX_EPISODE_STEPS = 400          
    OBST = True
    if OBST:
        map_name = MAPS_OBST[map_size]
    else:
        map_name = MAPS_FREE[map_size]
       
    tb_paths = {
        "SAC_RH": f"tensorboards/multiagent_tensorboard/hr",
        "SAC_HER": f"tensorboards/multiagent_tensorboard/her",
        "SAC": f"tensorboards/multiagent_tensorboard/sac",
        "SAC_RELAX": f"tensorboards/multiagent_tensorboard/dense",
    }
    
    num_agent = 1
    tb_paths = {
        "SAC_RH": f"tensorboards/agent_{num_agent}_tensorboard/hr",
        "SAC_HER": f"tensorboards/agent_{num_agent}_tensorboard/her",
        "SAC": f"tensorboards/agent_{num_agent}_tensorboard/sac",
        "SAC_RELAX": f"tensorboards/agent_{num_agent}_tensorboard/dense",
    }
    
    # tb_paths = {
    #     "0.1": f"tensorboards/gammas_tensorboard/0.1",
    #     "0.2": f"tensorboards/gammas_tensorboard/0.2",
    #     "0.3": f"tensorboards/gammas_tensorboard/0.3",
    #     "0.5": f"tensorboards/gammas_tensorboard/0.5",
    #     "0.8": f"tensorboards/gammas_tensorboard/0.8",
    #     "0.99": f"tensorboards/gammas_tensorboard/0.99",
    # }
    
    multi_tensordboard_plot(tensorboard_path=tb_paths, save_path = ROOT_DIR / "plots")

    a1_initial_pos = [get_agent_pos(5), get_agent_pos(5)]
    env1 = ContinuousUAVSb3HerWrapper(map_name =  map_name, agent_name="a1", size = map_size, max_episode_steps=MAX_EPISODE_STEPS, OBST=OBST, agent_initial_pos = a1_initial_pos, reward_type = env_reward_type, is_rendered = True, is_slippery = is_slippery, is_display=False, seed=seed)
    a3_initial_pos = [get_agent_pos(0), get_agent_pos(9)]
    env3 = ContinuousUAVSb3HerWrapper(map_name =  map_name, agent_name="a3", size = map_size, max_episode_steps=MAX_EPISODE_STEPS, OBST=OBST, agent_initial_pos = a3_initial_pos, reward_type = env_reward_type, is_rendered = True, is_slippery = is_slippery, is_display=False, seed=seed)
    
    save_path1 = f"./models/a1/{alg}/{map_size}x{map_size}_{seed}_0.1"
    agent1 = model_class.load(save_path1, env=env1)
    save_path2 = f"./models/a3/{alg}/{map_size}x{map_size}_{seed}_0.1"
    agent2 = model_class.load(save_path2, env=env3)
    
    # multiagent_env = MultiAgentContinuousUAVSb3HerWrapper(map_name =  map_name, num_agents = 2, agents_pos= [a1_initial_pos, a3_initial_pos], size = map_size, OBST=OBST, reward_type = env_reward_type, is_rendered = True, is_slippery = is_slippery, is_display=False)
    # multiagent_env.render_episode(agents=[agent1, agent2], max_steps=100)
    # multiagent_env.save_episode(100, name="multi_uav_cont")

if __name__ == "__main__":
    main()