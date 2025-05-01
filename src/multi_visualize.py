import os
import numpy as np
from src.environments.multi_agent.ma_cont_uav_env import MultiAgentContinuousUAV
import torch
from src.utils.paths import ROOT_DIR
from src.definitions import RewardType, TransitionMode
from src.environments.maps import MAPS_FREE
from src.environments.maps import MAPS_OBST
from stable_baselines3 import HerReplayBuffer, SAC
from src.environments.single_agent.cont_uav_env_sb3 import ContinuousUAVSb3HerWrapper
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
from scipy.interpolate import interp1d

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_agent_pos(idx: int, map_size: int) -> float:
    """
    idx goes from 0 to map_size (exclusive)
    """
    if idx >= map_size:
        raise ValueError(f"idx must be less than {map_size}")
    return (2*idx + 1) * 0.5

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

def main(alg="SAC", map_size=10, seed=13):
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
    OBST = False
    if OBST:
        map_name = MAPS_OBST[map_size]
    else:
        map_name = MAPS_FREE[map_size]
    
    DESIRED_ORIENTATIONS = [[44.,46.], [134.,136.], [269., 271.]]
    DESIRED_DISTANCES = [[0.9, 1.1],[0.9, 1.1],[0.9, 1.1]] # circular crown around the target
    task = "reach_target"

    agents_pos = {"a1":  [0.5, 0.5],
                   "a2":  [0.5, 0.5],
                   "a3":  [0.5, 0.5]}
    
    agents_model = {}
    for agent in agents_pos:
        save_path = f"./models/{agent}/{alg}/{map_size}x{map_size}_{seed}_0.99"
        agents_model[agent] = model_class.load(save_path) #, env=env1)

    
    multiagent_env = MultiAgentContinuousUAV(
        map =  map_name,
        agents_pos= agents_pos,
        OBST=OBST,
        reward_type = env_reward_type,
        max_episode_steps=MAX_EPISODE_STEPS,
        task = task,
        desired_orientations=DESIRED_ORIENTATIONS,
        desired_distances=DESIRED_DISTANCES,
        is_rendered = True,
        is_slippery = is_slippery,
        is_display=False
        )
    multiagent_env.render_episode_goal_mdp(agents_model=agents_model, max_steps=100)
    multiagent_env.save_episode(1, name="multi_uav_cont")

if __name__ == "__main__":
    main()