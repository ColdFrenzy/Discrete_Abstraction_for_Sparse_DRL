import os
import numpy as np
import torch
from src.utils.paths import QTABLE_DIR
from src.definitions import RewardType, TransitionMode
from src.environments.maps import MAPS_FREE
from src.environments.maps import MAPS_OBST
from src.value_function_computation import compute_value_function_single, compute_value_function
from src.utils.heatmap import generate_heatmaps_numbers
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import HerReplayBuffer, DDPG, DQN, SAC, TD3
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from src.environments.single_agent.cont_uav_env_sb3 import ContinuousUAVSb3HerWrapper
from src.utils.evaluation_metrics import WinRateCallback


def main(alg="SAC", map_size=5, seed=13):
    model_class = SAC  # works also with SAC, DDPG and TD3
    os.environ["IS_RENDER"] = "False"
    if alg == "SAC_HR":
        env_reward_type = RewardType.model
    else:
        env_reward_type = RewardType.sparse # or model, or sparse


    custom_callback = WinRateCallback()
    is_slippery = False
    map_size = map_size
    MAX_EPISODE_STEPS = 400
    NUM_EPISODES_DISCRETE = 50000 # 20000
    OBST = True
    if OBST:
        map_name = MAPS_OBST[map_size]
    else:
        map_name = MAPS_FREE[map_size]

    if alg == "SAC_HR":
        transition_mode = TransitionMode.stochastic if is_slippery else TransitionMode.deterministic
        experiment_name = f"sac-{env_reward_type.name}-{map_size}-obstacles_{OBST}"
        compute_value_function_single(map_name, size=map_size, OBST=OBST, num_episodes=NUM_EPISODES_DISCRETE, gamma = 0.8, stochastic=is_slippery, save=True)
    
        qtable = np.load(f"{QTABLE_DIR}/{transition_mode.name}/single_agent/qtable_{map_size}_obstacles_{OBST}.npz")
        generate_heatmaps_numbers(qtable)
    env = ContinuousUAVSb3HerWrapper(map_name =  map_name, agent_name="a1", size = map_size, max_episode_steps=MAX_EPISODE_STEPS, OBST=OBST, agent_initial_pos = [0.5, map_size-0.5], reward_type = env_reward_type, is_rendered = True, is_slippery = is_slippery, is_display=False, seed=seed)
    # Available strategies (cf paper): future, final, episode


    # test if the environment follows the gym interface
    # check_env(env)

    goal_selection_strategy = "future" # equivalent to GoalSelectionStrategy.FUTURE
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Initialize the model
    if alg == "SACHER":
        model= model_class(
            "MultiInputPolicy",
            env,
            replay_buffer_class=HerReplayBuffer, # Comment this line to use the default replay buffer
            learning_starts=1e4,
            # Parameters for HER
            replay_buffer_kwargs=dict(
                n_sampled_goal=4,
                goal_selection_strategy=goal_selection_strategy,
            ),
            tensorboard_log=f"./sa_her_sac_uav_tensorboard/{map_size}x{map_size}_{seed}",
            verbose=2,
            device=device,
        )
    elif alg == "SAC":
        model = model_class(
            "MultiInputPolicy",
            env,
            learning_starts=1e4,
            tensorboard_log=f"./sa_sac_uav_tensorboard/{map_size}x{map_size}_{seed}",
            verbose=2,
            device=device,
        )
    elif alg == "SAC_HR":
        model = model_class(
            "MultiInputPolicy",
            env,
            learning_starts=1e4,
            tensorboard_log=f"./sa_sac_hr_uav_tensorboard/{map_size}x{map_size}_{seed}",
            verbose=2,
            device=device,
            gamma=0.1,
        )

    # Train the model
    print("start learning")
    model.learn(100000, callback=custom_callback)
    print("learning done")
    save_path = f"./models/{alg}_{map_size}x{map_size}_{seed}"
    model.save(save_path)
    # Because it needs access to `env.compute_reward()`
    # HER must be loaded with the env
    model = model_class.load(save_path, env=env)


if __name__ == "__main__":
    algos = ["SAC", "SACHER"]
    maps = [10]
    seeds = [13, 42, 69]

    for alg in algos:
        for map_size in maps:
            for seed in seeds:
                print(f"Running {alg} on map size {map_size} with seed {seed}")
                main(alg=alg, map_size=map_size, seed=seed)