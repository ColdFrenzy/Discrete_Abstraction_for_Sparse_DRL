import os
import torch
from src.definitions import RewardType, TransitionMode
from src.environments.maps import MAPS_FREE
from src.environments.maps import MAPS_OBST
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import HerReplayBuffer, DDPG, DQN, SAC, TD3
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from src.environments.single_agent.cont_uav_env_sb3 import ContinuousUAVSb3HerWrapper


def main(alg="SAC", map_size=5, seed=13):
    model_class = SAC  # works also with SAC, DDPG and TD3
    os.environ["IS_RENDER"] = "False"
    env_reward_type = RewardType.sparse # or model, or sparse
    is_slippery = False
    map_size = map_size
    MAX_EPISODE_STEPS = 400
    OBST = True
    if OBST:
        map_name = MAPS_OBST[map_size]
    else:
        map_name = MAPS_FREE[map_size]


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
            tensorboard_log=f"./her_sac_uav_tensorboard/{map_size}x{map_size}_{seed}",
            verbose=2,
            device=device,
        )
    elif alg == "SAC":
        model = model_class(
            "MultiInputPolicy",
            env,
            learning_starts=1e4,
            tensorboard_log=f"./sac_uav_tensorboard/{map_size}x{map_size}_{seed}",
            verbose=2,
            device=device,
        )


    # Train the model
    print("start learning")
    model.learn(100000)
    print("learning done")
    save_path = f"./models/{alg}_{map_size}x{map_size}_{seed}"
    model.save(save_path)
    # Because it needs access to `env.compute_reward()`
    # HER must be loaded with the env
    model = model_class.load("./her_uav_env", env=env)

    print("start evaluating")
    obs, info = env.reset()
    for _ in range(100):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()

    print("evaluation done")


if __name__ == "__main__":
    algos = ["SAC", "SACHER"]
    maps = [3, 5, 10]
    seeds = [13, 42, 69]

    for alg in algos:
        for map_size in maps:
            for seed in seeds:
                print(f"Running {alg} on map size {map_size} with seed {seed}")
                main(alg=alg, map_size=map_size, seed=seed)