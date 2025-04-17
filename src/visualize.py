import os
import numpy as np
import torch
from src.utils.paths import QTABLE_DIR
from src.definitions import RewardType, TransitionMode
from src.environments.maps import MAPS_FREE
from src.environments.maps import MAPS_OBST
from stable_baselines3 import HerReplayBuffer, SAC
from src.environments.single_agent.cont_uav_env_sb3 import ContinuousUAVSb3HerWrapper
from src.utils.heatmap import generate_heatmaps_numbers


def main(alg="SAC", map_size=3, seed=13, agent="a3"):
    model_class = SAC  # works also with SAC, DDPG and TD3
    os.environ["IS_RENDER"] = "False"
    if alg == "SAC_HR":
        env_reward_type = RewardType.model
    else:
        env_reward_type = RewardType.sparse # or model, or sparse

    is_slippery = False
    map_size = map_size
    MAX_EPISODE_STEPS = 400
    OBST = False
    if OBST:
        map_name = MAPS_OBST[map_size]
    else:
        map_name = MAPS_FREE[map_size]

    # if alg == "SAC_HR":
    #     transition_mode = TransitionMode.stochastic if is_slippery else TransitionMode.deterministic

    #     qtable = np.load(f"{QTABLE_DIR}/{transition_mode.name}/single_agent/qtable_{map_size}_obstacles_{OBST}.npz")
    #     generate_heatmaps_numbers(qtable)

    DESIRED_ORIENTATIONS = {"a1": [44.,46.], "a2": [134.,136.], "a3": [269., 271.]} 
    DESIRED_DISTANCES = {"a1": [0.9, 1.1], "a2": [0.9, 1.1], "a3": [0.9, 1.1]} 
    env = ContinuousUAVSb3HerWrapper(map_name =  map_name, agent_name=agent, size = map_size, max_episode_steps=MAX_EPISODE_STEPS, OBST=OBST, agent_initial_pos = [0.5, map_size-0.5], task="encircle_target", desired_orientation= DESIRED_ORIENTATIONS[agent], desired_distance = DESIRED_DISTANCES[agent], reward_type = env_reward_type, is_rendered = True, is_slippery = is_slippery, is_display=False, seed=seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model_class(
            "MultiInputPolicy",
            env,
            learning_starts=1e4,
            tensorboard_log=f"./sac_uav_tensorboard/{map_size}x{map_size}_{seed}",
            verbose=2,
            device=device,
        )
    save_path = f"./models/{agent}/{alg}/{map_size}x{map_size}_{seed}_0.9"
    model = model_class.load(save_path, env=env)

    env.render_episode(model)
    env.save_episode(1, f"{alg}_{map_size}x{map_size}_{seed}")


if __name__ == "__main__":
    main()