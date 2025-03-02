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

def get_agent_pos(idx: int) -> float:
    """
    idx goes from 0 to map_size (exclusive)
    """
    return ((idx + 1) + idx) * 0.5

def main(alg="SAC_HR", map_size=10, seed=42):
    model_class = SAC  # works also with SAC, DDPG and TD3
    os.environ["IS_RENDER"] = "True"
    if alg == "SAC_HR":
        env_reward_type = RewardType.model
    else:
        env_reward_type = RewardType.sparse # or model, or sparse

    is_slippery = False
    map_size = map_size
    MAX_EPISODE_STEPS = 400
    OBST = True
    if OBST:
        map_name = MAPS_OBST[map_size]
    else:
        map_name = MAPS_FREE[map_size]

    a1_initial_pos = [get_agent_pos(5), get_agent_pos(5)]
    env = ContinuousUAVSb3HerWrapper(map_name =  map_name, agent_name="a1", size = map_size, max_episode_steps=MAX_EPISODE_STEPS, OBST=OBST, agent_initial_pos = a1_initial_pos, reward_type = env_reward_type, is_rendered = True, is_slippery = is_slippery, is_display=False, seed=seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model_class(
            "MultiInputPolicy",
            env,
            learning_starts=1e4,
            tensorboard_log=f"./sac_uav_tensorboard/{map_size}x{map_size}_{seed}",
            verbose=2,
            device=device,
        )
    save_path = f"./models/a1/{alg}/{map_size}x{map_size}_{seed}_0.2"
    model = model_class.load(save_path, env=env)

    env.render_episode(model)
    env.save_episode(1, f"{alg}_{map_size}x{map_size}_{seed}")


if __name__ == "__main__":
    main()