import warnings

warnings.filterwarnings("ignore")

import os
import gymnasium as gym
import wandb
import numpy as np

from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.env_util import make_vec_env
from src.environments.single_agent.cont_uav_env import ContinuousUAV
from src.environments.multi_agent.ma_cont_uav_env import MultiAgentContinuousUAV
from src.value_function_computation import compute_value_function
from src.definitions import RewardType, TransitionMode
from src.environments.maps import MAPS_FREE
from src.environments.maps import MAPS_OBST
from src.utils.heatmap import generate_heatmaps_numbers
from src.utils.paths import QTABLE_DIR

os.environ["IS_RENDER"] = "False"
env_reward_type = RewardType.model  # or model, or sparse
use_her = False
is_slippery = False
map_size = 8
cell_size = 0.5
MAX_EPISODE_STEPS = 400
OBST = True
if OBST:
    map_name = MAPS_OBST[map_size]
else:
    map_name = MAPS_FREE[map_size]

compute_qtable = False
training = True
NUM_EPISODES_DISCRETE = 30_000
NUM_EPISODES_CONT = 50_000
transition_mode = (
    TransitionMode.stochastic if is_slippery else TransitionMode.deterministic
)
experiment_name = f"sac-{env_reward_type.name}-{map_size}-obstacles_{OBST}"


def get_agent_pos(idx: int) -> float:
    """
    idx goes from 0 to map_size (exclusive)
    """
    return ((idx + 1) + idx) * 0.5

if compute_qtable:
    compute_value_function(
        map_name,
        size=map_size,
        OBST=OBST,
        num_episodes=NUM_EPISODES_DISCRETE,
        gamma=0.8,
        stochastic=is_slippery,
        save=True,
    )

if env_reward_type == RewardType.model:
    qtable = np.load(
        f"{QTABLE_DIR}/{transition_mode.name}/qtable_{map_size}_obstacles_{OBST}.npz"
    )
    generate_heatmaps_numbers(qtable)

env1 = ContinuousUAV(
    map_name=map_name,
    agent_name="a1",
    size=map_size,
    agent_initial_pos = tuple([get_agent_pos(5), get_agent_pos(0)]),
    max_episode_steps=MAX_EPISODE_STEPS,
    OBST=OBST,
    reward_type=env_reward_type,
    is_rendered=True,
    is_slippery=is_slippery,
    is_display=False,
)

if training:
    if not use_her:
        sac = SAC("MlpPolicy", env1, gamma = 0.9, verbose=1)
        sac.learn(total_timesteps=NUM_EPISODES_CONT, log_interval=4)
        sac.save("sac_heuristics")
    else:
        goal_selection_strategy = 'future'
        agent = SAC(
            "MultiInputPolicy",
            env1,
            replay_buffer_class=HerReplayBuffer,
            # Parameters for HER
            replay_buffer_kwargs=dict(
                n_sampled_goal=4,
                goal_selection_strategy=goal_selection_strategy,
            ),
            verbose=1,
        )

        # Train the agent
        agent.learn(total_timesteps=NUM_EPISODES_CONT, log_interval=4)
        agent.save("sac_her_heuristics")
    
model = SAC.load("sac_heuristics")
obs, info = env1.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env1.step(action)
    env1.render()
    if terminated or truncated:
        if terminated:
            if env1.goal_reached:
                print("GOAL REACHED")
        elif truncated:
            if env1.fell_in_hole:
                print("FELL IN HOLE")
            elif env1.max_steps_reached:
                print("MAX STEPS REACHED")
        obs, info = env1.reset()
    

