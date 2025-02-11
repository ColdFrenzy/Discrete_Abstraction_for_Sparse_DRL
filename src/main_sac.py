import warnings
warnings.filterwarnings('ignore')

import wandb
import os
import numpy as np

from src.environments.single_agent.cont_uav_env import ContinuousUAV
from src.value_function_computation import compute_value_function_single
from src.algorithms.SAC.sac import SAC
from src.definitions import RewardType, TransitionMode
from src.environments.maps import MAPS_FREE
from src.environments.maps import MAPS_OBST
from src.utils.heatmap import generate_heatmaps_numbers
from src.utils.paths import QTABLE_DIR

os.environ["IS_RENDER"] = "False"
env_reward_type = RewardType.sparse # or model, or sparse
is_slippery = False
map_size = 10
MAX_EPISODE_STEPS = 400
OBST = True
if OBST:
    map_name = MAPS_OBST[map_size]
else:
    map_name = MAPS_FREE[map_size]



training = True
compute_qtable = False

transition_mode = TransitionMode.stochastic if is_slippery else TransitionMode.deterministic
experiment_name = f"sac-{env_reward_type.name}-{map_size}-obstacles_{OBST}"

# root = project_root()
NUM_EPISODES_DISCRETE = 20000
NUM_EPISODES_CONT = 1000

if compute_qtable:
    compute_value_function_single(map_name, size=map_size, OBST=OBST, num_episodes=NUM_EPISODES_DISCRETE, gamma = 0.8, stochastic=is_slippery, save=True)
    
if env_reward_type == RewardType.model:
    qtable = np.load(f"{QTABLE_DIR}/{transition_mode.name}/single_agent/qtable_{map_size}_obstacles_{OBST}.npz")
    generate_heatmaps_numbers(qtable)


env = ContinuousUAV(map_name =  map_name, agent_name="a1", size = map_size, max_episode_steps=MAX_EPISODE_STEPS, OBST=OBST, agent_initial_pos = [0.5, map_size-0.5], reward_type = env_reward_type, is_rendered = True, is_slippery = is_slippery, is_display=False)
agent = SAC(name = experiment_name,agent_name="a1", env = env, max_episodes = NUM_EPISODES_CONT, alpha_initial=0.4, alpha_final=0.01, gamma=0.1, max_ep_alpha_decay=2000)

if training:
    wandb.init(project="drl-cont-uav", group=experiment_name, mode="disabled")
    agent.train()
    env.save_episode(1)
    env.render()
    wandb.finish()
    agent.save()

agent.load()
env.is_rendered = True
agent.evaluate()
env.quit_render()