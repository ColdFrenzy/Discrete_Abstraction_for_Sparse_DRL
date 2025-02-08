import warnings
warnings.filterwarnings('ignore')

import os
import wandb
import numpy as np

from src.environments.single_agent.cont_uav_env import ContinuousUAV
from src.environments.multi_agent.ma_cont_uav_env import MultiAgentContinuousUAV
from src.value_function_computation import compute_value_function
from src.algorithms.SAC.sac import SAC
from src.definitions import RewardType, TransitionMode
from src.environments.maps import MAPS_FREE
from src.environments.maps import MAPS_OBST
from src.utils.heatmap import generate_heatmaps_numbers
from src.utils.paths import QTABLE_DIR


os.environ["IS_RENDER"] = "False"
env_reward_type = RewardType.model # or model, or sparse
is_slippery = False
map_size = 10
MAX_EPISODE_STEPS = 400
OBST = True
if OBST:
    map_name = MAPS_OBST[map_size]
else:
    map_name = MAPS_FREE[map_size]



training = True
compute_qtable = True

transition_mode = TransitionMode.stochastic if is_slippery else TransitionMode.deterministic
experiment_name = f"sac-{env_reward_type.name}-{map_size}-obstacles_{OBST}"

# root = project_root()
NUM_EPISODES_DISCRETE = 1000
NUM_EPISODES_CONT = 10

if compute_qtable:
    compute_value_function(map_name, size=map_size, OBST=OBST, num_episodes=NUM_EPISODES_DISCRETE, gamma = 0.8, stochastic=is_slippery, save=True)

if env_reward_type == RewardType.model:
    # for agent_name in ["a1", "a3"]:
    qtable = np.load(f"{QTABLE_DIR}/{transition_mode.name}/qtable_{map_size}_obstacles_{OBST}.npz")
    # np.savez_compressed(f"{QTABLE_DIR}/{transition_mode.name}/qtable_{10}_obstacles_{OBST}.npz", **new_qtable)

    generate_heatmaps_numbers(qtable)


env1 = ContinuousUAV(map_name =  map_name, agent_name="a1", size = map_size, max_episode_steps=MAX_EPISODE_STEPS, OBST=OBST, agent_initial_pos = [0.5, map_size-0.5], reward_type = env_reward_type, is_rendered = True, is_slippery = is_slippery, is_display=False)
env2 = ContinuousUAV(map_name =  map_name, agent_name="a3", size = map_size, max_episode_steps=MAX_EPISODE_STEPS, OBST=OBST, agent_initial_pos = [0.5, 0.5], reward_type = env_reward_type, is_rendered = True, is_slippery = is_slippery, is_display=False)
agent1 = SAC(name = experiment_name, agent_name="a1",  env = env1, max_episodes = NUM_EPISODES_CONT, alpha_initial=0.4, alpha_final=0.01, gamma=0.1, max_ep_alpha_decay=2000)
agent2 = SAC(name = experiment_name, agent_name="a3",  env = env2, max_episodes = NUM_EPISODES_CONT, alpha_initial=0.4, alpha_final=0.01, gamma=0.1, max_ep_alpha_decay=2000) # alpha_final=0.05,



if training:
    wandb.init(project="drl-cont-uav", group=experiment_name)
    agent1.train()
    agent1.save()
    # env1.save_episode(999)
    agent2.train()
    # env2.save_episode(999)
    agent2.save()
    wandb.finish()
    



# agent1.load()
# agent2.load()
# env1.render_episode(agent1)
multiagent_env = MultiAgentContinuousUAV(map_name =  map_name, num_agents = 2, size = map_size, OBST=OBST, reward_type = env_reward_type, is_rendered = True, is_slippery = is_slippery, is_display=False)
multiagent_env.render_episode(agents=[agent1, agent2], max_steps=100)
multiagent_env.save_episode(100, name="multi_uav_cont")
# env1.is_rendered = True
# agent1.evaluate()
# env1.quit_render()