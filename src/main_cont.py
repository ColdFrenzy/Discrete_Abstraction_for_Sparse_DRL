import warnings

warnings.filterwarnings("ignore")

import wandb
import numpy as np

from src.environments.single_agent.cont_uav_env import ContinuousUAV
from src.value_function_computation import compute_value_function_single
from src.algorithms.SAC.sac import SAC
from src.algorithms.HER.sac_her import SAC_HER
from src.definitions import RewardType, TransitionMode
from src.environments.maps import MAPS_FREE
from src.environments.maps import MAPS_OBST
from src.utils.heatmap import generate_heatmaps_numbers
from src.utils.paths import QTABLE_DIR


env_reward_type = RewardType.model  # or model, or sparse
is_slippery = False
map_size = 9
OBST = True
if OBST:
    map_name = MAPS_OBST[map_size]
else:
    map_name = MAPS_FREE[map_size]

compute_qtable = False
training = True
evaluate = True

transition_mode = (
    TransitionMode.stochastic if is_slippery else TransitionMode.deterministic
)
experiment_name = f"sac-{env_reward_type.name}-{map_size}-obstacles_{OBST}"

NUM_EPISODES_DISCRETE = 20_000
NUM_EPISODES_CONT = 5_000

if compute_qtable:
    compute_value_function_single(
        map_name,
        size=map_size,
        OBST=OBST,
        num_episodes=NUM_EPISODES_DISCRETE,
        gamma=0.9,
        stochastic=is_slippery,
        save=True,
    )

if env_reward_type == RewardType.model:
    qtable = np.load(
        f"{QTABLE_DIR}/{transition_mode.name}/single_agent/qtable_{map_size}_obstacles_{OBST}.npz"
    )
    generate_heatmaps_numbers(qtable)


def get_agent_pos(idx: int) -> float:
    """
    idx goes from 0 to map_size (exclusive)
    """
    return ((idx + 1) + idx) * 0.5


env = ContinuousUAV(
    map_name=map_name,
    agent_name="a1",
    size=map_size,
    OBST=OBST,
    max_episode_steps=300,
    agent_initial_pos=tuple([get_agent_pos(4), get_agent_pos(8)]),
    reward_type=env_reward_type,
    is_rendered=False,
    is_slippery=is_slippery,
)
agent = SAC(
    name=experiment_name,
    agent_name="a1",
    env=env,
    max_episodes=NUM_EPISODES_CONT,
    alpha_initial=0.7,
    alpha_final=0.05,
    gamma=0.1,
    batch_size=8,
    ep_update_freq=1,
    gradient_steps=1,
    pi_lr=0.0005,
    q_lr=0.0005,
    max_ep_alpha_decay=NUM_EPISODES_CONT,
    buffer_capacity=10_000,
)

if training:
    wandb.init(project="drl-cont-uav", group=experiment_name)
    agent.train()
    agent.save()
    wandb.finish()

if evaluate:
    agent.load(NUM_EPISODES_CONT+1)
    env.is_rendered = True
    env.init_render()
    agent.evaluate(env)
    env.quit_render()
