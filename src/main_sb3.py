import os
from src.definitions import RewardType, TransitionMode
from src.environments.maps import MAPS_FREE
from src.environments.maps import MAPS_OBST
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import HerReplayBuffer, DDPG, DQN, SAC, TD3
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from src.environments.single_agent.cont_uav_env_sb3 import ContinuousUAVSb3

model_class = SAC  # works also with SAC, DDPG and TD3

os.environ["IS_RENDER"] = "False"
env_reward_type = RewardType.sparse # or model, or sparse
is_slippery = False
map_size = 10
MAX_EPISODE_STEPS = 400
NUM_BATCHES = 10
OBST = True
if OBST:
    map_name = MAPS_OBST[map_size]
else:
    map_name = MAPS_FREE[map_size]


env = ContinuousUAVSb3(map_name =  map_name, agent_name="a1", size = map_size, max_episode_steps=MAX_EPISODE_STEPS, OBST=OBST, agent_initial_pos = [0.5, map_size-0.5], reward_type = env_reward_type, is_rendered = True, is_slippery = is_slippery, is_display=False)
# Available strategies (cf paper): future, final, episode


# test if the environment follows the gym interface
# check_env(env)

goal_selection_strategy = "future" # equivalent to GoalSelectionStrategy.FUTURE

# Initialize the model
model = model_class(
    "MultiInputPolicy",
    env,
    replay_buffer_class=HerReplayBuffer,
    learning_starts=1e4,
    # Parameters for HER
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy=goal_selection_strategy,
    ),
    tensorboard_log="./her_sac_uav_tensorboard/",
    verbose=2,
)

# Train the model
print("start learning")
model.learn(100000)
print("learning done")
model.save("./her_uav_env")
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