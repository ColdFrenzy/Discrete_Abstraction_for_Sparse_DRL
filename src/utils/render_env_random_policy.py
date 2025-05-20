from src.environments.multi_agent.ma_cont_uav_env_pettinzoo import MultiAgentContinuousUAVPettingZooWrapper
from src.environments.multi_agent.ma_cont_uav_env import MultiAgentContinuousUAV
import time

def render_env_random_policy(env, num_episodes: int = 1):
    """
    Renders the environment using a random policy for a specified number of episodes.

    Args:
        env (MultiAgentContUAVEnv): The environment to render.
        num_episodes (int): The number of episodes to render. Default is 1.
    """
    for episode in range(num_episodes):
        obs,_ = env.reset()
        env.render()
        # render image 3D image
        done = False
        while not done:
            action = {agent: env.action_space(agent).sample() for agent in env.possible_agents}
            obs, reward, terminated, truncated, info = env.step(action)
            # print("obs: ", env.env.observations)
            env.render()
            time.sleep(0.1)
            if all(terminated.values()) or all(truncated.values()):
                done=True


if __name__ == "__main__":
    env = MultiAgentContinuousUAVPettingZooWrapper(env = MultiAgentContinuousUAV, is_display=True, render_mode="human", task="encircle_target")
    render_env_random_policy(env, num_episodes=1)