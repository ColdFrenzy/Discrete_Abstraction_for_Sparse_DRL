"""This file contains a Pettingzoo wrapper for the Multi-Agent Continuous UAV Environment
"""
import functools
import numpy as np
from pettingzoo import ParallelEnv
from src.environments.maps import MAPS_FREE
from src.environments.maps import MAPS_OBST
from src.definitions import RewardType, TransitionMode

class MultiAgentContinuousUAVPettingZooWrapper(ParallelEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "name": "multiagent_cont_uav_v0"}

    def __init__(self,
        env,
        map: str = 10,
        agents_pos: dict[str, list[float,float]] = {"a1":  [5.5, 4.5], # [8.5, 2.5], # 
                   "a2":  [0.5, 0.5], # [8.5, 0.5], # 
                   },
        OBST: bool = True,
        reward_type = RewardType.sparse,
        max_episode_steps: int = 200,
        task: str = "reach_target", # reach_target or encircle_target
        desired_orientations: dict[str, list[float,float]] = None,
        desired_distances: dict[str, list[float,float]] = None,
        is_slippery: bool = False,
        is_rendered: bool = True,
        is_display: bool = False,
        collision_radius: float = 0.5,
        render_mode: str = "rgb_array",
        ):
        if OBST:
            map_name = MAPS_OBST[map]
        else:
            map_name = MAPS_FREE[map]
        ParallelEnv.__init__(self)

        self.env = env(
            map=map_name, 
            size=map,
            agents_pos=agents_pos,
            OBST=OBST, 
            reward_type=reward_type,
            max_episode_steps=max_episode_steps,
            task=task,
            desired_orientations=desired_orientations,
            desired_distances=desired_distances,
            is_slippery=is_slippery,
            is_rendered=is_rendered,
            is_display=is_display,
            collision_radius=collision_radius
        )
        self.possible_agents = self.env.agents
        # mapping between agent name and ID
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self._action_spaces = {agent: self.action_space(agent) for agent in self.possible_agents}
        self._observation_spaces = {
            agent: self.observation_space(agent) for agent in self.possible_agents
        }
        self.render_mode = render_mode
        self._max_episode_steps = self.env._max_episode_steps
        # counter for the terminated steps of each agent (give reward only when agent is terminated for the first time)

    def observation_space(self, agent):
        return self.env.observation_space
    def action_space(self, agent):
        return self.env.action_space

    def reset(self, seed=None, options=None):
        # reset the environment
        self.agents = self.possible_agents[:]
        # counter for the terminated steps of each agent (give reward only when agent is terminated for the first time)
        observations, infos = self.env.reset(seed=seed)
        for agent in self.possible_agents:
            infos[agent] = {"GOAL REACHED": False}
        new_observations = {}
        for agent in self.possible_agents:
            one_hot_encoding = [0 for _ in range(len(self.possible_agents))]
            one_hot_encoding[self.agent_name_mapping[agent]] = 1.
            one_hot_encoding = np.array(one_hot_encoding)
            new_observations[agent] = np.concat((one_hot_encoding, (observations[agent] / self.env.size), (self.env.goal/self.env.size)))
        return new_observations, infos

    def step(self, actions):
        # step the environment
        observations, rewards, env_terminations, env_truncations, infos = self.env.step(actions)
        new_observations = {}
        new_infos = {agent: {} for agent in self.possible_agents}
        for agent in self.possible_agents:
            # don't update the observation if 
            one_hot_encoding = [0 for _ in range(len(self.possible_agents))]
            one_hot_encoding[self.agent_name_mapping[agent]] = 1.
            one_hot_encoding = np.array(one_hot_encoding)
            new_observations[agent] = np.concat((one_hot_encoding, (observations[agent] / self.env.size), (self.env.goal/self.env.size))) if observations[agent][0] >= 0 else np.concat((one_hot_encoding, observations[agent] , (self.env.goal/self.env.size)))

            if "GOAL REACHED" in infos[agent]:
                new_infos[agent]["GOAL REACHED"] = infos[agent]["GOAL REACHED"]
            else:
                new_infos[agent] = {"GOAL REACHED":  False}

        # terminate the environment only if all agents are terminated
        if not all(env_terminations.values()):
            env_terminations = {agent: False for agent in env_terminations.keys()}

        return new_observations, rewards, env_terminations, env_truncations, new_infos

    def render(self, mode="rgb_array"):
        # render the environment
        rgb_image = self.env.render(mode=mode)
        return rgb_image

if __name__ == "__main__":
    from pettingzoo.test import parallel_api_test
    from src.environments.maps import MAPS_FREE
    from src.environments.maps import MAPS_OBST
    from src.definitions import RewardType, TransitionMode

    map_size = 10
    OBST = False
    if OBST:
        map_name = MAPS_OBST[map_size]
    else:
        map_name = MAPS_FREE[map_size]
    MAX_EPISODE_STEPS = 200
    task = "reach_target"
    agents_pos = {"a1":  [0.5, 0.5],
                   "a2":  [0.5, 0.5],
                   "a3":  [0.5, 0.5]
                   }
    is_slippery = False
    env_reward_type = RewardType.sparse
    env = MultiAgentContinuousUAVPettingZooWrapper(
        map = map_size,
        agents_pos = agents_pos,
        OBST = OBST,
        reward_type = RewardType.sparse,
        max_episode_steps = MAX_EPISODE_STEPS,
        task = task, # reach_target or encircle_target
        desired_orientations = None,
        desired_distances = None,
        is_slippery = is_slippery,
        is_rendered = True,
        is_display = True,
        collision_radius = 0.5,
        render_mode = "human"
        )
    parallel_api_test(env, num_cycles=100)

