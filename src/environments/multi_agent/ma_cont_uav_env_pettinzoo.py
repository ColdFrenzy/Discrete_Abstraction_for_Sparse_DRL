"""This file contains a Pettingzoo wrapper for the Multi-Agent Continuous UAV Environment
"""
import functools
from pettingzoo import ParallelEnv
from src.environments.multi_agent.ma_cont_uav_env import MultiAgentContinuousUAV


class MultiAgentContinuousUAVPettingZooWrapper(MultiAgentContinuousUAV, ParallelEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "name": "multiagent_cont_uav_v0"}

    def __init__(self,
        map: str = 10,
        agents_pos: dict[str, list[float,float]] = {"a1":  [0.5, 0.5],
                   "a2":  [0.5, 0.5],
                   "a3":  [0.5, 0.5]
                   },
        OBST: bool = False,
        reward_type = 1,
        max_episode_steps: int = 200,
        task: str = "reach_target", # reach_target or encircle_target
        desired_orientations: dict[str, list[float,float]] = None,
        desired_distances: dict[str, list[float,float]] = None,
        is_slippery: bool = False,
        is_rendered: bool = False,
        is_display: bool = True,
        collision_radius: float = 0.5,
        render_mode: str = "human",
        ):
        MultiAgentContinuousUAV.__init__(
            self, 
            map=map, 
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
        ParallelEnv.__init__(self)
        self.possible_agents = self.agents
        # mapping between agent name and ID
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self._action_spaces = {agent: self.action_space(agent) for agent in self.possible_agents}
        self._observation_spaces = {
            agent: self.observation_space(agent) for agent in self.possible_agents
        }
        self.render_mode = None


    def reset(self, seed=None, options=None):
        # reset the environment
        self.agents = self.possible_agents[:]
        return MultiAgentContinuousUAV.reset(self, seed=seed)




if __name__ == "__main__":
    from pettingzoo.test import parallel_api_test
    from environments.maps import MAPS_FREE
    from environments.maps import MAPS_OBST
    from utils.definitions import RewardType, TransitionMode

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

