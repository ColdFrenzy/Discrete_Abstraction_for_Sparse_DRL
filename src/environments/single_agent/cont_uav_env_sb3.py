from typing import Tuple

import os
import cv2
import imageio
import numpy as np
import pygame
import torch
import gymnasium as gym
from gymnasium import error


from gymnasium.core import Env
from gymnasium.spaces import Box, Dict
from typing import Optional
from pygame.image import load
from pygame.transform import scale
from src.utils.paths import IMAGE_DIR, EPISODES_DIR

from src.definitions import RewardType, TransitionMode
from src.environments.single_agent.cont_uav_env import ContinuousUAV

from src.utils.utils import parse_map_emoji

from src.utils.paths import QTABLE_DIR



class ContinuousUAVSb3HerWrapper(ContinuousUAV):
    """
    The UAVEnv environment is a simple gridworld MDP with a start state, a
    goal state, and holes. For simplicity, the map is assumed to be a squared
    grid.

    The agent can move in the two-dimensional grid with continuous velocities.
    Positive y is downwards, positive x is to the right.
    """
    def __init__(
        self,
        map_name: str = "6x6",
        agent_name: str = "a1",
        size: int =  10,
        agent_initial_pos: Tuple[float, float] = (2.5, 4.5),
        OBST: bool = False,
        reward_type: RewardType = RewardType.dense,
        max_episode_steps: int = 100,
        is_slippery: bool = False,
        is_rendered: bool = False,
        is_display: bool = True,
        seed: int = 13,
    ):
        super().__init__(
            map_name=map_name,
            agent_name=agent_name,
            size=size,
            agent_initial_pos=agent_initial_pos,
            OBST=OBST,
            reward_type=reward_type,
            max_episode_steps=max_episode_steps,
            is_slippery=is_slippery,
            is_rendered=is_rendered,
            is_display=is_display,
        )
        self.seed = seed
        # Environment parameters
        self.observation_space = Dict({
            'observation': Box(low=0, high=self.size, shape=(2,),  dtype=np.float32),
            'achieved_goal': Box(low=0, high=self.size, shape=(2,),  dtype=np.float32),
            'desired_goal': Box(low=0, high=self.size, shape=(2,),  dtype=np.float32),
        })


    def compute_reward(self, achieved_goal, desired_goal, info):
        """Compute the step reward. This externalizes the reward function and makes
        it dependent on a desired goal and the one that was achieved. If you wish to include
        additional rewards that are independent of the goal, you can include the necessary values
        to derive it in 'info' and compute it accordingly.

        Args:
            achieved_goal (object): the goal that was achieved during execution
            desired_goal (object): the desired goal that we asked the agent to attempt to achieve
            info (dict): an info dictionary with additional information

        Returns:
            float: The reward that corresponds to the provided achieved goal w.r.t. to the desired
            goal. Note that the following should always hold true:

                ob, reward, terminated, truncated, info = env.step()
                assert reward == env.compute_reward(ob['achieved_goal'], ob['desired_goal'], info)
        """
        
        if len(achieved_goal.shape) == 2:
            batch_size = len(achieved_goal)
            batch_reward = []
            for i in range(batch_size):
                reward = 0
                desired_goal_cell = self.frame2grid(desired_goal[i])
                # Check for wall 
                if achieved_goal[i][0] <= 0 or achieved_goal[i][0] >= self.size:
                    reward = -1
                elif achieved_goal[i][1] <= 0 or achieved_goal[i][1] >= self.size:
                    reward = -1     

                # Check for failure termination
                elif self.holes is not None:
                    for hole in self.holes:
                        if self.is_inside_cell(achieved_goal[i], hole):
                            reward = -10
                            break
                # Check for successful termination
                elif self.is_inside_cell(achieved_goal[i], desired_goal_cell):
                    reward = 10

                
                batch_reward.append(reward)
            return np.array(batch_reward)
                
        else:
            reward = 0
            desired_goal_cell = self.frame2grid(desired_goal)
            # Check for wall 
            if achieved_goal[0] <= 0 or achieved_goal[0] >= self.size:
                reward = -1
            elif achieved_goal[1] <= 0 or achieved_goal[1] >= self.size:
                reward = -1     

            # Check for failure termination
            if self.holes is not None:
                for hole in self.holes:
                    if self.is_inside_cell(achieved_goal, hole):
                        reward = -10
                        break
            # Check for successful termination
            elif self.is_inside_cell(achieved_goal, desired_goal_cell):
                reward = 10

            return reward



    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        The agent takes a step in the environment.
        """
        observation, reward, terminated, truncated, log = super().step(action)


        obs_goal = {
            'observation': observation.copy().astype(np.float32),
            'achieved_goal': observation.copy().astype(np.float32),
            'desired_goal': self.goal.copy().astype(np.float32),
        }
        # new_obs, rewards, dones, infos
        return (
            obs_goal,
            reward,
            terminated,
            truncated,
            log,
        )

    def render(self, mode="human"):
        super().render()

    def reset(self, seed=None) -> dict:
        seed = seed if seed is not None else self.seed
        super().reset(seed=seed)
        # Enforce that each GoalEnv uses a Goal-compatible observation space.
        if not isinstance(self.observation_space, gym.spaces.Dict):
            raise error.Error(
                "GoalEnv requires an observation space of type gym.spaces.Dict"
            )
        for key in ["observation", "achieved_goal", "desired_goal"]:
            if key not in self.observation_space.spaces:
                raise error.Error(
                    'GoalEnv requires the "{}" key to be part of the observation dictionary.'.format(
                        key
                    )
                )
        observation = {
            'observation': self.observation.copy().astype(np.float32),
            'achieved_goal': self.observation.copy().astype(np.float32),
            'desired_goal': self.goal.copy().astype(np.float32),
        }
    
        return observation, {}



    def compute_truncated(self, achieved_goal, desired_goal, info):
        """Compute the step truncation. Allows to customize the truncated states depending on the
        desired and the achieved goal. If you wish to determine truncated states independent of the goal,
        you can include necessary values to derive it in 'info' and compute it accordingly. Truncated states
        are those that are out of the scope of the Markov Decision Process (MDP) such as time constraints in a
        continuing task. More information can be found in: https://farama.org/New-Step-API#theory

        Args:
            achieved_goal (object): the goal that was achieved during execution
            desired_goal (object): the desired goal that we asked the agent to attempt to achieve
            info (dict): an info dictionary with additional information

        Returns:
            bool: The truncated state that corresponds to the provided achieved goal w.r.t. to the desired
            goal. Note that the following should always hold true:

                ob, reward, terminated, truncated, info = env.step()
                assert truncated == env.compute_truncated(ob['achieved_goal'], ob['desired_goal'], info)
        """
        if self.num_steps >= self.max_episode_steps:
            return True

        return False
    

    def compute_terminated(self, achieved_goal, desired_goal, info):
        """Compute the step termination. Allows to customize the termination states depending on the
        desired and the achieved goal. If you wish to determine termination states independent of the goal,
        you can include necessary values to derive it in 'info' and compute it accordingly. The envirtonment reaches
        a termination state when this state leads to an episode ending in an episodic task thus breaking .
        More information can be found in: https://farama.org/New-Step-API#theory

        Termination states are

        Args:
            achieved_goal (object): the goal that was achieved during execution
            desired_goal (object): the desired goal that we asked the agent to attempt to achieve
            info (dict): an info dictionary with additional information

        Returns:
            bool: The termination state that corresponds to the provided achieved goal w.r.t. to the desired
            goal. Note that the following should always hold true:

                ob, reward, terminated, truncated, info = env.step()
                assert terminated == env.compute_terminated(ob['achieved_goal'], ob['desired_goal'], info)
        """
        if self.is_inside_cell(achieved_goal, self.goal):
            return True

        return False