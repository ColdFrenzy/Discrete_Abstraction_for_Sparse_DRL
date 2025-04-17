import os
from typing import Tuple

import numpy as np
import pygame
import torch

from src.definitions import RewardType
from src.environments.single_agent.cont_uav_env import ContinuousUAV


class MultiAgentContinuousUAV(ContinuousUAV):
    """
    Multi-agent version of the ContinuousUAV environment.
    """

    def __init__(
        self,
        num_agents: int,
        map_name: str = "6x6",
        agents_pos: list = [(0.5, 9.5), (0.5, 0.5)],
        size: int = 10,
        OBST: bool = False,
        reward_type: RewardType = RewardType.dense,
        task: str = "encircle_target", # reach_target or encircle_target
        desired_orientations: list[list[float,float]] = None,
        desired_distances: list[list[float,float]] = None,
        is_slippery: bool = False,
        is_rendered: bool = False,
        is_display: bool = True,
    ):
        super().__init__(map_name=map_name, size=size, OBST=OBST, reward_type=reward_type, is_slippery=is_slippery, task = task, is_rendered=is_rendered, is_display=is_display)
        self.num_agents = num_agents
        self.agents_initial_pos = [agents_pos[i] for i in range(num_agents)]
        self.start_observations = self.agents_initial_pos.copy()
        self.rewards = [0.0 for _ in range(num_agents)]
        self.prev_observations = self.agents_initial_pos.copy()
        self.prev_actions = [np.zeros(2) for _ in range(num_agents)]
        self.trajectories = [[] for _ in range(num_agents)]
        self.task = task
        if self.task == "encircle_target":
            self.desired_orientations = desired_orientations
            self.desired_distances = desired_distances
        self.terminated = {i: False for i in range(num_agents)}
        self.truncated = {i: False for i in range(num_agents)}

    def step(self, actions: list) -> Tuple[list, list, bool, dict]:
        """
        The agents take a step in the environment.
        """

        logs = []

        for i, action in enumerate(actions):
            if self.terminated[i] or self.truncated[i]:
                continue
            self.prev_observations[i] = self.observations[i]
            action = np.clip(action, self.action_space.low, self.action_space.high)

            new_x = self.observations[i]["observation"][0,0] + action[0, 0]
            new_y = self.observations[i]["observation"][0,1] + action[0, 1]

            if self.is_slippery:
                new_x += np.random.normal(0, 0.01)
                new_y += np.random.normal(0, 0.01)
            if self.task == "encircle_target":
                self.desired_distance = self.desired_distances[i]
                self.desired_orientation = self.desired_orientations[i]
            self.observations[i]["observation"], self.rewards[i], agent_terminated, agent_truncated, log = self.reward_function(
                np.array([new_x, new_y])
            )

            self.trajectories[i].append(torch.tensor(self.observations[i]["observation"]))
            self.observations[i]["observation"] = torch.tensor(self.observations[i]["observation"], dtype=torch.float32).unsqueeze(0)
            self.num_steps += 1

            if agent_terminated:
                self.terminated[i] = True
            if agent_truncated:
                self.truncated[i] = True
            logs.append(log)

        return self.observations, self.rewards, self.terminated, self.truncated, {"logs": logs}

    def reset(self) -> Tuple[list, dict]:
        super().reset()
        self.num_steps = 0
        self.observations = [np.array(initial_pos) for initial_pos in self.agents_initial_pos]
        self.start_observations = self.observations.copy()
        self.trajectories = [[] for _ in range(self.num_agents)]
        self.frames = []
        return self.observations, {}
    
    
    def render(self):
        """
        Renders the environment with the given observations.
        """
        if not self.is_pygame_initialized and self.is_rendered:
            self.init_render()
            self.is_pygame_initialized = True
            self.trajectories = [[] for _ in range(self.num_agents)]
            self.frames = []

        for x in range(0, self.screen_width, self._cell_size):
            for y in range(0, self.screen_height, self._cell_size):
                # Draw the ice
                rect = ((x, y), (self._cell_size, self._cell_size))
                self.screen.blit(self.ice_img, (x, y))
                pygame.draw.rect(self.screen, (200, 240, 240), rect, 1)

        # Draw the holes
        for hole in self.holes:
            hole_x, hole_y = hole
            hole_x = hole_x * self._cell_size
            hole_y = hole_y * self._cell_size
            self.screen.blit(self.hole_img, (hole_x, hole_y))

        for goal_char, (g_x, g_y) in self.goals.items():
            goal_rect = pygame.Rect(
                g_x * self._cell_size,
                g_y * self._cell_size,
                self._cell_size,
                self._cell_size,
            )
            pygame.draw.rect(self.screen, (255, 215, 0), goal_rect)
            goal_text = self.font.render(str(goal_char), True, (0, 0, 0))
            text_rect = goal_text.get_rect(center=goal_rect.center)
            self.screen.blit(goal_text, text_rect)

        # Draw the agents
        for i, observation in enumerate(self.observations):
            x, y = self.frame2grid(observation["observation"][0])
            agent_x = x * self._cell_size - self.agent_img.get_width() // 2
            agent_y = y * self._cell_size - self.agent_img.get_height() // 2
            self.screen.blit(self.agent_img, (agent_x, agent_y))

            # Draw the trajectory
            if self.trajectories[i]:
                for point in self.trajectories[i]:
                    traj_x, traj_y = self.frame2grid(point)
                    traj_x = traj_x * self._cell_size
                    traj_y = traj_y * self._cell_size
                    pygame.draw.circle(self.screen, (255, 0, 0), (traj_x, traj_y), 5)

        if not self.is_display:
            image_data = pygame.surfarray.array3d(self.screen)
        else:
            image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        image_data = image_data.transpose([1, 0, 2])
        self.frames.append(image_data)

        if self.is_display:
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(60)

    def render_episode(self, agents: list, max_steps: int = None):
        """
        Renders an episode with the given agents.

        Args:
            agents (list): List of agent positions.
            max_steps (int): Maximum number of steps in the episode.
        """
        self.observations,_ = self.reset()
        all_obs= []
        for i, _ in enumerate(agents):
            all_obs.append({o: torch.tensor(self.observations[i][o],dtype=torch.float32).unsqueeze(0) for o in self.observations[i]})
        self.observations = all_obs
        if max_steps is None:
            max_steps = self._max_episode_steps
        for step in range(max_steps):
            with torch.no_grad():
                actions = []
                for i, agent in enumerate(agents):
                    action, _ = agent.predict(self.observations[i], deterministic=True)
                    actions.append(action)
            self.observations,  rewards,  terminated, truncated, _ = self.step(actions)


                    # Debug prints
            # print(f"Step: {step}")
            # print(f"Actions: {actions}")
            # print(f"Observations: {self.observations}")
            # print(f"Rewards: {rewards}")
            # print(f"Terminated: {terminated}")
            # print(f"Truncated: {truncated}")
            
            
            self.render()
            if all(truncated.values()):
                break