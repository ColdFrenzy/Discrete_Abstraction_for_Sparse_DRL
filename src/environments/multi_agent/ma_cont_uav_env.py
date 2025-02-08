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
        is_slippery: bool = False,
        is_rendered: bool = False,
        is_display: bool = True,
    ):
        super().__init__(map_name=map_name, size=size, OBST=OBST, reward_type=reward_type, is_slippery=is_slippery, is_rendered=is_rendered, is_display=is_display)
        self.num_agents = num_agents
        self.agents_initial_pos = [agents_pos[i] for i in range(num_agents)]
        self.start_observations = self.agents_initial_pos.copy()
        self.rewards = [0.0 for _ in range(num_agents)]
        self.prev_observations = self.agents_initial_pos.copy()
        self.prev_actions = [np.zeros(2) for _ in range(num_agents)]
        self.trajectories = [[] for _ in range(num_agents)]
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
            action = np.clip(action.cpu().numpy(), self.action_space.low, self.action_space.high)

            new_x = self.observations[i][0] + action[0]
            new_y = self.observations[i][1] + action[1]

            if self.is_slippery:
                new_x += np.random.normal(0, 0.01)
                new_y += np.random.normal(0, 0.01)

            self.observations[i], self.rewards[i], agent_terminated, agent_truncated, log = self.reward_function(
                np.array([new_x, new_y])
            )

            self.trajectories[i].append(self.observations[i])
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
            x, y = self.frame2grid(observation)
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
        self.render()
        if max_steps is None:
            max_steps = self._max_episode_steps
        for step in range(max_steps):
            with torch.no_grad():
                actions = []
                for i, agent in enumerate(agents):
                    action, _ = agent.actor(
                            torch.tensor(self.observations[i],dtype=torch.float32), deterministic=True, with_logprob=False
                        )
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

    def render_episode_hardcoded(self, agents: list, max_steps: int = 100):
        observations,_ = self.reset()
        self.render()
        all_states = []
        actions = np.array(
            [[[0.3909, 0.3991], [ 0.1238, -0.1925]],
            [[ 0.3889, -0.3796], [ 0.2673, -0.1021]],
            [[0.3496, 0.4000], [ 0.1318, -0.3205]], 
            [[0.3769, 0.3913], [-0.0013, -0.2847]], 
            [[ 0.3483, -0.1835], [0.1793, 0.2648]], 
            [[ 0.2439, -0.0809], [ 0.1238, -0.1925]],
            [[0.1548, 0.0490], [ 0.1238, -0.1925]], 
            [[ 0.3141, -0.1259], [ 0.1238, -0.1925]], 
            [[ 0.3661, -0.0634], [ 0.1238, -0.1925]], 
            [[ 0.1635, -0.1150], [ 0.1238, -0.1925]], 
            [[ 0.3588, -0.1668], [ 0.1238, -0.1925]], 
            [[ 0.2503, -0.1484], [ 0.1238, -0.1925]], 
            [[0.2419, 0.0659], [ 0.3313, 0.15]], 
            [[ 0.0934, -0.0581], [ 0.3313, -0.2]], 
            [[ 0.3272, -0.0566], [ 0.3313, 0.13]], 
            [[ 0.3826, -0.0848], [ 0.3313, 0.13]], 
            [[ 0.1885, -0.0855], [ 0.3313, -0.3]], 
            [[0.3923, 0.0312], [ 0.3313, 0.3]], 
            [[ 0.3729, -0.0124], [ 0.3313, 0.11]], 
            [[ 0.3402, -0.0679], [ 0.3313, -0.12]], 
            [[ 0.3603, -0.0267], [ 0.3313, 0.3]], 
            [[-0.1101, -0.0334], [ 0.3313, 0.11]], 
            [[0.3967, 0.0497], [ 0.3313, 0.]], 
            [[ 0.2494, -0.0666], [ 0.3313, 0.]], 
            [[-0.3973,  0.0010], [ 0.3313, 0.]], 
            [[0.2827, 0.0560], [ 0.3754, -0.2163]], 
            [[-0.1879, -0.0444], [ 0.3313, 0.]], 
            [[ 0.3423, -0.0145], [ 0.3313, 0.]], 
            [[0.3995, 0.0442], [ 0.3788, 0.1953]], 
            [[ 0.3964, -0.0578], [ 0.3766, -0.3214]], 
            [[ 0.3438, -0.0218], [ 0.3313, 0.]], 
            [[-0.4000,  0.0021], [ 0.3313, 0.]], 
            [[0.3999, 0.0014], [ 0.3313, 0.]], 
            [[-0.3263, -0.0154], [0.2251, -0.4]], 
            [[0.3999, 0.0187], [0.2251, -0.4]], 
            [[ 0.2595, -0.0197], [0.2251, -0.4]], 
            [[ 0.3470, -0.0025], [0.2251, -0.4]], 
            [[ 0.3999, -0.0230], [0.2251, -0.4]], 
            [[0.4000, 0.0224], [0.2251, -0.4]], 
            [[-0.3435, -0.0125], [0.2251, -0.4]], 
            [[0.3845, 0.3081], [0.2251, -0.4]], 
            [[-0.3926, 0.4], [0.2251, -0.4]], 
            [[0.3929, 0.4], [0.2251, -0.4]], 
            [[ 0.2838, 0.4], [0.2251, -0.4]], 
            [[0.3999, 0.4], [0.2251, -0.4]], 
            [[0.4000, 0.4], [-0.3665, -0.3278]], 
            [[ 0.1766, -0.0081], [0.1140, 0.1436]], 
            [[-0.2820,  0.0091], [0.3646, 0.3201]], 
            [[-0.3994,  0.0142], [-0.3626, -0.2311]], 
            [[ 0.4000, -0.0241], [0.3129, 0.0814]], 
            [[0.4000, 0.0073], [ 0.2973, -0.1234]]]
        )

        for step in range(len(actions)):
            observations,  _,  terminated, truncated, _ = self.step(actions[step])
            self.render()
            if all(truncated.values()):
                break
