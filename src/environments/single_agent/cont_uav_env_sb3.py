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

SEED = 13


class ContinuousUAVSb3(ContinuousUAV):
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
    ):
        super().__init__(
            self,
            map_name: map_name,
            agent_name: agent_name,
            size: size,
            agent_initial_pos: agent_initial_pos,
            OBST: OBST,
            reward_type: reward_type,
            max_episode_steps: max_episode_steps,
            is_slippery: is_slippery,
            is_rendered: is_rendered,
            is_display: is_display,
        )

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
        
        reward = 0
        desired_goal_cell = self.frame2grid()
        # Check for wall 
        if achieved_goal[0] <= 0 or achieved_goal[0] >= self.size:
            reward = -1
        if achieved_goal[1] <= 0 or achieved_goal[1] >= self.size:
            reward = -1     

        # Check for successful termination
        if self.is_inside_cell(achieved_goal, desired_goal_cell):
            reward += 10


        # Check for failure termination
        for hole in self.holes:
            if self.is_inside_cell(achieved_goal, hole):

                log = "HOLE"
                break

        return reward


        raise NotImplementedError



    def reward_function_batch(self, obs: torch.tensor, desired_goal: torch.tensor, info: dict = None) -> Tuple[float, bool]:
        # obs size [batch_size, 2]
        reward = []
        log = None
        # Check for wall 
        for i in range(obs.size(0)):
            step_reward = 0
            if obs[i, 0] <= 0 or obs[i, 0] >= self.size:
                obs[i, 0] = np.clip(obs[i, 0], 0.05, self.size - 0.05)
                step_reward = -1
            if obs[i, 1] <= 0 or obs[i, 1] >= self.size:
                obs[i, 1] = np.clip(obs[i, 1], 0.05, self.size - 0.05)
                step_reward = -1
            

            # check for goal
            if self.is_inside_cell(obs[i], desired_goal[i]):
                log = "GOAL REACHED"
                step_reward = 10
            
            # Check for failure termination
            for hole in self.holes:
                if self.is_inside_cell(obs[i], hole):
                    step_reward = -10
                    log = "HOLE"
                    break
            reward.append(step_reward)
        reward = torch.tensor(reward).unsqueeze(1)

        return reward

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        The agent takes a step in the environment.
        """
        self.prev_observation = self.observation
        action = np.clip(
            action, self.action_space.low, self.action_space.high
        )  # Ensure action is within bounds
        # actions right np.array([4., 0.]), left np.array([-4., 0.]), up np.array([0., 4.]), down np.array([0., -4.])
        new_x = self.observation[0] + action[0]
        new_y = self.observation[1] + action[1]

        if self.is_slippery:
            new_x += np.random.normal(0, 0.01)
            new_y += np.random.normal(0, 0.01)

        self.observation, self.reward, terminated, truncated, log = self.reward_function(
            np.array([new_x, new_y])
        )

        self.trajectory.append(self.observation)
        self.num_steps += 1

        return (
            self.observation,
            self.reward,
            terminated,
            truncated,
            {"log": log},
        )

    def reset(self) -> Tuple[np.ndarray, dict]:
        super().reset(seed=SEED)
        # Enforce that each GoalEnv uses a Goal-compatible observation space.
        if not isinstance(self.observation_space, gym.spaces.Dict):
            raise error.Error(
                "GoalEnv requires an observation space of type gym.spaces.Dict"
            )
        self.num_steps = 0
        for key in ["observation", "achieved_goal", "desired_goal"]:
            if key not in self.observation_space.spaces:
                raise error.Error(
                    'GoalEnv requires the "{}" key to be part of the observation dictionary.'.format(
                        key
                    )
                )
        observation = {
            'observation': self.observation.copy(),
            'achieved_goal': self.observation.copy(),
            'desired_goal': self.goal.copy(),
        }
    
        self.start_obs = self.observation
        self.trajectory = []
        self.frames = []
        self.reward = 0
        return observation



    def is_inside_cell(self, pos: np.ndarray, cell: np.ndarray) -> bool:
        """
        Check if a position is inside a cell.
        """
        cell_coord = self.grid2frame(cell)
        inside = True
        if pos[0] < cell_coord[0] - 0.5:
            inside = False  # left
        if pos[0] > cell_coord[0] + 0.5:
            inside = False  # right
        if pos[1] < cell_coord[1] - 0.5:
            inside = False  # bottom
        if pos[1] > cell_coord[1] + 0.5:
            inside = False  # top
        return inside

    def load_values(self) -> np.ndarray:
        """
        Load the Q-table from the file system.
        """
        qtable = np.load(f"{QTABLE_DIR}/{self.transition_mode.name}/qtable_{self.size}_obstacles_{self.OBST}.npz")[self.agent_name]
        return qtable

    def frame2matrix(self, frame_pos: np.ndarray) -> np.ndarray:
        """
        Convert a frame position to a matrix index.
        """
        x, y = self.frame2grid(frame_pos)

        # Inverting the coordinates
        # x actually represents the columns and y the rows
        indices = np.floor(np.array([y, x])).astype(int)

        return indices

    def frame2grid(self, frame_pos: np.ndarray) -> np.ndarray:
        """
        Convert a frame position to a grid position.
        """
        assert len(frame_pos) == 2
        x, y = frame_pos

        # Flipping y axis
        y = self.size - y

        cell = np.array([x, y])

        return cell

    def grid2frame(self, grid_pos: np.ndarray) -> np.ndarray:
        """
        Convert a grid position to a frame position.
        """
        assert len(grid_pos) == 2
        x, y = grid_pos
        # Flipping y axis
        y = self.size - y

        # Centering the position
        x += 0.5
        y -= 0.5

        return np.array([x, y])

    def render(self):
        """
        Renders the environment with the given observation.

        Args:
            obs (np.ndarray): The observation to render.
        """
        if not self.is_pygame_initialized and self.is_rendered:
            self.init_render()
            self.is_pygame_initialized = True
            self.trajectory = []
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
            hole_x = (
                hole_x * self._cell_size
            )  # - self.hole_img.get_width() // 2
            hole_y = (
                hole_y * self._cell_size
            )  # - self.hole_img.get_height() // 2
            if self.is_inside_cell(self.observation, hole):
                self.screen.blit(self.cracked_hole_img, (hole_x, hole_y))
            else:
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

        # Draw the agent (it lives in the continuous reference frame with different coordinates)
        # Namely, [0,0] is in the bottom left corner, as if the grid was a cartesian plane
        # Thus, positive y speed is upwards, positive x speed is to the right
        x, y = self.frame2grid(self.observation)
        agent_x = x * self._cell_size - self.agent_img.get_width() // 2
        agent_y = y * self._cell_size - self.agent_img.get_height() // 2
        self.screen.blit(self.agent_img, (agent_x, agent_y)) 

        # Draw the trajectory
        if self.trajectory:
            for point in self.trajectory:
                traj_x, traj_y = self.frame2grid(point)
                traj_x = traj_x * self._cell_size
                traj_y = traj_y * self._cell_size
                pygame.draw.circle(
                    self.screen, (255, 0, 0), (traj_x, traj_y), 5
                )

        reward_text = f"Reward: {self.reward}"
        text_surface = self.font.render(reward_text, True, (0, 0, 0))  # White color
        self.screen.blit(text_surface, (10, 10))  # Display text at top-left corner


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
        
    def render_episode(self, agent: "RLAgent", max_steps: int = None):
        """
        Renders an episode with the given agents.
        """
        self.observations,_ = self.reset()
        self.render()
        if max_steps is None:
            max_steps = self._max_episode_steps
        for step in range(max_steps):
            with torch.no_grad():
                obs_goal = torch.cat([torch.tensor(self.observations, dtype=torch.float32), torch.tensor(self.goal, dtype=torch.float32)])
                actions, _ = agent.actor(
                            obs_goal, deterministic=True, with_logprob=False
                        )
            self.observations,  self.reward, terminated, truncated, _ = self.step(actions.cpu().numpy())
            self.render()
            if truncated:
                break


    def save_episode(self, episode, name="uav_cont"):
        # Creare la cartella "episodes" se non esiste

        episodes_dir = EPISODES_DIR
        if not os.path.exists(episodes_dir):
            os.makedirs(episodes_dir, exist_ok=True)


        if self.frames:
            # Salva in formato AVI senza convertire i frames in BGR, poiché pygame li fornisce in RGB
            video_path = f"{episodes_dir}/{name}_episode_{episode}.avi"
            height, width, layers = self.frames[0].shape
            video = cv2.VideoWriter(
                video_path, cv2.VideoWriter_fourcc(*"DIVX"), 2, (width, height)
            )

            for frame in self.frames:
                video.write(
                    cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                )  # Usa i frames direttamente senza conversione

            video.release()

            # Salva in formato GIF
            gif_path = f"{episodes_dir}/{name}_episode_{episode}.gif"
            # I frames sono già in RGB, quindi li usiamo direttamente
            imageio.mimsave(gif_path, self.frames, fps=10, loop=0)

            self.frames = []  # Pulisci la lista dei frames

    def init_render(self):
        """
        Initialize the Pygame environment.
        """
        pygame.init()
        self.clock = pygame.time.Clock()
        # Screen
        self.screen_width = self.grid_width * self._cell_size
        self.screen_height = self.grid_width * self._cell_size
        if not self.is_display:
            self.screen = pygame.Surface((self.screen_width, self.screen_height))
        else:
            self.screen = pygame.display.set_mode(
                (int(self.screen_width), int(self.screen_height))
            )

        # Images
        self.font = pygame.font.SysFont("Arial", 25)  # Crea un oggetto font

        agent_img_path = IMAGE_DIR / "drone.png"
        self.agent_img = scale(
            load(agent_img_path), (self._cell_size, self._cell_size)
        )
        ice_img_path = IMAGE_DIR / "white.png"
        self.ice_img = scale(
            load(ice_img_path), (self._cell_size, self._cell_size)
        )
        hole_img_path = IMAGE_DIR / "red.png"
        self.hole_img = scale(
            load(hole_img_path), (self._cell_size, self._cell_size)
        )
        cracked_hole_img_path = IMAGE_DIR / "red.png"
        self.cracked_hole_img = scale(
            load(cracked_hole_img_path), (self._cell_size, self._cell_size)
        )
        goal_img_path = IMAGE_DIR / "yellow.png"
        self.goal_img = scale(
            load(goal_img_path), (self._cell_size // 3, self._cell_size // 3)
        )

    def quit_render(self):
        """
        Quit the Pygame environment.
        """
        pygame.quit()




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
        raise NotImplementedError


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
        raise NotImplementedError