import os
import cv2
import imageio
import pygame
import torch
import numpy as np

from gymnasium.core import Env
from typing import Tuple
from copy import deepcopy
from pygame.image import load
from pygame.transform import scale
from gymnasium.spaces import Box


from src.utils.paths import IMAGE_DIR, EPISODES_DIR
from src.definitions import RewardType, TransitionMode
from src.environments.single_agent.cont_uav_env import ContinuousUAV
from src.utils.utils import parse_map_emoji
from src.utils.paths import QTABLE_DIR


class MultiAgentContinuousUAV(Env):
    """
    Multi-agent version of the ContinuousUAV environment.
    """
    def __init__(
        self,
        map: str = "empty",
        size: int = 10,
        agents_pos: dict[str, list[float,float]] = {"a1": [2.5, 4.5]},
        OBST: bool = False,
        BS: bool = False,
        reward_type: RewardType = RewardType.dense,
        max_episode_steps: int = 100,
        task: str = "encircle_target", # reach_target or encircle_target
        desired_orientations: dict[str, list[float,float]] = None,
        desired_distances: dict[str, list[float,float]] = None,
        optimal_view: float = 30.0,
        is_slippery: bool = False,
        is_rendered: bool = False,
        is_display: bool = True,
        collision_radius: float = 0.5,
        bs_radius: float = 3,
        total_bandwidth: float = 2,
    ):
        """
        Initialize the MultiAgentContinuousUAV environment.
        Args:
            map(str): emoji map string
            agents_pos (dict): Initial positions of the agents. 
            OBST (bool): Whether to include obstacles.
            reward_type (RewardType): Type of reward.
            max_episode_steps (int): Maximum number of steps in an episode.
            task (str): Task type ("reach_target" or "encircle_target").
            desired_orientations (dict): Desired orientations for encircle_target task.
            desired_distances (dict): Desired distances for encircle_target task.
            optimal_view (float): Optimal angular view for recording the target/goal. angle in degrees from north, counterclockwise.
            is_slippery (bool): Whether the environment is slippery.
            is_rendered (bool): Whether to render the environment.
            is_display (bool): Whether to display the environment.
            collision_radius (float): Collision radius for the agents.
            bs_radius (float): Base station radius for communication.
            total_bandwidth (float): Total bandwidth for the environment.

        The grid has the following coordinates:
        (0,0)------------------(1,0)------------------(N,0)
        |                                                  |
        |                                                  |
        (0,N)------------------(1,N)------------------(N,N)
        """
        # General env parameters
        self.map = map
        self.OBST = OBST    # if there is any obstacles
        self.BS = BS        # if there is any Base Station
        self.reward_type = reward_type
        self.is_slippery = is_slippery
        self.transition_mode = TransitionMode.stochastic if is_slippery else TransitionMode.deterministic
        self.task = task
        self.collision_radius = collision_radius
        self.bs_radius = bs_radius
        if self.task == "encircle_target":
            self.desired_orientations = desired_orientations
            self.desired_distances = desired_distances
            self.optimal_view = optimal_view
            self.total_bandwidth = total_bandwidth
            self.agents_reached_goal = {agent: False for agent in agents_pos}
        self.rng = None
        self.goal_rew = 100.
        self.agent_collision = False
        self.agent_collision_rew = 0    # -1.
        self.wall_rew = 0
        self.hole_rew = 0

        # Grid Topology
        if self.BS:
            self.holes, self.goals, self.base_stations = parse_map_emoji(self.map)
            self.base_stations = [self.frame2center(np.array(bs, dtype=np.float32)) for bs in self.base_stations]
        else:
            self.holes, self.goals = parse_map_emoji(self.map)
        self.size = self.map.count("\n") - 1 # for now we consider square maps
        self.num_goals = len(self.goals)
        self.grid_height = self.size
        self.grid_width = self.size
        self._cell_size = 100

        # Environment parameters
        self.goal = self.frame2center(np.array(list(self.goals.values())[0]))
        self.observation_space = Box(low=-1, high=self.grid_width, shape=(4 + len(agents_pos),))
        self.action_space = Box(low=-0.4, high=0.4, shape=(2,))

        # Reward related stuff
        self._max_episode_steps = max_episode_steps

        # agents related stuff
        for agent in agents_pos:
            if type(agents_pos[agent]) != np.array:
                agents_pos[agent] = np.array(agents_pos[agent])
        self.agents_initial_pos = deepcopy(agents_pos)  # useful for reset
        assert len(self.agents_initial_pos) > 0, "At least one agent is required"
        for agent, pos in self.agents_initial_pos.items():
            assert len(pos) == 2, f"Agent {agent} position must be a list of two elements"
            assert pos[0] >= 0 and pos[0] < self.grid_width, f"Agent {agent} x position out of bounds"
            assert pos[1] >= 0 and pos[1] < self.grid_height, f"Agent {agent} y position out of bounds" 
            for hole in self.holes:
                assert not self.is_inside_cell(pos, hole), f"Agent {agent} position inside a hole"
        self.agents = list(self.agents_initial_pos.keys())
        self.rewards = {agent: 0 for agent in self.agents}
        self.trajectories = {agent: [] for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.switch_reward = {agent: False for agent in self.agents}  # when using reward_type == RewardType.model, this is used to switch the reward from the model to the real reward
        # Rendering stuff
        self.is_pygame_initialized = False
        self.frames = []
        self.is_rendered = is_rendered
        self.is_display = is_display
        if self.is_rendered:
            self.init_render()
        
        # Load Q-table
        if self.reward_type == RewardType.model:
            self.values = self.load_values()

    def step(self, actions: dict[str, list[float, float]]) -> Tuple[list, dict, dict, dict, dict]:
        """
        The agents take a step in the environment.
        actions: dict[str, list[float, float]]: Actions for each agent.
        actions are [dx, dy] where positive dx is right and positive dy is up.
        """
        new_pos = {agent: np.zeros((2,), dtype=np.float32) for agent in self.agents}
        # iterate over the agents
        new_observations = {}
        # reset rewards
        self.rewards = {agent: 0 for agent in self.agents}
        for agent, action in actions.items():
            if self.terminations[agent]:
                # absorbing state for terminated agents
                new_observations[agent] = np.array([-1.0, -1.0], dtype=np.float32)
                self.rewards[agent] = 0.0
                # remove the agent from the new positions
                del new_pos[agent]
                continue
            # The policy activation function is the tanh function, the action space is between -1 and 1
            action = np.clip(action, self.action_space.low, self.action_space.high)

            new_pos[agent][0] = self.observations[agent][0] + action[0]
            new_pos[agent][1] = self.observations[agent][1] + action[1]

            if self.is_slippery:
                new_pos[agent][0] += np.random.normal(0, 0.01)
                new_pos[agent][1] += np.random.normal(0, 0.01)
        # update positions 
        rewards, logs = self.update_positions(new_pos)
        # update self.rewards
        for agent in rewards:
            self.rewards[agent] += rewards[agent]

        if self.task == "encircle_target":
            rewards, logs = self.reward_function_upstream()
        elif self.task == "reach_target":
            rewards, logs = self.reward_function()

        for agent in rewards:
            self.rewards[agent] += rewards[agent] 

        self.num_steps += 1
        # Check for maximum steps termination
        if self.num_steps >= self._max_episode_steps:
            for agent in logs:
                logs[agent]["MAX STEPS"] = True
                self.truncations[agent] = True


        final_observations = deepcopy(self.observations)
        for agent in new_observations:
            final_observations[agent] = new_observations[agent]
        

        return final_observations, deepcopy(self.rewards), deepcopy(self.terminations), deepcopy(self.truncations), logs

    def reset(self, seed=42) -> Tuple[list, dict]:
        # super().reset(seed=seed)
        if self.rng is None:
            self.rng = np.random.default_rng(seed)
        self.num_steps = 0
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.rewards = {agent: 0 for agent in self.agents}
        self.trajectories = {agent: [] for agent in self.agents}
        self.observations = deepcopy(self.agents_initial_pos)
        self.switch_reward = {agent: False for agent in self.agents}
        # Reset the agents' positions. Uncomment the following for random positions
        # agents_position = []
        # self.observations = {}
        # for agent in self.agents:
        #     valid_position = False
        #     while not valid_position:
        #         x = self.rng.integers(0, self.grid_width)
        #         y = self.rng.integers(0, self.grid_height)
        #         # Check if the position is valid (not on a wall)
        #         valid_position = self.check_initial_position(x, y, agents_position)

        #     self.observations[agent] = np.array([x, y])
        #     agents_position.append((x, y))
        self.frames = []
        return deepcopy(self.observations), {}
    
    def check_initial_position(self, x: int, y: int, agents_position: list[tuple[int, int]]) -> bool:
        """
        Check if the initial position is valid (not on a wall and not occupied by another agent).
        """
        for hole in self.holes:
            if self.is_inside_cell([x, y], hole):
                return False
        if self.is_inside_cell([x, y], self.goal):
            return False
        if (x,y) in agents_position:
            return False
        return True

    def render(self, mode="human"):
        """
        Renders the environment with the given observations.
        """
        if not self.is_pygame_initialized and self.is_rendered:
            self.init_render()
            self.is_pygame_initialized = True
            self.trajectories = {agent: [] for agent in self.agents}
            self.frames = []
        # Remembed that when blit the images, the origin is at the top left corner of the coordinate passed
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
        # Draw the goals
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
            # Draw the circular crown around the goal
            if self.task == "encircle_target":
                surface = pygame.Surface((self._cell_size * 4, self._cell_size * 4), pygame.SRCALPHA)
                # Draw the outer circle (radius 2)
                pygame.draw.circle(
                surface,
                (0, 0, 255, 50),  # Blue color with transparency
                (self._cell_size * 2, self._cell_size * 2),
                int(self.desired_distances["a1"][1] * self._cell_size)
                )
                # Draw the inner circle (radius 1)
                pygame.draw.circle(
                surface,
                (0, 0, 0, 0),  # Fully transparent
                (self._cell_size * 2, self._cell_size * 2),
                int(self.desired_distances["a1"][0] * self._cell_size)
                )
                # Cut out the inner circle to create the crown effect
                pygame.draw.circle(
                surface,
                (0, 0, 255, 0),  # Transparent color
                (self._cell_size * 2, self._cell_size * 2),
                int(self.desired_distances["a1"][0] * self._cell_size),
                width=0
                )
                # Position the surface centered on the goal
                goal_x, goal_y = self.goal
                self.screen.blit(surface, (
                int(goal_x * self._cell_size - 2 * self._cell_size),
                int(goal_y * self._cell_size - 2 * self._cell_size)
                ))

        # Draw the Base Stations
        if self.BS: 
            for bs in self.base_stations:
                bs_x, bs_y = bs
                bs_x = bs_x * self._cell_size - self.base_station_img.get_width() // 2
                bs_y = bs_y * self._cell_size - self.base_station_img.get_height() // 2
                self.screen.blit(self.base_station_img,  (bs_x, bs_y))
                # Draw a circle around the base station to indicate its radius
                surface = pygame.Surface((self._cell_size * 2*self.bs_radius, self._cell_size * 2*self.bs_radius), pygame.SRCALPHA)
                pygame.draw.circle(
                    surface,
                    (0, 255, 0, 80),  # Green color with transparency
                    # draw the circle in the middle of the surface
                    (self._cell_size * self.bs_radius, self._cell_size * self.bs_radius),
                    int(self.bs_radius * self._cell_size)
                )
                # the surface should be centered on the base station
                self.screen.blit(surface, (bs_x + (self._cell_size//2)- (self._cell_size*self.bs_radius), bs_y + (self._cell_size//2) - (self._cell_size*self.bs_radius)))

        # Draw the agents
        for agent in self.observations:
            x, y = self.observations[agent]
            agent_x = x * self._cell_size - self.agent_img.get_width() // 2
            agent_y = y * self._cell_size - self.agent_img.get_height() // 2
            self.screen.blit(self.agent_img, (agent_x, agent_y))
            # Draw a circle around the agent to indicate its collision radius
            surface = pygame.Surface((self._cell_size * 2, self._cell_size * 2), pygame.SRCALPHA)
            pygame.draw.circle(
                surface,
                (255, 255, 0, 125),  # Yellow color with transparency
                (self._cell_size, self._cell_size),
                int(self.collision_radius * self._cell_size)
            )
            self.screen.blit(surface, (int(x * self._cell_size - self._cell_size), int(y * self._cell_size - self._cell_size)))
            # Render the agent's ID on the screen
            agent_text = self.font.render(agent, True, (0, 0, 0))  # Black color
            text_rect = agent_text.get_rect(center=(int(x * self._cell_size), int(y * self._cell_size)))
            self.screen.blit(agent_text, text_rect)
            # Draw the trajectory
            if self.trajectories[agent]:
                for point in self.trajectories[agent]:
                    traj_x, traj_y = point
                    traj_x = traj_x * self._cell_size
                    traj_y = traj_y * self._cell_size
                    pygame.draw.circle(self.screen, (255, 0, 0), (traj_x, traj_y), 5)

        if not self.is_display:
            image_data = pygame.surfarray.array3d(self.screen)
        else:
            image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        image_data = image_data.transpose([1, 0, 2])
        self.frames.append(image_data)
        
        if self.is_display and mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(60)
        
        if mode == "rgb_array":
            return image_data
            


    def reward_function(self) -> Tuple[dict, dict]:
        rewards = {agent: 0 for agent in self.agents}
        log = {agent: {} for agent in self.agents}
        # Check if agents are inside the goal. In this case we ignore collisions
        for agent_num1, agent in enumerate(self.observations):
            if self.reward_type == RewardType.dense:
                goal_distance = np.linalg.norm(self.observations[agent] - self.goal)
                rewards[agent] = -goal_distance
            elif self.reward_type == RewardType.model:
                i, j = self.frame2matrix(self.observations[agent])
                rewards[agent] = self.values[agent][i, j] 
            elif self.reward_type == RewardType.sparse:
                if self.is_inside_cell(self.observations[agent], self.goal):
                    log[agent]["GOAL REACHED"] = True
                    rewards[agent] = self.goal_rew
                    self.terminations[agent] = True
                    continue # if the agent is inside the goal, we don't want to check for collisions

        return rewards, log


    def reward_function_upstream(self) -> Tuple[dict, dict]:
        ###########################################################################################
        # When computing angles, remember that the origin is at the top left corner of the screen #
        # we need to flip the y axis                                                              #
        ###########################################################################################
        rewards = {agent: 0 for agent in self.agents}
        log = {agent: {} for agent in self.agents}
        # beta indicates if the agent is in the desired distance from the goal
        agents_beta = self.compute_agents_beta
        agent_angle_from_goal = {}
        # if we are using the model reward type, agents will receive the model reward reward up until they are close enough to the goal
        if self.reward_type == RewardType.model:
            for agent in self.agents:
                # self.compute_switch_reward(agent)
                # if self.switch_reward[agent]:
                #     continue
                # else:
                i, j = self.frame2matrix(self.observations[agent])
                rewards[agent] = self.values[agent][i, j] / 150 # TODO: scale down the reward, let's see if this works


        for agent in agents_beta:
            goal_pos = self.goal
            if all(goal_pos == self.observations[agent]):
                # special case where the agent is inside the goal
                angle_from_goal = 0
            else:
                # arctan2 is in radians and returns values in [-pi, pi] with respect to the x axis
                angle_from_goal = np.arctan2(-self.observations[agent][1] + goal_pos[1], self.observations[agent][0]- goal_pos[0])
                angle_from_goal = (np.degrees(angle_from_goal) + 360) % 360  # Convert to degrees and normalize to [0, 360)
                agent_angle_from_goal[agent] = angle_from_goal
                # starts with the Streaming reward
                # phi is the angular distance from the desired view.
                delta_phi = smallest_positive_angle(angle_from_goal, self.optimal_view)
                # compute cosine of the angle if the angle is in the range [0, 180] otherwise it is 0
                if 0 <= delta_phi <= 90:
                    rho = np.cos(np.radians(delta_phi))
                else:
                    rho = 0
                distance_from_bs = np.linalg.norm(self.observations[agent] - self.base_stations[0])/ self.bs_radius
                # eta is the spectral efficiency
                eta = self.compute_spectral_efficiency(distance_from_bs)
                # theta is the spectral efficiency w.r.t the upstream bandwidth and the number of UAVs
                theta = eta * (self.total_bandwidth / len(agents_beta))
                # r1 is the rewards that balance spectral efficiency and distance from the goal 
                r1 = rho*theta
                rewards[agent] += float(r1) / 100 # TODO: manual scaling, implement a better scaling

            # check if the angular distance between the agents is enough
            angular_distance = True if len(agent_angle_from_goal) > 1 else False
            for agent1 in agent_angle_from_goal:
                for agent2 in agent_angle_from_goal:
                    if agent1 == agent2:
                        continue
                    angle_diff = smallest_positive_angle(agent_angle_from_goal[agent1], agent_angle_from_goal[agent2])
                    if angle_diff < (360 / (len(agents_beta)+1)):
                        angular_distance = False
            if angular_distance:
                for agent in agent_angle_from_goal:
                    rewards[agent] += 5

            # else:
            #     if self.desired_distance[agent][0] < distance_from_goal < self.desired_distance[agent][1]:
            #         if self.desired_orientation[agent][0] < angle_from_goal < self.desired_orientation[agent][1]:
            #             rewards[agent] = self.goal_rew
            #             log[agent]["GOAL REACHED"] = True
            #             self.terminations[agent] = True
            #             continue # if the agent is inside the goal, we don't want to check for collisions
        return rewards, log

    def update_positions(self, new_pos) -> Tuple[dict, dict]:
        """
        Check for collisions with walls, holes and other agents.
        don't update the agent positions if they collide
        :param new_pos: dict[str, list[float, float]]: New positions of the agents.
        :return: Tuple[dict, dict]: rewards and logs for each agent.
        """
        rewards = {agent: 0 for agent in self.agents}
        log = {agent: {} for agent in self.agents}
        # don't update the agent positions if they collide 
        update_agent = {agent: True for agent in new_pos}
        for agent_num1, agent in enumerate(new_pos):
            # check for collisions with walls
            if new_pos[agent][0] <= 0 or new_pos[agent][0] >= self.size:
                update_agent[agent] = False
                rewards[agent] += self.wall_rew
                log[agent]["WALL"] = True
            if new_pos[agent][1] <= 0 or new_pos[agent][1] >= self.size:
                update_agent[agent] = False
                rewards[agent] += self.wall_rew
                log[agent]["WALL"] = True
            # check for collisions with holes
            for hole in self.holes:
                if self.is_inside_cell(new_pos[agent], hole):
                    update_agent[agent] = False
                    rewards[agent] += self.hole_rew
                    log[agent]["HOLE"] = True                   
            if self.agent_collision:
                # check for collisions with other agents if the agent is active
                for agent_num2, agent2 in enumerate(new_pos):
                    if agent_num1 >= agent_num2:
                        continue
                    else:
                        if check_uav_collision(new_pos[agent], new_pos[agent2], self.collision_radius):
                            rewards[agent] += self.agent_collision_rew
                            rewards[agent2] += self.agent_collision_rew
                            log[agent]["COLLISION"] = True
                            log[agent2]["COLLISION"] = True
                            update_agent[agent] = False
                            update_agent[agent2] = False
        
        # update agent positions 
        for agent in new_pos.keys():
            if update_agent[agent]:
                self.observations[agent] = new_pos[agent]
                self.trajectories[agent].append(self.observations[agent])

        return rewards, log

    def init_render(self):
        """
        Initialize the Pygame environment.
        It loads the images and sets up the display.
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
        base_station_img_path = IMAGE_DIR / "base_station.png"
        self.base_station_img = scale(
            load(base_station_img_path), (self._cell_size, self._cell_size)
        )

    def quit_render(self):
        """
        Quit the Pygame environment.
        """
        pygame.quit()

    def load_values(self) -> dict[np.ndarray]:
        """
        Load the Q-tables from the file system.
        """
        qtables = np.load(f"{QTABLE_DIR}/{self.transition_mode.name}/single_agent/qtable_{self.size}_obstacles_{self.OBST}.npz")
        return qtables
    
    def is_inside_cell(self, pos: np.ndarray, cell: np.ndarray) -> bool:
        """
        Check if a position is inside a cell.
        """
        # cell_coord = self.grid2frame(cell)
        cell_coord = cell
        inside = True
        if pos[0] < cell_coord[0]:
            inside = False  # left
        if pos[0] > cell_coord[0] + 1:
            inside = False  # right
        if pos[1] < cell_coord[1]:
            inside = False  # bottom
        if pos[1] > cell_coord[1] + 1:
            inside = False  # top
        return inside

    def frame2matrix(self, frame_pos: np.ndarray) -> np.ndarray:
        """
        Convert a frame (cartesian) position to a matrix index.
        The frame is defined as a cartesian coordinate system with the origin at the top left corner. A frame position repesent the center of the cell.
        Matrix is the matrix with the Q-table values. the origin is at the bottom left corner and it requires integer values
        The grid instead has the y axis inverted, with respect to the frame and it also represents the center of the cell.
        """
        x, y = self.frame2grid(frame_pos)

        # Inverting the coordinates
        # x actually represents the columns and y the rows
        indices = np.floor(np.array([y, x])).astype(int)

        return indices

    def frame2grid(self, frame_pos: np.ndarray) -> np.ndarray:
        """
        Convert a frame (cartesian) position to a grid position.
        The frame is defined as a cartesian coordinate system with the origin at the top left corner. A frame position repesent the center of the cell.
        The grid is defined as a matrix with the origin at the bottom left corner (y axis inverted).
        """
        assert len(frame_pos) == 2
        x, y = frame_pos

        # Flipping y axis
        # y = self.size - y

        cell = np.array([x, y])

        return cell

    def grid2frame(self, grid_pos: np.ndarray) -> np.ndarray:
        """
        Convert a grid position to a frame position.
        The grid is defined as a matrix with the origin at the bottom left corner and 
        The frame is defined as a cartesian coordinate system with the origin at the top left corner.
        """
        assert len(grid_pos) == 2
        x, y = grid_pos
        # Flipping y axis
        y = self.size - y

        # Centering the position
        # x += 0.5
        # y -= 0.5

        return np.array([x, y])
    
    def frame2center(self, grid_pos: np.ndarray) -> np.ndarray:
        """
        Convert a grid position to its center position.
        """
        assert len(grid_pos) == 2
        x, y = np.floor(grid_pos).astype(int)
        # Centering the position
        x += 0.5
        y += 0.5

        return np.array([x, y])


    def render_episode(self, agents: dict, max_steps: int = None):
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
            
            self.render()
            if all(truncated.values()):
                break
    
    def render_episode_goal_mdp(self, agents_model: dict, max_steps: int = None):
        """
        Renders an episode with the given agents.
        Since it's a goal MDP, the observations will be dictionary including the following keys:
        - observation: the current observation of the agent
        - achieved_goal: the current achieved goal of the agent
        - desired_goal: the current desired goal of the agent
        Args:
            agents (list): List of agent positions.
            max_steps (int): Maximum number of steps in the episode.
        """
        observations,_ = self.reset()
        if max_steps is None:
            max_steps = self._max_episode_steps
        for step in range(max_steps):
            with torch.no_grad():
                goal_mdp_observations = {agent: {} for agent in self.agents}
                actions = {}
                for agent in self.agents:
                    goal_mdp_observations[agent]['observation'] = torch.tensor(observations[agent], dtype=torch.float32).unsqueeze(0)
                    goal_mdp_observations[agent]['achieved_goal'] = torch.tensor(observations[agent], dtype=torch.float32).unsqueeze(0)
                    goal_mdp_observations[agent]['desired_goal'] = torch.tensor(self.goal, dtype=torch.float32).unsqueeze(0)
                    action, _ = agents_model[agent].predict(goal_mdp_observations[agent], deterministic=True)
                    actions[agent] = action.squeeze(0)
            observations,  rewards,  terminated, truncated, _ = self.step(actions)            
            
            self.render()
            if all(truncated.values()) or all(terminated.values()):
                break
    
    def save_episode(self, episode, name="uav_cont"):
        """Save the episode in a directory."""

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



    def compute_spectral_efficiency(self, x_value: float) -> float:
        """
        Perform stepwise linear interpolation to find the y value for a given x value.

        Args:
            x_value (float): The x value for which to find the corresponding y value.
        Returns:
            float: The interpolated y value.
        """
        x = [0.1, 0.13, 0.16, 0.19, 0.22, 0.25, 0.28, 0.31, 0.34, 0.37, 0.4, 0.43, 0.46, 0.49, 0.52, 0.55]
        y = [10., 9.10, 8.20, 7.40, 6.60, 5.90, 5.30, 4.80, 4.30, 3.90, 3.60, 3.30, 3., 2.80, 2.60, 2.40]  
        # If x_value is outside the range, return the closest boundary value
        if x_value < x[0]:
            return y[0]
        elif x_value >= x[-1]:
            return y[-1]

        for i in range(len(x) - 1):
            if x[i] <= x_value < x[i + 1]:
                # Perform linear interpolation
                slope = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
                return y[i] + slope * (x_value - x[i])
            
    def compute_streaming_reward(self, x_value: float) -> float:
        raise NotImplementedError("Streaming reward is not implemented yet.")


    def compute_angular_relevance(self, uav_pos: np.ndarray):
        """
        Compute the angular relevance of the UAV position with respect to the goal.

        Args:
            uav_pos (np.ndarray): Position of the UAV as a numpy array [x, y].

        Returns:
            float: The angular relevance value.
        """
        goal_pos = self.goal
        angle_from_goal = np.arctan2(goal_pos[1] - uav_pos[1], goal_pos[0] - uav_pos[0])
        angle_from_goal = (np.degrees(angle_from_goal) + 360) % 360
    
    # function parameter decorator
    @property
    def compute_agents_beta(self):
        beta = []
        for agent in self.agents:
            distance_from_goal = np.linalg.norm(self.observations[agent] - self.goal)
            if self.desired_distances[agent][0] < distance_from_goal < self.desired_distances[agent][1]:
                beta.append(agent)
        return beta
    
    def compute_switch_reward(self, agent: str) -> float:
        """
        Compute the distance from the goal for a given agent.

        Args:
            agent (str): The agent's identifier.

        Returns:
            float: The distance from the goal.
        """

        distance_from_goal = np.linalg.norm(self.observations[agent] - self.goal)
        if distance_from_goal < self.desired_distances[agent][1]:
            self.switch_reward[agent] = True

def check_uav_collision(pos1: np.ndarray, pos2: np.ndarray, r: float) -> bool:
    """
    Check if two circles with centers pos1 and pos2 and radius r intersect.

    Args:
        pos1 (np.ndarray): Position of the first circle's center as a numpy array [x, y].
        pos2 (np.ndarray): Position of the second circle's center as a numpy array [x, y].
        r (float): Radius of both circles.

    Returns:
        bool: True if the circles intersect, False otherwise.
    """
    distance_squared = np.sum((pos2 - pos1) ** 2)
    return distance_squared <= (2 * r) ** 2

def smallest_positive_angle(angle1, angle2):
    """
    Compute the smallest angle between two angles in degrees.

    Args:
        angle1 (float): First angle in degrees.
        angle2 (float): Second angle in degrees.

    Returns:
        float: The smallest angle between the two angles in degrees.
    """
    diff = (angle2 - angle1) % 360
    if diff > 180:
        diff = 360 - diff
    return diff


if __name__ == "__main__":
    from src.environments.maps import MAPS_FREE
    from src.environments.maps import MAPS_OBST
    import time
    map_size = 10
    map = MAPS_OBST[map_size]
    # Create the environment
    env = MultiAgentContinuousUAV(
        map=map,
        agents_pos={"a1": [2.5, 8.5], "a2": [3.5, 5.5], 'a3': [1., 1.]},
        OBST=True,
        reward_type=RewardType.sparse,
        max_episode_steps=50,
        task="reach_target",
        is_rendered=True,
        is_display=True,
    )

    # Reset the environment
    observations, _ = env.reset()

    # Run the environment with random actions
    for _ in range(env._max_episode_steps):
        actions = {agent: env.action_space.sample() for agent in env.agents}
        observations, rewards, terminations, truncations, info = env.step(actions)
        env.render()
        print(info)
        time.sleep(0.1)
        if all(terminations.values()) or all(truncations.values()):
            break

    # Quit rendering
    env.quit_render()