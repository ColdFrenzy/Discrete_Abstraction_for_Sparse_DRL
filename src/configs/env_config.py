""" Create a configuration file for environment variables.
"""
from src.definitions import RewardType

class EnvConfig:
    map: str = 10,
    # agents_pos: str: x, y
    agents_pos: dict[str, list[float,float]] = {"a1":  [0.5, 9.5], # [0.5, 0.5],
                "a2":  [9.5, 0.5], # [1.5, 0.5],
                "a3":  [0.5, 0.5], # [2.5, 0.5], 
                },
    OBST: bool = True, # Obstacles
    BS: bool = True,    # Base station 
    reward_type = RewardType.model, # dense = 0, sparse = 1, model = 2
    max_episode_steps: int = 200,
    task: str = "encircle_target", # reach_target or encircle_target
    # for the desired orientation use the format [start_angle, end_angle]
    desired_orientations: dict[str, list[float,float]] = {"a1": [44.,46.], "a2": [134.,136.], "a3": [269., 271.]}, # None,
    desired_distances: dict[str, list[float,float]] = {"a1": [5, 15], "a2": [5, 15], "a3": [5, 15]}, # distance from the target in meters
    optimal_view = 180., # east
    total_bandwidth = 10, # in MHz
    bs_radius = 800, # in meters
    is_slippery: bool = False,
    is_rendered: bool = True,
    is_display: bool = False,
    collision_radius: float = 5, # in meters
    render_mode: str = "rgb_array",
