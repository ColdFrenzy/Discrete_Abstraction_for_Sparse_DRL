#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
import pathlib
import os
import numpy as np
from BenchMARL.benchmarl.algorithms import MappoConfig, IppoConfig, QmixConfig, MasacConfig
from BenchMARL.benchmarl.environments.customenv.common import (
    MultiAgentContinuousUAVTasks
)
from BenchMARL.benchmarl.environments import VmasTask
from BenchMARL.benchmarl.experiment import Experiment, ExperimentConfig
from BenchMARL.benchmarl.models.mlp import MlpConfig
from BenchMARL.benchmarl.models.lstm import LstmConfig
from src.environments.multi_agent.ma_cont_uav_env import check_initial_position
from src.utils.utils import parse_map_emoji
from src.utils.paths import CONFIG_DIR
from src.definitions import RewardType, TransitionMode
from src.value_function_computation import compute_value_function_single
from src.utils.heatmap import generate_heatmaps_numbers
from src.utils.paths import QTABLE_DIR
from src.environments.maps import MAPS_FREE, MAPS_FREE_BASESTATION, MAPS_OBST_BASESTATION, MAPS_OBST
# TODO: add to requirements pip install moviepy==1.0.3 , pip install benchmarl, pip install wandb[media]
TEST_CONFIG = False
USE_LSTM = False
USE_ALGO = "mappo"


def main(algos=["mappo", "mappo_hr"], seeds=[4,8,15,16,23,42], map_size=10, OBST=True, bs = True, training = True, compute_qtable=True):

    for algo in algos:
        for seed in seeds:
            map_size = int(map_size)
            training = training
            compute_qtable = compute_qtable
            is_slippery = False
            BS = bs  # Base Station
            transition_mode = TransitionMode.stochastic if is_slippery else TransitionMode.deterministic
            # root = project_root()
            NUM_EPISODES_DISCRETE = 40000
            OBST = OBST
            if OBST:
                map_name = MAPS_OBST[map_size]# if not BS else MAPS_OBST_BASESTATION[map_size]
            else:
                map_name = MAPS_FREE[map_size]# if not BS else MAPS_FREE_BASESTATION[map_size]
            # if BS:
            #     holes, goals, base_stations = parse_map_emoji(map_name)
            # else:
            #     holes, goals = parse_map_emoji(map_name)
            # goal = np.array(list(goals.values())[0]) + np.array([0.5, 0.5]) 
            
            
            if algo == "mappo":
                env_reward_type = RewardType.sparse 
            elif algo == "mappo_hr":
                env_reward_type = RewardType.model
            else:
                raise ValueError(f"Algorithm {algo} not supported")

            agents_pos = {"a1":  [0, 9], # [0.5, 0.5],
                    "a2":  [9, 0], # [1.5, 0.5],
                    "a3":  [0, 0], # [2.5, 0.5], 
                    # "a4":  [9, 9],
                    # "a5":  [0, 5],
                   }
            if compute_qtable:
                compute_value_function_single(map_name, size=map_size, OBST=OBST, agents=agents_pos, num_episodes=NUM_EPISODES_DISCRETE, gamma = 0.8, stochastic=is_slippery, save=True)
                
            if env_reward_type == RewardType.model:
                qtable = np.load(f"{QTABLE_DIR}/{transition_mode.name}/single_agent/qtable_{map_size}_obstacles_{OBST}_{len(agents_pos)}_agents.npz")
                # generate_heatmaps_numbers(qtable)

            if training:
                # Loads from "benchmarl/conf/experiment/base_experiment.yaml"
                if TEST_CONFIG:
                    experiment_config_file = CONFIG_DIR / "experiments" / "test_experiment_config.yaml"
                else:
                    experiment_config_file =  CONFIG_DIR / "experiments" / "custom_experiment_config.yaml"

                experiment_config = ExperimentConfig.get_from_yaml(experiment_config_file)

                task_file = CONFIG_DIR / "tasks" / "custom_task_config.yaml"
                task = MultiAgentContinuousUAVTasks.task_1.get_from_yaml(path=task_file)

                # Loads from "benchmarl/conf/algorithm/mappo.yaml"
                if USE_ALGO == "masac": # OFF Policy
                    masac_config_file = CONFIG_DIR / "algorithms" / "custom_masac_config.yaml"
                    algorithm_config = MasacConfig.get_from_yaml(masac_config_file)
                elif USE_ALGO == "mappo": # ON POLICY
                    mappo_config_file = CONFIG_DIR / "algorithms" / "custom_mappo_config.yaml"
                    algorithm_config = MappoConfig.get_from_yaml(mappo_config_file)
                else: 
                    raise ValueError(f"Algorithm {USE_ALGO} not supported")
                # Loads from "benchmarl/conf/model/layers/mlp.yaml"
                if USE_LSTM:
                    model_config_file = CONFIG_DIR / "models" / "custom_lstm_config.yaml"
                    model_config = LstmConfig.get_from_yaml(model_config_file)
                    critic_model_config_file = CONFIG_DIR / "models" / "custom_lstm_config.yaml"
                    critic_model_config = LstmConfig.get_from_yaml(critic_model_config_file)
                else:
                    model_config_file = CONFIG_DIR / "models" / "custom_mlp_config.yaml"
                    model_config = MlpConfig.get_from_yaml(model_config_file)
                    critic_model_config_file = CONFIG_DIR / "models" / "custom_mlp_config.yaml"
                    critic_model_config = MlpConfig.get_from_yaml(critic_model_config_file)

                experiment = Experiment(
                    task=task,
                    algorithm_config=algorithm_config,
                    model_config=model_config,
                    critic_model_config=critic_model_config,
                    seed=seed,
                    config=experiment_config,
                )
                experiment.run()


if __name__ == "__main__":
    main(algos=["mappo"], seeds=[4,8,15], map_size=10, OBST=True, bs = True, training = True, compute_qtable=False)
    # 4, 8, 15, 16, 23, 42