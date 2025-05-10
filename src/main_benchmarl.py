#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
import pathlib
import os
from BenchMARL.benchmarl.algorithms import MappoConfig, IppoConfig, QmixConfig, MasacConfig
from BenchMARL.benchmarl.environments.customenv.common import (
    MultiAgentContinuousUAVTasks
)
from BenchMARL.benchmarl.environments import VmasTask
from BenchMARL.benchmarl.experiment import Experiment, ExperimentConfig
from BenchMARL.benchmarl.models.mlp import MlpConfig
from BenchMARL.benchmarl.models.lstm import LstmConfig
from src.utils.paths import CONFIG_DIR
# TODO: add to requirements pip install moviepy==1.0.3 , pip install benchmarl, pip install wandb[media]
TEST_CONFIG = False
USE_LSTM = False
USE_ALGO = "mappo"

def main():
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
        seed=123,
        config=experiment_config,
    )
    experiment.run()


if __name__ == "__main__":
    main()