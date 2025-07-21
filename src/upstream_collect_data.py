"""
Given a trained model, this script loads the model and collects the following data:
- UAVs’ Throughput over time
- UAVs’ Angular distance from target over time
- UAVs’ Euclidean distance from target over time
- UAVs’ Speed over time
- UAVs’ Angular speed over time
- UAVs’ Euclidean speed over time
- UAVs’ Angular acceleration over time
- UAVs’ Euclidean acceleration over time

and saves the data in a CSV file.
"""

import wandb
import pandas as pd
from typing import List
from src.utils.paths import ROOT_DIR, CONFIG_DIR, SRC_DIR
from BenchMARL.benchmarl.experiment import Experiment, ExperimentConfig
from BenchMARL.benchmarl.algorithms import MappoConfig
from BenchMARL.benchmarl.environments.customenv.common import (
    MultiAgentContinuousUAVTasks
)
from BenchMARL.benchmarl.models.mlp import MlpConfig
from BenchMARL.benchmarl.hydra_config import reload_experiment_from_file
from BenchMARL.benchmarl.experiment.callback import Callback
from tensordict import TensorDict, TensorDictBase


class DataCollectionCallback(Callback):

    def on_evaluation_end(self, rollouts: List[TensorDictBase]):
        raise NotImplementedError("This callback is not implemented yet.")


# wandb.init(mode="disabled")  # Disable Weights & Biases logging

# RUNS = {
#     "500m_20Mhz_4UAVs": "src/wandb_tests/mappo_task_1_mlp__2a65c6c0_25_07_20-16_33_40"
# }

def main(seed=4):
    # Define the directory containing the trained models
    run_name = "500m_20Mhz_5UAVs"
    model_file = SRC_DIR / "wandb_tests" / run_name / "checkpoints" / "checkpoint_1000000.pt"


    # experiment_config_file =  CONFIG_DIR / "experiments" / "custom_experiment_config.yaml"
    # experiment_config = ExperimentConfig.get_from_yaml(experiment_config_file)
    # task_file = CONFIG_DIR / "tasks" / "custom_task_config.yaml"
    # task = MultiAgentContinuousUAVTasks.task_1.get_from_yaml(path=task_file)
    # mappo_config_file = CONFIG_DIR / "algorithms" / "custom_mappo_config.yaml"
    # algorithm_config = MappoConfig.get_from_yaml(mappo_config_file)
    # model_config_file = CONFIG_DIR / "models" / "custom_mlp_config.yaml"
    # model_config = MlpConfig.get_from_yaml(model_config_file)
    # critic_model_config_file = CONFIG_DIR / "models" / "custom_mlp_config.yaml"
    # critic_model_config = MlpConfig.get_from_yaml(critic_model_config_file)
    # experiment = Experiment(
    #     task=task,
    #     algorithm_config=algorithm_config,
    #     model_config=model_config,
    #     critic_model_config=critic_model_config,
    #     seed=seed,
    #     config=experiment_config,
    # )


    # Now we tell it where to resume from
    restored_experiment = reload_experiment_from_file(model_file)  # Restore from first checkpoint
    # change some configs
    # restored_experiment.callbacks.append(DataCollectionCallback())
    # restored_experiment.config.loggers = [] 

    # evaluate
    restored_experiment.evaluate()
    data = restored_experiment.test_env.env.stats
    # Flatten the dictionary
    flattened_data = {}
    for agent, features in data.items():
        for feature, values in features.items():
            key = f"{agent}_{feature}"
            flattened_data[key] = values

    df = pd.DataFrame(flattened_data)
    # Save the final dataframe to a CSV file
    csv_dir = ROOT_DIR / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    output_file = csv_dir / f"{run_name}.csv"
    df.to_csv(output_file, index=False)

    print(f"Data collected and saved to {output_file}")

if __name__ == "__main__":
    print("Collecting data from trained models...")
    main(seed=4)