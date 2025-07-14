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
from src.utils.paths import ROOT_DIR, CONFIG_DIR
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


wandb.init(mode="disabled")  # Disable Weights & Biases logging

def main(seed=4):
    # Define the directory containing the trained models
    model_file = ROOT_DIR / "mappo_task_1_mlp__7a4b8afe_25_06_21-17_59_19" / "checkpoints" / "checkpoint_1000000.pt"

    # Now we tell it where to resume from
    restored_experiment = reload_experiment_from_file((model_file))  # Restore from first checkpoint
    # change some configs
    # restored_experiment.callbacks.append(DataCollectionCallback())
    restored_experiment.config.loggers = [] 

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
    output_file = ROOT_DIR / "collected_data.csv"
    df.to_csv(output_file, index=False)

    print(f"Data collected and saved to {output_file}")

if __name__ == "__main__":
    print("Collecting data from trained models...")
    main(seed=4)