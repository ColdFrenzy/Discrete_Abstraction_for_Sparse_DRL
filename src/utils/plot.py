import os
# Get IS_RENDER from the environment or default to False
# IS_RENDER = os.getenv("IS_RENDER", "False").lower() == "true"
# # Set the backend conditionally
# if not IS_RENDER:
#     import matplotlib
#     matplotlib.use('Agg')  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import os
import numpy as np
from src.utils.paths import ROOT_DIR
from scipy.interpolate import interp1d
from tensorboard.backend.event_processing import event_accumulator

def save_reward_plots(reward_dict, save_path="reward_plots"):
    """
    Salva un'immagine del grafico dell'andamento delle reward episodio per episodio per ogni agente.

    :param reward_dict: Dizionario con i nomi degli agenti come chiavi e le liste delle reward come valori.
    :param save_path: Percorso della cartella in cui salvare i grafici.
    """
    # Crea la directory se non esiste
    os.makedirs(save_path, exist_ok=True)

    # Crea un grafico per ogni agente
    for agent_name, rewards in reward_dict.items():
        file_path = os.path.join(save_path, f"reward_plot_{agent_name}.png")

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(rewards) + 1), rewards, marker="o", linestyle="-")
        # plt.title(f"Episodio vs Reward Totale - {agent_name}")
        # plt.title(f"{agent_name} reward")
        plt.title(f"reward")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.grid(True)
        plt.savefig(file_path)
        plt.close()  # Chiude il grafico per evitare problemi di memoria

        print(f"Grafico per {agent_name} salvato in: {file_path}")


def save_steps_plot(steps_list, save_path="plots", filename="steps_plot.png"):
    """
    Salva un'immagine del grafico dell'andamento degli step per episodio.

    :param steps_list: Lista degli step per ogni episodio.
    :param save_path: Percorso della cartella in cui salvare il grafico.
    :param filename: Nome del file immagine.
    """
    # Crea la directory se non esiste
    os.makedirs(save_path, exist_ok=True)

    # Imposta il percorso completo del file
    file_path = os.path.join(save_path, filename)

    # Crea il grafico
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(steps_list) + 1), steps_list, marker="o", linestyle="-")
    plt.title("Steps per episode")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.grid(True)
    plt.savefig(file_path)
    plt.close()  # Chiude il grafico per evitare problemi di memoria

    print(f"Grafico degli step salvato in: {file_path}")




def tensordboard_plot(tensorboard_path: str, save_path: str):
    # Initialize figure for plotting
    plt.figure(figsize=(10, 6))
    for algorithm_name, log_dir in tensorboard_path.items():
        all_rewards = []
        all_steps = []
        run_dirs = [os.path.join(log_dir, run) for run in os.listdir(log_dir)]

        # Read the event files from each run directory
        for run in run_dirs:
            event_acc = event_accumulator.EventAccumulator(run)
            event_acc.Reload()


            # Extract step and reward values
            steps = [event.step for event in event_acc.Scalars('rollout/ep_len_mean')]
            reward_values = [event.value for event in event_acc.Scalars('rollout/ep_len_mean')]

            # Store rewards of each run
            all_steps.append(np.array(steps))
            all_rewards.append(np.array(reward_values))

        # Find the unique set of steps across all runs
        all_unique_steps = np.unique(np.concatenate(all_steps))
        ###############
        # REWARD PLOT #
        ###############
        aligned_rewards = []
        for steps, rewards in zip(all_steps, all_rewards):
            # Create an interpolation function based on the current run's steps
            interp_func = interp1d(steps, rewards, kind='linear', bounds_error=False, fill_value="extrapolate")
            
            # Interpolate the rewards to match the unique steps
            new_rewards = interp_func(all_unique_steps)
            
            aligned_rewards.append(new_rewards)
        # Convert the list of rewards to a NumPy array
        aligned_rewards = np.array(aligned_rewards)
        # Compute the mean and std across all runs (along axis=0 corresponds to different seeds/runs)
        mean_rewards = np.mean(aligned_rewards, axis=0)
        std_rewards = np.std(aligned_rewards, axis=0)
    
        plt.plot(all_unique_steps, mean_rewards, label=f"{algorithm_name}", lw=2)
        plt.fill_between(all_unique_steps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2, )
    
    plt.xlabel('Steps', fontdict={'size': 23})
    plt.ylabel('Ep_Len_Mean', fontdict={'size': 23})
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=13, loc='upper left')
    # plt.savefig("10x10_single_win_rate_plot_new.png", dpi=600)


    # Save the plot to a file (e.g., PNG)
    plt.savefig(save_path)

    # Close the plot to free up memory
    plt.close()
    # steps = [event.step for event in event_acc.Scalars('reward')]  # Assuming step is consistent across runs

    # # Create a SummaryWriter to log the aggregated data
    # with SummaryWriter(aggregated_log_dir) as writer:
    #     # Log mean and std as scalars
    #     for i, (mean, std) in enumerate(zip(mean_rewards, std_rewards)):
    #         writer.add_scalar('mean_reward', mean, steps[i])
    #         writer.add_scalar('std_reward', std, steps[i])

    # print(f"Aggregated data logged to: {aggregated_log_dir}")




if __name__ == "__main__":
    name = "10x10"
    tb_paths = {"SAC": f"sa_sac_uav_tensorboard/{name}",
               "SAC_HER": f"sa_her_sac_uav_tensorboard/{name}",
               "SAC_HR": f"sa_sac_hr_uav_tensorboard/{name}",
               "SAC_RELAX": f"sa_sac_dense_uav_tensorboard/{name}"}
    img_plot = ROOT_DIR / "plots" / f"{name}_single_ep_len_mean_plot.png"
    tensordboard_plot(tensorboard_path=tb_paths, save_path=img_plot)