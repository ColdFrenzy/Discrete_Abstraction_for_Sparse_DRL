import os
# Get IS_RENDER from the environment or default to False
IS_RENDER = os.getenv("IS_RENDER", "False").lower() == "true"
# Set the backend conditionally
if not IS_RENDER:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for headless environments
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




def tensordboard_plot(tensorboard_path: str):
    # Define the path to your TensorBoard logs
    
    run_dirs = [os.path.join(tensorboard_path, run) for run in os.listdir(tensorboard_path)]

    # Initialize a list to store the reward values for each run
    all_rewards = []
    all_steps = []
    # Read the event files from each run directory
    for run in run_dirs:
        event_acc = event_accumulator.EventAccumulator(run)
        event_acc.Reload()

        # Assuming the tag is 'reward' in your TensorBoard logs

        # Extract step and reward values
        steps = [event.step for event in event_acc.Scalars('rollout/ep_rew_mean')]
        reward_values = [event.value for event in event_acc.Scalars('rollout/ep_rew_mean')]

        # Store rewards of each run
        all_steps.append(np.array(steps))
        all_rewards.append(np.array(reward_values))

    # Find the unique set of steps across all runs
    all_unique_steps = np.unique(np.concatenate(all_steps))


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

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(all_unique_steps, mean_rewards, label='Mean Reward', color='blue')
    plt.fill_between(all_unique_steps, mean_rewards - std_rewards, mean_rewards + std_rewards, color='blue', alpha=0.2, label='Std Deviation')
    plt.xlabel('Steps')
    plt.ylabel('Reward')
    plt.title('Mean and Std of Rewards Across Different Runs')
    plt.legend()
    plt.show()

    # steps = [event.step for event in event_acc.Scalars('reward')]  # Assuming step is consistent across runs

    # # Create a SummaryWriter to log the aggregated data
    # with SummaryWriter(aggregated_log_dir) as writer:
    #     # Log mean and std as scalars
    #     for i, (mean, std) in enumerate(zip(mean_rewards, std_rewards)):
    #         writer.add_scalar('mean_reward', mean, steps[i])
    #         writer.add_scalar('std_reward', std, steps[i])

    # print(f"Aggregated data logged to: {aggregated_log_dir}")




if __name__ == "__main__":
    tb_path = ROOT_DIR / "sac_uav_tensorboard" / "5x5" 
    tensordboard_plot(tensorboard_path=tb_path)