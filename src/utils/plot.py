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
from scipy.stats import ttest_ind, shapiro, mannwhitneyu
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
        
        print(run_dirs)
        exit()
        
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
    # os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # plt.savefig(save_path)

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


def statistical_significance(tensorboard_path: str, threshold: float = 0.5):

    first_timestep =  {name: [] for name in tensorboard_path.keys()}
    for algorithm_name, log_dir in tensorboard_path.items():
        
        run_dirs = [os.path.join(log_dir, run) for run in os.listdir(log_dir)]
                
        # Read the event files from each run directory
        for run in run_dirs:
            event_acc = event_accumulator.EventAccumulator(run)
            event_acc.Reload()

            # Extract step and reward values
            steps = [event.step for event in event_acc.Scalars('custom/win_rate')]
            reward_values = [event.value for event in event_acc.Scalars('custom/win_rate')]

            consistent=False
            temp = -1
            # Find the first timestep from when the algorithm is always above the threshold
            for step, rew in zip(steps, reward_values):
                # Find the first timestep where the win rate is above the threshold
                if rew >= threshold and temp == -1: 
                    temp = step
                elif rew < threshold:
                    temp = -1
            # If the algorithm never reaches the threshold, temp will remain -1            
            first_timestep[algorithm_name].append(temp)

    print(f"{algorithm_name} first timestep: {first_timestep[algorithm_name]}")

    # access dictionary elements without knowing the keys
    algo_1 = list(first_timestep.keys())[0]
    algo_2 = list(first_timestep.keys())[1]
    stat1, p1 = shapiro(first_timestep[algo_1])
    stat2, p2 = shapiro(first_timestep[algo_2])

    print(f"Algo A normality p-value: {p1:.4f}")
    print(f"Algo B normality p-value: {p2:.4f}")

    # show_histogram(first_timestep[algo_2], first_timestep[algo_1])
    t_stat, p_value = ttest_ind(first_timestep[algo_1], first_timestep[algo_2], equal_var=False)

    alpha = 0.05
    if p_value < alpha:
        print("✅ There is a statistically significant difference between the two algorithms. the p-value is {:.7f}".format(p_value))
    else:
        print("❌ No statistically significant difference was found.")

    # suppose that both algorithms have no normal distribution, we can run non-parametric tests such as mann-whitney U test
    u_stat, p_mw = mannwhitneyu(first_timestep[algo_1], first_timestep[algo_2], alternative='two-sided') # alternative can be also 'greater' or 'less' depending on the hypothesis
    print(f"Mann–Whitney U test statistic: {u_stat}")
    print(f"Mann–Whitney U test p-value: {p_mw:.6f}")

    # bootstrap the difference of means
    mean_diff, lower_ci, upper_ci = bootstrap_diff_means(first_timestep[algo_1], first_timestep[algo_2])
    print(f"Bootstrap mean difference: {mean_diff:.3f}")
    print(f"{95}% CI: [{lower_ci:.3f}, {upper_ci:.3f}]")

    # compute cohen's d
    d = cohen_d(first_timestep[algo_1], first_timestep[algo_2])
    print(f"Cohen's d: {d:.4f}")

    # compute how more sample efficient in terms of percentage is SAC_HR compared to SAC
    mean_a = np.mean(first_timestep[algo_1])
    mean_b = np.mean(first_timestep[algo_2])
    # mean_b should be the faster algorithm (less timestep otherwise we get negative value), so we compute the percentage improvement of SAC_HR over SAC
    percentage_improvement = ((mean_a - mean_b) / mean_a) * 100


def bootstrap_diff_means(x, y, n_bootstrap=10000, ci=95):
    """used to compute the confidence interval of the difference of means between two samples x and y
    Args:
        x (array-like): First sample.
        y (array-like): Second sample.
        n_bootstrap (int): Number of bootstrap samples to draw.
        ci (float): Confidence interval percentage (default is 95).
    Returns:
        tuple: Mean difference, lower bound, and upper bound of the confidence interval.
    
    """
    diffs = []
    combined = np.concatenate([x, y])
    nx, ny = len(x), len(y)

    for _ in range(n_bootstrap):
        sample_x = np.random.choice(x, size=nx, replace=True)
        sample_y = np.random.choice(y, size=ny, replace=True)
        diffs.append(np.mean(sample_x) - np.mean(sample_y))

    lower = np.percentile(diffs, (100 - ci) / 2)
    upper = np.percentile(diffs, 100 - (100 - ci) / 2)
    mean_diff = np.mean(diffs)
    return mean_diff, lower, upper


def cohen_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx - 1)*np.var(x, ddof=1) + (ny - 1)*np.var(y, ddof=1)) / dof)
    d = (np.mean(x) - np.mean(y)) / pooled_std
    return d

def show_histogram(algo_a, algo_b):
    plt.figure(figsize= (10, 6))

    # Histogram
    plt.hist(algo_a, bins=8, alpha=0.7, label='SAC_HR')
    plt.hist(algo_b, bins=8, alpha=0.7, label='SAC')
    plt.title('Histogram of Convergence Times')
    plt.xlabel('Time to Converge')
    plt.legend()

    plt.show()

def show_separate_histogram(algo_a, algo_b):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].hist(algo_a, bins=8, alpha=0.7, color='steelblue')
    axs[0].set_title('SAC with gamma 0.94 Convergence Times')

    axs[1].hist(algo_b, bins=8, alpha=0.7, color='orange')
    axs[1].set_title('SAC with gamma 0.95 Convergence Times')

    for ax in axs:
        ax.set_xlabel('Time to Converge')
        ax.set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

def show_boxplot(algo_a, algo_b):
    plt.figure(figsize= (10, 6))

    # Boxplot
    plt.boxplot([algo_a, algo_b], labels=['SAC_HR', 'SAC'])
    plt.title('Boxplot of Convergence Times')
    plt.ylabel('Time to Converge')


if __name__ == "__main__":
    name = "7x7"
    tb_paths = {"SAC_gamma_99": f"sa_sac_uav_tensorboard/{name}",
               # "SAC_HER": f"sa_her_sac_uav_tensorboard/{name}",
               "SAC_gamma_98": f"sa_sac_uav_tensorboard_gamma98/{name}",
               "SAC_gamma_95": f"sa_sac_uav_tensorboard_gamma95/{name}",
               "SAC_gamma_97": f"sa_sac_uav_tensorboard_gamma97/{name}",
               "SAC_gamma_94": f"sa_sac_uav_tensorboard_gamma94/{name}",
               # "SAC_HR": f"sa_sac_hr_uav_tensorboard/{name}",
               # "SAC_RELAX": f"sa_sac_dense_uav_tensorboard/{name}"
               }
    img_plot = ROOT_DIR / "plots" / f"{name}_single_ep_len_mean_plot.png"


    elem = statistical_significance(tensorboard_path=tb_paths, threshold=0.9)
    # tensordboard_plot(tensorboard_path=tb_paths, save_path=img_plot)