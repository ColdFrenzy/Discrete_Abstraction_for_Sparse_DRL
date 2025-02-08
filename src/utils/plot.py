import os
# Get IS_RENDER from the environment or default to False
IS_RENDER = os.getenv("IS_RENDER", "False").lower() == "true"
# Set the backend conditionally
if not IS_RENDER:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import os

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