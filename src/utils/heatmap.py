import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from typing import Dict, List
from src.utils.paths import HEATMAPS_DIR

from src.algorithms.agent_rl import AgentRL
# Configurazioni
# dim_griglia = (10, 10)  # Dimensioni della griglia (righe, colonne)
# num_stati_rm = 3      # Numero di stati nella Reward Machine

# Caricamento della Q-table
# data = np.load('q_tables.npz')
# q_table = data['q_table_a3']  # Sostituisci 'q_table_a3' con la chiave corretta se necessario


def generate_heatmaps(q_table, dim_griglia, num_stati_rm):
    # Estrazione e Ridimensionamento dei Valori Q Massimi
    max_q_values = q_table.max(axis=1)
    reshaped_max_q_values = max_q_values.reshape((*dim_griglia, num_stati_rm))

    # Calcolo delle Azioni Ottimali
    optimal_actions = np.argmax(q_table, axis=1)
    reshaped_optimal_actions = optimal_actions.reshape((*dim_griglia, num_stati_rm))

    # Mappa dei codici delle azioni ai simboli corrispondenti
    action_symbols = {0: "↑", 1: "↓", 2: "←", 3: "→"}

    # Visualizzazione delle Heatmaps
    fig, axes = plt.subplots(1, num_stati_rm, figsize=(20, 6))

    # Assicurati che axes sia sempre un array
    if num_stati_rm == 1:
        axes = [axes]  # Rendi axes un array se c'è solo un subplot

    for i in range(num_stati_rm):
        sns.heatmap(
            reshaped_max_q_values[:, :, i],
            annot=np.vectorize(action_symbols.get)(reshaped_optimal_actions[:, :, i]),
            fmt="",
            cmap="coolwarm",
            ax=axes[i],
        )
        axes[i].set_title(f"RM State {i + 1}")
        axes[i].set_xlabel("Column")
        axes[i].set_ylabel("Row")

    plt.tight_layout()
    plt.show()


def save_heatmaps(q_table, dim_griglia, num_stati_rm, agent_id):
    # Crea la directory per salvare le heatmaps, se non esiste
    save_dir = HEATMAPS_DIR
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Estrazione e Ridimensionamento dei Valori Q Massimi
    max_q_values = q_table.max(axis=1)
    reshaped_max_q_values = max_q_values.reshape((*dim_griglia, num_stati_rm))

    # Calcolo delle Azioni Ottimali
    optimal_actions = np.argmax(q_table, axis=1)
    reshaped_optimal_actions = optimal_actions.reshape((*dim_griglia, num_stati_rm))

    # Mappa dei codici delle azioni ai simboli corrispondenti
    action_symbols = {0: "↑", 1: "↓", 2: "←", 3: "→"}

    # Visualizzazione delle Heatmaps
    fig, axes = plt.subplots(1, num_stati_rm, figsize=(20, 6))

    # Assicurati che axes sia sempre un array
    if num_stati_rm == 1:
        axes = [axes]  # Rendi axes un array se c'è solo un subplot

    for i in range(num_stati_rm):
        sns.heatmap(
            reshaped_max_q_values[:, :, i],
            annot=np.vectorize(action_symbols.get)(reshaped_optimal_actions[:, :, i]),
            fmt="",
            cmap="coolwarm",
            ax=axes[i],
        )
        axes[i].set_title(f"RM State {i + 1}")
        axes[i].set_xlabel("Column")
        axes[i].set_ylabel("Row")

    plt.tight_layout()

    # Salva la heatmap in un file PNG
    file_path = os.path.join(save_dir, f"heatmap_agent_{agent_id}.png")
    plt.savefig(file_path)
    plt.close(fig)  # Chiudi la figura per evitare conflitti


def generate_heatmaps_time(q_table, dim_griglia, num_stati_rm, max_time):
    # Calcola il valore medio di Q per ciascun stato RM e posizione, ignorando il tempo
    reshaped_q_table = q_table.reshape((*dim_griglia, max_time, num_stati_rm, -1))
    mean_q_values = reshaped_q_table.mean(axis=2)  # Media lungo l'asse del tempo

    max_q_values = mean_q_values.max(axis=-1)  # Ottieni i massimi valori Q
    optimal_actions = mean_q_values.argmax(axis=-1)  # Trova le azioni ottimali

    # Mappa dei codici delle azioni ai simboli corrispondenti
    action_symbols = {0: "↑", 1: "↓", 2: "←", 3: "→"}

    # Visualizzazione delle Heatmaps
    fig, axes = plt.subplots(1, num_stati_rm, figsize=(20, 6))

    if num_stati_rm == 1:
        axes = [axes]  # Assicurati che axes sia sempre un array

    for i in range(num_stati_rm):
        sns.heatmap(
            max_q_values[:, :, i],
            annot=np.vectorize(action_symbols.get)(optimal_actions[:, :, i]),
            fmt="",
            cmap="coolwarm",
            ax=axes[i],
        )
        axes[i].set_title(f"RM State {i + 1}")
        axes[i].set_xlabel("Column")
        axes[i].set_ylabel("Row")

    plt.tight_layout()
    plt.show()


"""
# Assuming q_tables_dict contains Q-tables for all agents as shown in your code
for agent_name, q_table in data.items():
    generate_heatmaps(q_table, dim_griglia=(10, 10), num_stati_rm=3)  # Adjust RM_3 to the correct RM instance if necessary
"""


def generate_heatmaps_for_agents(agents, q_tables_data, grid_dims):
    """
    Genera heatmap per le Q-table di ogni agente.

    Args:
        agents (list): Lista degli agenti.
        q_tables_data (npz file): Dati delle Q-table salvati come file npz.
        grid_dims (tuple): Dimensioni della griglia, es. (grid_width, grid_height).
    """
    for agent in agents:
        
        agent_name = agent.name
        agent_id = int(agent_name[-1])
        q_table = q_tables_data[
            f"q_table_{agent_name}"
        ]  # Recupera la Q-table dell'agente corrente
        num_rm_states = (
            agent.get_reward_machine().numbers_state()
        )  # Ottiene il numero di stati RM dinamicamente

        # Chiama la funzione generate_heatmaps per ogni agente
        # Assumendo che generate_heatmaps accetti i seguenti parametri: q_table, dimensioni griglia, e numero stati RM
        save_heatmaps(q_table, dim_griglia=grid_dims, num_stati_rm=num_rm_states, agent_id=agent_id)


def generate_heatmaps_for_agents_time(agents, q_tables_data, grid_dims, max_time):
    """
    Genera heatmap per le Q-table di ogni agente.

    Args:
        agents (list): Lista degli agenti.
        q_tables_data (npz file): Dati delle Q-table salvati come file npz.
        grid_dims (tuple): Dimensioni della griglia, es. (grid_width, grid_height).
    """
    for agent in agents:
        agent_name = agent.name
        q_table = q_tables_data[
            f"q_table_{agent_name}"
        ]  # Recupera la Q-table dell'agente corrente
        num_rm_states = (
            agent.get_reward_machine().numbers_state()
        )  # Ottiene il numero di stati RM dinamicamente

        # Chiama la funzione generate_heatmaps per ogni agente
        # Assumendo che generate_heatmaps accetti i seguenti parametri: q_table, dimensioni griglia, e numero stati RM
        generate_heatmaps_time(
            q_table,
            dim_griglia=grid_dims,
            num_stati_rm=num_rm_states,
            max_time=max_time,
        )


def generate_heatmaps_numbers(qtable: Dict[str, np.ndarray], map_size: int = None, seed: int = None):
    """
    Generate and save heatmaps for the Q-tables of all agents.

    Args:
        qtable (dict): Dictionary containing Q-tables for all agents.
        save_dir (str): Directory to save the heatmaps.
    """
    # Crea la directory di salvataggio se non esiste
    save_dir = HEATMAPS_DIR
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for idx, (agent_name, q_table) in enumerate(qtable.items()):
        _, axes = plt.subplots(figsize=(20, 20))

        # Se c'è un solo subplot, axes non è una lista: trasformalo in una lista
        if len(qtable.items()) == 1:
            axes = [axes]
        minval = q_table.min()
        maxval = q_table.max()
        row, column = np.unravel_index(q_table.argmax(), q_table.shape)
        while maxval >= 99:
            datacopy: np.ndarray = q_table.copy()
            datacopy[row, column] = 0
            maxval = datacopy.max()

        sns.heatmap(
            q_table,
            annot=q_table,  # Annotate with numbers
            fmt=".2f",  # Format numbers to 2 decimal places
            cbar=True,
            cbar_kws={"shrink": 0.8},
            annot_kws={"size": 15},
            cmap=sns.color_palette("coolwarm", as_cmap=True),
            vmin=minval - 0.05 * minval,
            vmax=maxval + 0.05 * maxval,
            # ax=axes[idx],  # Usa il subplot corretto
        )

        # Salva la heatmap per l'agente corrente
        save_path = os.path.join(save_dir, f"heatmap_{agent_name}_{map_size}_{seed}.png")
        plt.savefig(save_path)
        print(f"Heatmap salvata in: {save_path}")

    plt.tight_layout()
    plt.close()  # Chiude la figura per evitare conflitti