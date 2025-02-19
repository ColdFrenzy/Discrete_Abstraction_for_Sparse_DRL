import numpy as np
import copy
import wandb
import os
from stable_baselines3.common.callbacks import BaseCallback
from collections import deque

class WinRateCallback(BaseCallback):
    def __init__(self, window_size=50, verbose=0):
        super().__init__(verbose)
        self.window_size = window_size  # La dimensione della finestra per il win_rate
        self.wins = deque(maxlen=window_size)  # La finestra degli ultimi episodi
        self.episode_reward = 0  # Puoi usare questo per memorizzare il reward cumulativo dell'episodio
        self.episode_done = False  # Flag per determinare se l'episodio è finito

    def _on_step(self) -> bool:
        """
        Questa funzione viene chiamata ad ogni passo dell'episodio.
        La logica per determinare se l'episodio è finito (terminato o troncato)
        viene eseguita solo quando `done` è True.
        """
        # Accedi a terminated e truncated
        terminated = self.locals['dones'][0]
        truncated = self.locals['infos'][0].get('TimeLimit.truncated', False)

        if terminated or truncated:
            win = self.locals['infos'][0]['log'] == 'GOAL REACHED'
            # Se l'episodio è finito, calcoliamo il risultato (vittoria/sconfitta)
            self.wins.append(1 if win else 0)

            # Calcola il win rate come la media degli ultimi n episodi
            win_rate = sum(self.wins) / len(self.wins) if self.wins else 0
            
            # Logga il win rate su TensorBoard
            self.logger.record("custom/win_rate", win_rate)
            
            # Reset della variabile reward per il prossimo episodio
        #     self.episode_reward = 0
        
        # # Aggiungi il reward dell'episodio in corso (aggiornato ad ogni passo)
        # if terminated or truncated:
        #     self.episode_done = True
        # else:
        #     self.episode_reward += self.locals['rewards'][0]

        return True


def calcola_media_mobile(valori, finestra):
    medie = np.convolve(valori, np.ones(finestra) / finestra, mode="valid")
    return medie


def test_policy_optima(env, episodi_test=2):
    # Inizia un nuovo run di wandb per i test
    env_test = copy.deepcopy(env)  # Crea una copia dell'ambiente per i test
    successi_per_agente = {ag.name: 0 for ag in env_test.agents}

    for episodio in range(episodi_test):
        states, infos = env_test.reset()  # Resetta l'ambiente di test per ogni episodio
        done = {ag.name: False for ag in env_test.agents}
        agent_done = {ag.name: False for ag in env_test.agents}

        while not all(done.values()):
            actions = {}
            for ag in env_test.agents:
                azione = ag.select_action(
                    states[ag.name], best=True
                )  # Seleziona l'azione in modalità test
                actions[ag.name] = azione

            new_states, rewards, done, truncations, infos = env_test.step(
                actions
            )  # Esegui un passo per tutti gli agenti

            for ag in env_test.agents:
                if agent_done[ag.name]:
                    continue
                if (
                    done[ag.name]
                    and ag.get_reward_machine().get_current_state()
                    == ag.get_reward_machine().get_final_state()
                ):
                    successi_per_agente[
                        ag.name
                    ] += 1  # Conta come successo se l'agente raggiunge lo stato finale della RM
                    agent_done[ag.name] = True
                    # break  # Interrompe il ciclo interno se l'agente ha terminato

            if all(
                done.values()
            ):  # Se tutti gli agenti hanno terminato, interrompi il ciclo per l'episodio
                break

            if all(truncations.values()):
                break

            states = copy.deepcopy(
                new_states
            )  # Aggiorna lo stato per il prossimo timestep

    success_rate_per_agente = {
        ag_name: (successi / episodi_test) * 100
        for ag_name, successi in successi_per_agente.items()
    }  # Calcola la percentuale di successo per ogni agente

    return success_rate_per_agente


def save_q_tables(agents, directory="data"):
    q_tables_dict = {}
    for agent in agents:
        try:
            q_table = agent.get_learning_algorithm().q_table
        except:
            q_table = agent.get_learning_algorithm().nTSQA
        q_tables_dict[f"q_table_{agent.name}"] = q_table
    if not os.path.exists(directory):
        os.makedirs(directory)
    np.savez_compressed(f"{directory}/q_tables.npz", **q_tables_dict)


def update_actions_log(actions_log, actions, episode):
    """
    Aggiorna il log delle azioni per ogni agente durante un episodio.

    Args:
        actions_log (dict): Dizionario contenente i log delle azioni per ogni agente.
        actions (dict): Dizionario delle azioni eseguite dagli agenti nell'ultimo step.
        episode (int): Numero dell'episodio corrente.
    """
    if episode not in actions_log:
        actions_log[episode] = {}
    for agent_name, action in actions.items():
        if agent_name not in actions_log[episode]:
            actions_log[episode][agent_name] = []
        actions_log[episode][agent_name].append(action.name)


def save_actions_log(actions_log, file_path="data/final_episode_log.json"):
    """
    Salva il log delle azioni in un file JSON.

    Args:
        actions_log (dict): Dizionario contenente i log delle azioni per ogni agente.
        file_path (str): Percorso del file dove salvare il log.
    """
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    with open(file_path, "w") as f:
        json.dump(actions_log, f, indent=4)


def update_successes(env, rewards_agents, successi_per_agente, done):
    for agent in env.agents:
        rm_current_state = agent.reward_machine.get_current_state()
        rm_final_state = agent.reward_machine.get_final_state()
        if (
            done[agent.name]
            and rm_current_state == rm_final_state
            and rewards_agents[agent.name] > 0
        ):
            successi_per_agente[agent.name] += 1


def prepare_log_data(
    env,
    episode,
    rewards_agents,
    successi_per_agente,
    ricompense_per_episodio,
    finestra_media_mobile,
):
    log_data = {"epsilon": env.epsilon, "episode": episode, "total_step": env.timestep}

    for agent in env.agents:
        agent_name = agent.name
        """if env.active_agents[agent.name] == False:
            continue"""
        reward = rewards_agents[agent_name]
        steps = env.agent_steps[agent_name]
        success_rate = (successi_per_agente[agent_name] / (episode + 1)) * 100
        ricompense_per_episodio[agent_name].append(reward)

        if len(ricompense_per_episodio[agent_name]) >= finestra_media_mobile:
            media_mobile = calcola_media_mobile(
                ricompense_per_episodio[agent_name], finestra_media_mobile
            )
            log_data[f"media_mobile_{agent_name}"] = media_mobile[-1]

        log_data.update(
            {
                f"reward_{agent_name}": reward,
                f"step_{agent_name}": steps,
                f"success_rate_training_{agent_name}": success_rate,
            }
        )

    return log_data


def get_epsilon_summary(agents):
    epsilon_parts = []
    for agent in agents:
        epsilon_parts.append(
            f"{agent.name}: N/A"
        )  # Per algoritmi che non usano epsilon
    return ", ".join(epsilon_parts)
