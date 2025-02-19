import numpy as np
from tqdm import tqdm
from src.reward_machines.reward_machine_cont import RewardMachine
from src.algorithms.agent_rl import AgentRL
from src.algorithms.qlearning import QLearning
from src.environments.state_encoder_uav import (
    StateEncoderUAV,
)
from src.environments.multi_agent.ma_uav import (
    MultiAgentUAV,
)

from src.utils.evaluation_metrics import *
import copy
from src.environments.detect_event import (
    PositionEventDetector,
)  # Import del nuovo EventDetector
from src.environments.multi_agent.rm_environment_wrapper import (
    RMEnvironmentWrapper,
)  # Import del wrapper
from src.utils.utils import *
from src.utils.paths import QTABLE_DIR
from src.utils.utils import parse_map_emoji
from src.utils.plot import save_reward_plots





def train(rm_env: RMEnvironmentWrapper, num_episodes: int) -> None:
    """
    Train the reinforcement learning agents in the given environment for a specified number of episodes.
    Args:
        rm_env (RMEnvironmentWrapper): The environment wrapper containing the agents and environment to be trained.
        num_episodes (int): The number of episodes to train the agents.
    Returns:
        None
    This function initializes the training process for each agent, runs the training loop for the specified number of episodes,
    and updates the agents' policies based on the rewards received. It also logs the actions and rewards for each agent, 
    updates the success metrics, and saves reward plots at the end of the training.
    """
    successi_per_agente = {agent.name: 0 for agent in rm_env.agents}
    actions_log = {agent.name: [] for agent in rm_env.agents}
    reward_per_episode = {agent.name: [] for agent in rm_env.agents}  # Per registrare le ricompense

    rm_env.reset()
    for agent in rm_env.agents:
        agent.get_learning_algorithm().learn_init()

    with tqdm(total=num_episodes) as pbar:
        for episode in range(num_episodes):
            states, infos = rm_env.reset()
            states = copy.deepcopy(states)
            done = {a.name: False for a in rm_env.agents}
            rewards_agents = {
                a.name: 0 for a in rm_env.agents
            }  # Inizializza le ricompense episodiche

            while not all(done.values()):
                # states, infos = rm_env.reset()
                # states = copy.deepcopy(states)
                actions = {}
                rewards = {a.name: 0 for a in rm_env.agents}
                # infos = {a.name: {} for a in rm_env.agents}
                for ag in rm_env.agents:
                    current_state = rm_env.env.get_state(ag)
                    action = ag.select_action(current_state)
                    actions[ag.name] = action
                    # Log delle azioni nell'ultimo episodio
                    update_actions_log(actions_log, actions, num_episodes)

                new_states, rewards, done, truncations, infos = rm_env.step(actions)

                for agent in rm_env.agents:
                    """if not rm_env.env.active_agents[agent.name]:
                    continue"""
                    terminated = done[agent.name] or truncations[agent.name]

                    agent.update_policy(
                        state=states[agent.name],
                        action=actions[agent.name],
                        reward=rewards[agent.name],
                        next_state=new_states[agent.name],
                        terminated=terminated,
                        infos=infos[agent.name],
                    )

                    rewards_agents[agent.name] += rewards[agent.name]
                states = copy.deepcopy(new_states)
                # end-training

                if all(truncations.values() or done.values()):
                    break
            # Registra le ricompense per episodio
            for agent in rm_env.agents:
                reward_per_episode[agent.name].append(rewards_agents[agent.name])

            update_successes(rm_env.env, rewards_agents, successi_per_agente, done)
            epsilon_str = get_epsilon_summary(rm_env.agents)
            pbar.set_description(
                f"Episodio {episode + 1}: Ricompensa = {rewards_agents}, Total Step: {rm_env.env.timestep}, Agents Step = {rm_env.env.agent_steps}, Epsilon agents= [{epsilon_str}]"
            )
            pbar.update(1)
            
    # Salva i grafici delle ricompense
    save_reward_plots(reward_per_episode, save_path="reward_plots")



def compute_value_function(map_name, size, OBST, num_episodes = 10000, gamma: float = 0.8, stochastic: bool = False, save: bool = True, show: bool = True) -> None:
    """
    Computes the value function for a multi-agent reinforcement learning environment.

    Args:
        map_name (str): The name of the map to be used for the environment.
        size (int): The size of the environment grid.
        OBST (int): The number of obstacles in the environment.
        num_episodes (int, optional): The number of episodes for training. Default is 10000.
        gamma (float, optional): The discount factor for the reinforcement learning algorithm. Default is 0.8.
        stochastic (bool, optional): If True, the environment will be stochastic. Default is False.
        save (bool, optional): If True, the Q-table will be saved after training. Default is True.
        show (bool, optional): If True, the environment will be displayed. Default is True.

    Returns:
        None

    This function sets up a multi-agent reinforcement learning environment, adds agents to it, defines reward machines, 
    and trains the agents using the RMax algorithm. The Q-table is saved at the end of training if the save parameter is set to True.
    """
    # Env creation from map
    holes, goals = parse_map_emoji(map_name)
    env = MultiAgentUAV(
        width=size,
        height=size,
        holes=holes,
    )

    # Environment Options
    env.stochastic_action = stochastic
    env.penalty_amount = 0
    env.delay_action = False  # Abilita la funzione "wait"

    # Add agents to the environment
    a1 = AgentRL("a1", env)
    a1.set_initial_position(0, 0)  # Aggiungo la pos anche allo stato dell'agente
    a1.add_state_encoder(StateEncoderUAV(a1))

    
    a3 = AgentRL("a3", env)
    a3.set_initial_position(0, size-2)  # Aggiungo la pos anche allo stato dell'agente
    a3.add_state_encoder(StateEncoderUAV(a3))

    # Adding actions
    for action in env.get_actions():
        a1.add_action(action)
        a3.add_action(action)

    state_start = "state0"
    state_reached_1 = "state1"
    rm_states = [state_start, state_reached_1]

    rewards = [10]

    # Definisci le transizioni della RM
    # {(stato_corrente, evento): (nuovo_stato, ricompensa)}
    transitions = {}
    for i in range(len(rm_states)-1):
        transitions.update({(rm_states[i], list(goals.values())[i]): (rm_states[i+1], rewards[i])})
    event_detector = PositionEventDetector(list(goals.values()))

    # Crea la RM
    RM_1 = RewardMachine(transitions, event_detector)
    a1.set_reward_machine(RM_1)
    env.add_agent(a1)

    # Crea la RM
    RM_3 = RewardMachine(transitions, event_detector)
    a3.set_reward_machine(RM_3)
    env.add_agent(a3)
    # Avvolgi l'ambiente con il wrapper RMEnvironmentWrapper
    rm_env = RMEnvironmentWrapper(env, [a1, a3]) # aggiungici a3 se vuoi averne 2 nella lista

    alg_1 = QLearning(
        gamma=gamma,
        action_selection='greedy',
        learning_rate=None,
        epsilon_start=1.0,
        epsilon_end=0.2,
        epsilon_decay=0.99,
        state_space_size=env.grid_width * env.grid_height * RM_3.numbers_state(),
        action_space_size=4,
    )
    alg_3 = QLearning(
        gamma=gamma,
        action_selection='greedy',
        learning_rate=None,
        epsilon_start=1.0,
        epsilon_end=0.2,
        epsilon_decay=0.99,
        state_space_size=env.grid_width * env.grid_height * RM_3.numbers_state(),
        action_space_size=4,
    )

    a1.set_learning_algorithm(alg_1)
    a3.set_learning_algorithm(alg_3)
    
    train(rm_env, num_episodes)

    # Salva la Q-table all'ultimo episodio
    if save:
        qtable = {}
        for agent in rm_env.agents:
            print(agent.name)
            for i in range(agent.get_reward_machine().numbers_state() - 1):
                max_qtable: np.ndarray = agent.get_learning_algorithm().q_table.max(axis=1)
                reshaped_qtable = max_qtable.reshape((size, size, 2))
                qtable[agent.name] = reshaped_qtable[:, :, i]

                for (x,y) in goals.values():
                    qtable[agent.name][y, x] = 15.
                            
        transition_mode = "stochastic" if stochastic else "deterministic" # TODO: hardcoded
        os.makedirs(f"{QTABLE_DIR}/{transition_mode}", exist_ok=True)
        np.savez_compressed(f"{QTABLE_DIR}/{transition_mode}/qtable_{size}_obstacles_{OBST}.npz", **qtable)

def refine_qtable(qtable):
    """Find the absolute best path in the value matrix that connects the first column to the last column.
    The path should be contiguous and all other elements of the matrix not belonging to the path should be zero.
    
    Parameters:
    qtable (np.ndarray): 2D numpy array representing the value function of a grid world.
    
    Returns:
    np.ndarray: 2D numpy array with the best path marked and all other elements set to zero.
    """
    rows, cols = qtable.shape
    path_matrix = np.zeros_like(qtable)
    
    # Initialize the best path and its value
    best_path = []
    best_value = -np.inf
    
    # Function to find the best path from a given starting point
    def find_path(x, y, current_path, current_value):
        nonlocal best_path, best_value
        
        if y == cols - 1:
            if current_value > best_value:
                best_value = current_value
                best_path = current_path[:]
            return
        
        if x > 0:
            find_path(x - 1, y + 1, current_path + [(x - 1, y + 1)], current_value + qtable[x - 1, y + 1])
        find_path(x, y + 1, current_path + [(x, y + 1)], current_value + qtable[x, y + 1])
        if x < rows - 1:
            find_path(x + 1, y + 1, current_path + [(x + 1, y + 1)], current_value + qtable[x + 1, y + 1])
    
    # Start from each cell in the first column
    for i in range(rows):
        find_path(i, 0, [(i, 0)], qtable[i, 0])
    
    # Mark the best path in the path_matrix
    for x, y in best_path:
        path_matrix[x, y] = qtable[x, y]
    
    return path_matrix

def refine_qtable(qtable, start, end):
    """Find the best path in the value matrix from the start position to the end position.
    The path should be contiguous and all other elements of the matrix not belonging to the path should be zero.
    
    Parameters:
    qtable (np.ndarray): 2D numpy array representing the value function of a grid world.
    start (tuple): Starting position (row, column).
    end (tuple): Ending position (row, column).
    
    Returns:
    np.ndarray: 2D numpy array with the best path marked and all other elements set to zero.
    """
    rows, cols = qtable.shape
    path_matrix = np.zeros_like(qtable)
    
    # Initialize the best path and its value
    best_path = []
    best_value = -np.inf
    
    # Function to find the best path from a given starting point
    def find_path(x, y, current_path, current_value):
        nonlocal best_path, best_value
        
        if (x, y) == end:
            if current_value > best_value:
                best_value = current_value
                best_path = current_path[:]
            return
        
        if x > 0:
            find_path(x - 1, y, current_path + [(x - 1, y)], current_value + qtable[x - 1, y])
        if y > 0:
            find_path(x, y - 1, current_path + [(x, y - 1)], current_value + qtable[x, y - 1])
        if x < rows - 1:
            find_path(x + 1, y, current_path + [(x + 1, y)], current_value + qtable[x + 1, y])
        if y < cols - 1:
            find_path(x, y + 1, current_path + [(x, y + 1)], current_value + qtable[x, y + 1])
    
    # Start from the given starting position
    find_path(start[0], start[1], [start], qtable[start[0], start[1]])
    
    # Mark the best path in the path_matrix
    for x, y in best_path:
        path_matrix[x, y] = qtable[x, y]
    
    return path_matrix



def compute_value_function_single(map_name, size, OBST, num_episodes = 10000, gamma: float = 0.8, stochastic: bool = False, save: bool = True, show: bool = True) -> None:
    
    # Env creation from map
    holes, goals = parse_map_emoji(map_name)
    env = MultiAgentUAV(
        width=size,
        height=size,
        holes=holes,
    )

    # Environment Options
    env.frozen_lake = stochastic
    env.penalty_amount = 0
    env.delay_action = False  # Abilita la funzione "wait"

    # Add agents to the environment
    a1 = AgentRL("a1", env)
    a1.set_initial_position(0, 0)  # Aggiungo la pos anche allo stato dell'agente
    a1.add_state_encoder(StateEncoderUAV(a1))

    
    # a3 = AgentRL("a3", env)
    # a3.set_initial_position(size - 1, size - 1)  # Aggiungo la pos anche allo stato dell'agente
    # a3.add_state_encoder(StateEncoderUAV(a3))

    # Adding actions
    for action in env.get_actions():
        a1.add_action(action)
        # a3.add_action(action)

    # Reward Machine States (TODO: generalize this)
    state_start = "state0"
    state_reached_1 = "state1"
    rm_states = [state_start, state_reached_1]

    # Reward Machine Rewards (TODO: generalize this)
    rewards = [10]

    # Definisci le transizioni della RM
    # {(stato_corrente, evento): (nuovo_stato, ricompensa)}
    transitions = {}
    for i in range(len(rm_states)-1):
        transitions.update({(rm_states[i], list(goals.values())[i]): (rm_states[i+1], rewards[i])})
    event_detector = PositionEventDetector(list(goals.values()))

    # Crea la RM
    RM_1 = RewardMachine(transitions, event_detector)
    a1.set_reward_machine(RM_1)
    env.add_agent(a1)

    # Crea la RM
    # RM_3 = RewardMachine(transitions, event_detector)
    # a3.set_reward_machine(RM_1)
    # env.add_agent(a3)

    # Avvolgi l'ambiente con il wrapper RMEnvironmentWrapper
    rm_env = RMEnvironmentWrapper(env, [a1]) # aggiungici a3 se vuoi averne 2 nella lista

    # Avvolgi l'ambiente con il wrapper RMEnvironmentWrapper
    # rm_env = RMEnvironmentWrapper(env, [a3])

    alg_1 = QLearning(
        gamma=gamma,
        action_selection='greedy',
        learning_rate=None,
        epsilon_start=1.0,
        epsilon_end=0.2,
        epsilon_decay=0.99,
        state_space_size=env.grid_width * env.grid_height * RM_1.numbers_state(),
        action_space_size=4,
    )
    # alg_3 = QLearning(
    #     state_space_size=env.grid_width * env.grid_height * RM_3.numbers_state(),
    #     action_space_size=4,
    #     s_a_threshold=500,
    #     max_reward=10,
    #     gamma=gamma,
    #     epsilon_one=0.99,
    # )
    a1.set_learning_algorithm(alg_1)
    # a3.set_learning_algorithm(alg_3)
    
    train(rm_env, num_episodes)

    # Salva la Q-table all'ultimo episodio
    if save:
        qtable = {}
        for agent in rm_env.agents:
            print(agent.name)
            for i in range(agent.get_reward_machine().numbers_state() - 1):
                max_qtable: np.ndarray = agent.get_learning_algorithm().q_table.max(axis=1)
                reshaped_qtable = max_qtable.reshape((size, size, 2))
                qtable[agent.name] = reshaped_qtable[:, :, i]
                
                # max_counts: np.ndarray = alg_3.s_a_counts.max(axis=1)
                # reshaped_counts = max_counts.reshape((size, size, 2))
                # counts = reshaped_counts[:, :, i]
                
                # for i in range(counts.shape[0]):
                #     for j in range(counts.shape[1]):
                #         if counts[i, j] < 1 and (j, i) not in goals.values():
                #             qtable[agent.name][i, j] = 0.0
                for (x,y) in goals.values():
                    qtable[agent.name][y, x] = 15. 

                for hole in holes:
                    x, y = hole
                    qtable[agent.name][y, x] = 0.   
    
        transition_mode = "stochastic" if stochastic else "deterministic" # TODO: hardcoded
        os.makedirs(f"{QTABLE_DIR}/{transition_mode}/single_agent", exist_ok=True)
        np.savez_compressed(f"{QTABLE_DIR}/{transition_mode}/single_agent/qtable_{size}_obstacles_{OBST}.npz", **qtable)