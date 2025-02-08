import numpy as np


def encode_state(agent, state, state_reward_machine):
    """
    Codifica lo stato corrente e lo stato della Reward Machine in un singolo indice intero.

    :param agent: L'agente per il quale codificare lo stato.
    :param state: Lo stato corrente dell'agente.
    :return: Indice intero che rappresenta lo stato codificato.
    """

    # Estrai le informazioni necessarie da 'agent' e 'state'

    RM_agent = agent.get_reward_machine()
    num_rm_states = RM_agent.numbers_state()
    pos_x, pos_y = state["pos_x"], state["pos_y"]
    rm_state_index = RM_agent.get_state_index(state_reward_machine)
    max_x_value, max_y_value = agent.ma_problem.grid_width, agent.ma_problem.grid_height

    # Calcola l'indice basato sulla posizione
    pos_index = pos_y * max_x_value + pos_x

    # Codifica lo stato combinando la posizione e lo stato della Reward Machine
    encoded_state = pos_index * num_rm_states + rm_state_index

    # Controlla che l'indice codificato non superi le dimensioni totali dello spazio degli stati
    if encoded_state >= max_x_value * max_y_value * num_rm_states:
        raise ValueError(
            "Indice di stato codificato supera la dimensione dello spazio degli stati",
            encoded_state,
            ">=",
            max_x_value * max_y_value * num_rm_states,
        )

    return encoded_state


def encode_state_with_time(agent, state, state_reward_machine):
    """
    Codifica lo stato corrente, lo stato della Reward Machine e il timestamp in un singolo indice intero.

    :param agent: L'agente per il quale codificare lo stato.
    :param state: Lo stato corrente dell'agente, inclusi pos_x, pos_y e timestamp.
    :return: Indice intero che rappresenta lo stato codificato.
    """

    # Estrai le informazioni necessarie da 'agent' e 'state'
    RM_agent = agent.get_reward_machine()
    num_rm_states = RM_agent.numbers_state()
    pos_x, pos_y, time_index = state["pos_x"], state["pos_y"], state["timestamp"]
    rm_state_index = RM_agent.get_state_index(state_reward_machine)
    max_x_value, max_y_value = agent.ma_problem.grid_width, agent.ma_problem.grid_height
    max_time_value = (
        agent.ma_problem.max_time
    )  # Assicurati che max_time sia definito nell'ambiente/problem

    # Calcola l'indice basato sulla posizione e sul tempo
    pos_index = pos_y * max_x_value + pos_x
    time_component = time_index  # Potresti voler scalare o modificare questa componente in base alle necessità

    # Codifica lo stato combinando la posizione, il tempo e lo stato della Reward Machine
    # Ora la formula tiene conto della dimensione temporale oltre alle dimensioni spaziali e allo stato della RM
    encoded_state = (
        pos_index * max_time_value + time_component
    ) * num_rm_states + rm_state_index

    # Controlla che l'indice codificato non superi le dimensioni totali dello spazio degli stati
    total_states = max_x_value * max_y_value * max_time_value * num_rm_states
    if encoded_state >= total_states:
        raise ValueError(
            "Indice di stato codificato supera la dimensione dello spazio degli stati",
            encoded_state,
            ">=",
            total_states,
        )

    return encoded_state


def encode_state_time(agent, state, state_reward_machine):
    """
    Codifica lo stato corrente, lo stato della Reward Machine, e il timestap in un singolo indice intero.

    :param agent: L'agente per il quale codificare lo stato.
    :param state: Lo stato corrente dell'agente.
    :param state_reward_machine: Stato corrente della Reward Machine.
    :param timestep: Timestap corrente, un intero da 0 a 50.
    :return: Indice intero che rappresenta lo stato codificato.
    """
    # Estrai le informazioni necessarie da 'agent' e 'state'
    RM_agent = agent.get_reward_machine()
    num_rm_states = RM_agent.numbers_state()
    pos_x, pos_y, timestep = state["pos_x"], state["pos_y"], state["timestep"]
    rm_state_index = RM_agent.get_state_index(state_reward_machine)
    max_x_value, max_y_value = agent.ma_problem.grid_width, agent.ma_problem.grid_height
    num_timesteps = (
        agent.ma_problem.max_time
    )  # Timesteps vanno da 0 a 50, quindi abbiamo 51 timesteps possibili

    # Calcola l'indice basato sulla posizione
    pos_index = pos_y * max_x_value + pos_x

    # Estendi la codifica per includere il tempo
    base_index = pos_index * num_rm_states + rm_state_index
    encoded_state = base_index * num_timesteps + timestep
    # encoded_state = np.int16(encoded_state)

    # Calcola il numero totale possibile di stati
    total_states = max_x_value * max_y_value * num_rm_states * num_timesteps

    # print(base_index, num_timesteps, timestep, "total states:", total_states, "state:", state)

    # Controlla che l'indice codificato non superi le dimensioni totali dello spazio degli stati
    if encoded_state >= total_states:
        raise ValueError(
            "Indice di stato codificato supera la dimensione dello spazio degli stati",
            encoded_state,
            ">=",
            total_states,
        )

    return encoded_state


def encode_environment_state(agent, state):
    max_x_value = agent.ma_problem.grid_width
    pos_x, pos_y = state["pos_x"], state["pos_y"]
    encoded_state = pos_y * max_x_value + pos_x
    return encoded_state


def encode_environment_state_time(agent, state):
    max_x_value = agent.ma_problem.grid_width
    pos_x, pos_y, timestep = state["pos_x"], state["pos_y"], state["timestep"]
    num_timesteps = agent.ma_problem.max_time
    pos_index = pos_y * max_x_value + pos_x
    encoded_state = pos_index * num_timesteps + timestep
    return encoded_state


"""def encode_environment_state(agent, state):
    max_x_value = agent.ma_problem.grid_width
    max_time_value = agent.ma_problem.max_time
    pos_x, pos_y, time_index = state["pos_x"], state["pos_y"], state['timestamp']

    pos_index = pos_y * max_x_value + pos_x
    #time_index = timestamp

    encoded_state = pos_index * max_time_value + time_index
    return encoded_state"""


def encode_reward_machine_state(agent, q):
    # Assumi che RM_agent mantenga un elenco ordinato degli stati o una mappatura da stati a indici
    RM_agent = agent.get_reward_machine()
    encoded_q = RM_agent.get_state_index(
        q
    )  # Supponendo che esista una funzione come questa
    return encoded_q


def parse_map_string(map_string):
    holes = []
    goals = {}
    for y, row in enumerate(map_string.strip().split("\n")):
        for x, cell in enumerate(row.strip()):
            if cell == "H":
                holes.append((x, y))
            elif cell.isdigit():  # Controlla se il carattere è un numero
                goals[int(cell)] = (x, y)  # Memorizza il goal con il numero come chiave
    return holes, goals


def parse_map_emoji(map_string):
    holes = []
    goals = {}
    for y, row in enumerate(map_string.strip().split("\n")):
        x = 0  # Inizializza il contatore della posizione x
        cell = ""  # Inizializza la cella vuota
        for char in row:
            if char == " " and cell:  # Se trovi uno spazio e la cella non è vuota
                if "⛔" in cell:  # Controlla se la cella contiene un buco
                    holes.append((x, y))
                elif (
                    cell.strip().isdigit()
                ):  # Controlla se la cella contiene un numero/goal
                    goal_number = int(cell.strip())
                    goals[goal_number] = (x, y)
                cell = ""  # Resetta la cella per la prossima iterazione
                x += 1  # Incrementa la posizione x dopo aver processato una cella
            else:
                cell += char  # Aggiungi il carattere alla cella corrente

        # Gestisci l'ultima cella della riga (se non termina con uno spazio)
        if cell:
            if "⛔" in cell:
                holes.append((x, y))
            elif cell.strip().isdigit():
                goal_number = int(cell.strip())
                goals[goal_number] = (x, y)
    
    return holes, goals


def generate_transitions(obstacles, goals, max_time):
    states = ["state_" + str(i) for i in range(max_time + 1)]
    states += [f"state_reached_1_{i}" for i in range(max_time + 1)]
    states += [f"state_reached_2_{i}" for i in range(max_time + 1)]
    states.append("free_play")

    transitions = {}

    # Aggiungi transizioni per collisioni
    for state in states:
        for x, y, t in obstacles:
            transitions[(state, f"collision_{x}_{y}_{t}")] = (state, -20)

    # Transizioni per avanzamento del tempo
    for i in range(max_time):
        transitions[(f"state_{i}", f"timestep_{i + 1}")] = (f"state_{i + 1}", 0)
        transitions[(f"state_reached_1_{i}", f"timestep_{i + 1}")] = (
            f"state_reached_1_{i + 1}",
            0,
        )
        transitions[(f"state_reached_2_{i}", f"timestep_{i + 1}")] = (
            f"state_reached_2_{i + 1}",
            0,
        )

    # Transizioni al free_play dopo il timestep 25
    transitions[(f"state_{max_time}", "timestep_{max_time+1}")] = ("free_play", 0)
    transitions[(f"state_reached_1_{max_time}", "timestep_{max_time+1}")] = (
        "free_play",
        0,
    )
    transitions[(f"state_reached_2_{max_time}", "timestep_{max_time+1}")] = (
        "free_play",
        0,
    )

    # Transizioni per gli obiettivi
    for state in states[: max_time + 1] + [
        f"state_reached_1_{i}" for i in range(max_time + 1)
    ]:
        transitions[(state, "(2, 1)")] = (f"state_reached_1_{0}", 10)
    for state in [f"state_reached_1_{i}" for i in range(max_time + 1)]:
        transitions[(state, "(12, 2)")] = (f"state_reached_2_{0}", 20)

    # Cicli all'interno dello stato free_play
    transitions[("free_play", "continue")] = ("free_play", 0)

    return transitions
