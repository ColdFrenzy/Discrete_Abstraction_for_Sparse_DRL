from typing import Optional, List, Union, Iterable


class RewardMachine:
    def __init__(self, transitions, event_detector):
        self.transitions = transitions  # {(current_state, event): (new_state, reward)}
        self.initial_state = self._get_start_state()  # Memorizza lo stato iniziale
        self.current_state = self.initial_state
        self.state_indices = self._generate_state_indices()
        self.event_detector = event_detector
        self.potentials = None  # Aggiungi questa linea per memorizzare i potenziali

    def _generate_state_indices(self):
        # Raccogli tutti gli stati univoci (sia di partenza che di arrivo) dalle transizioni
        unique_states = set()
        for (from_state, _), (to_state, _) in self.transitions.items():
            unique_states.add(from_state)
            unique_states.add(to_state)

        # Assicurati che lo stato iniziale sia incluso e mappato a zero
        unique_states.add(self.current_state)
        sorted_states = sorted(unique_states)
        sorted_states.remove(self.current_state)
        sorted_states.insert(0, self.current_state)
        # breakpoint()
        # Assegna un indice univoco a ciascuno stato
        return {state: i for i, state in enumerate(sorted_states)}

    def get_state_index(self, rm_state):
        return self.state_indices[rm_state]

    def step(self, current_state):
        """
        Rileva l'evento corrente, esegue la transizione di stato e restituisce la ricompensa.
        """
        event = self.event_detector.detect_event(current_state)
        # print(f"Detected event: {event} for current state: {current_state}")
        if (self.current_state, event) in self.transitions:
            new_state, reward = self.transitions[(self.current_state, event)]
            self.current_state = new_state
            return reward
        return 0

    def get_reward(self, event):
        """
        Restituisce la ricompensa associata a un evento specifico senza eseguire la transizione di stato.
        """
        if (self.current_state, event) in self.transitions:
            new_state, reward = self.transitions[(self.current_state, event)]
            self.current_state = new_state

            return reward

        return 0

    def get_reward_for_non_current_state(self, state_rm, event):
        # Converti l'evento in un tipo hashable se necessario
        if isinstance(event, list):
            print("It's a list!")
            breakpoint()
            event = tuple(event)

        if (state_rm, event) in self.transitions:
            new_state, reward = self.transitions[(state_rm, event)]
            return new_state, reward
        else:
            return None, 0

    def get_all_states(self):
        # Creare un set per tenere traccia degli stati già aggiunti
        seen_states = set()
        all_states = []

        # Aggiungere gli stati in ordine di apparizione
        for (from_state, _), (to_state, _) in self.transitions.items():
            if from_state not in seen_states:
                all_states.append(from_state)
                seen_states.add(from_state)
            if to_state not in seen_states:
                all_states.append(to_state)
                seen_states.add(to_state)

        return all_states

    def get_possible_events(self, state_rm):
        possible_events = []
        for (current_state, event), (new_state, _) in self.transitions.items():
            if current_state == state_rm:
                possible_events.append(event)
        return possible_events

    def get_current_state(self):
        return self.current_state

    def numbers_state(self):
        states = set()
        for (from_state, _), (to_state, _) in self.transitions.items():
            states.add(from_state)
            states.add(to_state)
        return len(states)

    @property
    def get_transitions(self):
        return self.transitions

    def reset_to_initial_state(self):
        self.current_state = (
            self.initial_state
        )  # Assumi che 'initial_state' sia memorizzato come attributo
        return self.initial_state

    def get_final_state(self):
        # Cerca l'ultimo stato di arrivo tra le transizioni
        if self.transitions:  # Assicurati che ci siano transizioni
            last_to_state = next(reversed(self.transitions.values()))[0]
            return last_to_state
        else:
            # Nessuna transizione definita, gestisci come preferisci
            return None

    def _get_start_state(self):
        # Assicurati che ci siano transizioni definite
        if not self.transitions:
            return None

        # Prendi il primo stato di partenza dalla prima transizione della lista delle transizioni
        first_transition = next(iter(self.transitions))
        start_state = first_transition[0]

        return start_state

    def add_reward_shaping(self, gamma, rs_gamma):
        self.gamma = gamma
        self.potentials = self.value_iteration(
            list(self.state_indices.keys()),
            self.get_delta_u(),
            self.get_delta_r(),
            self.get_final_state(),
            rs_gamma,
        )
        for u in self.potentials:
            self.potentials[u] = -self.potentials[u]

    def get_delta_u(self):
        delta_u = {}
        for (u1, event), (u2, _) in self.transitions.items():
            if u1 not in delta_u:
                delta_u[u1] = {}
            if u2 not in delta_u:
                delta_u[u2] = {}  # Ensure u2 is also present
            delta_u[u1][u2] = event
        return delta_u

    def get_delta_r(self):
        delta_r = {}
        for (u1, event), (u2, reward) in self.transitions.items():
            if u1 not in delta_r:
                delta_r[u1] = {}
            if u2 not in delta_r:
                delta_r[u2] = {}  # Ensure u2 is also present
            delta_r[u1][u2] = ConstantRewardFunction(reward)
        return delta_r

    def value_iteration(self, U, delta_u, delta_r, terminal_u, gamma):
        V = dict([(u, 0) for u in U])
        V[terminal_u] = 0
        V_error = 1

        # Debugging print statements
        print("Initial V:", V)
        print("Delta U:", delta_u)
        print("Delta R:", delta_r)

        while V_error > 0.0000001:
            V_error = 0
            for u1 in U:
                if not delta_u[u1]:  # Check if there are no outgoing transitions
                    continue
                q_u2 = []
                for u2 in delta_u[u1]:
                    if delta_r[u1][u2].get_type() == "constant":
                        r = delta_r[u1][u2].get_reward(None)
                    else:
                        r = 0  # Se la funzione di ricompensa non è costante, assume che ritorni zero
                    q_u2.append(r + gamma * V[u2])
                if q_u2:  # Ensure q_u2 is not empty
                    v_new = max(q_u2)
                    V_error = max([V_error, abs(v_new - V[u1])])
                    V[u1] = v_new
        return V


# Definizione di ConstantRewardFunction
class RewardFunction:
    def __init__(self):
        pass

    def get_reward(self, s_info):
        raise NotImplementedError("To be implemented")

    def get_type(self):
        raise NotImplementedError("To be implemented")


class ConstantRewardFunction(RewardFunction):
    def __init__(self, c):
        super().__init__()
        self.c = c

    def get_type(self):
        return "constant"

    def get_reward(self, s_info):
        return self.c
