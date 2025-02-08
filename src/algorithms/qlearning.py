import numpy as np
from src.algorithms.learning_algorithm import BaseLearningAlgorithm


class QLearning(BaseLearningAlgorithm):
    def __init__(
        self,
        gamma,
        action_selection,
        learning_rate=None,
        epsilon_start=1.0,
        epsilon_end=0.2,
        epsilon_decay=0.99,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.action_selection = action_selection  # 'softmax' o 'greedy'
        self.q_table = np.ones((self.state_space_size, self.action_space_size))
        self.visits = np.zeros((self.state_space_size, self.action_space_size))
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = self.epsilon_start
        self.tosave += ["q_table", "visits", "epsilon"]

    def update(
        self, encoded_state, encoded_next_state, action, reward, terminated, **kwargs
    ):
        current_q = self.q_table[encoded_state, action]
        self.visits[encoded_state, action] += 1
        # auto learning rate
        if self.learning_rate is None:
            learning_rate = 1 / self.visits[encoded_state, action]
        else:
            learning_rate = self.learning_rate
        max_future_q = (not terminated) * np.max(self.q_table[encoded_next_state])
        new_q = (1 - learning_rate) * current_q + learning_rate * (
            reward + self.gamma * max_future_q
        )
        self.q_table[encoded_state, action] = new_q

    def choose_action(self, encoded_state, best=False, rng=None, **kwargs):
        if best:
            return np.argmax(self.q_table[encoded_state])
        if rng is None:
            rng = self.rng
        if rng.uniform(0, 1) < self.epsilon:
            # Esplorazione casuale
            return rng.choice(range(self.action_space_size))
        else:
            # Scegli l'azione in base al metodo specificato
            if self.action_selection == "softmax":
                return self.choose_action_softmax(encoded_state, rng)
            elif self.action_selection == "greedy":
                return self.choose_action_greedy(encoded_state, rng)
            else:
                raise ValueError("Metodo di selezione dell'azione non supportato")

    def choose_action_softmax(self, encoded_state, rng):
        softmax_probs = self.softmax(self.q_table[encoded_state])
        return rng.choice(np.arange(self.action_space_size), p=softmax_probs)

    def choose_action_greedy(self, encoded_state, rng):
        # choose random action among best ones
        Qa = self.q_table[encoded_state]
        va = np.argmax(Qa)
        maxs = [i for i, v in enumerate(Qa) if v == Qa[va]]
        action = rng.choice(maxs)
        return action  # np.argmax(self.q_table[encoded_state])

    def softmax(self, q_values, beta=4):
        exp_q = np.exp(
            beta * q_values - np.max(beta * q_values)
        )  # Sottrai il max per la stabilità numerica
        probabilities = exp_q / np.sum(exp_q)
        return probabilities

    def learn_done_episode(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        # print(f"{self.epsilon:6.3f}")

    def reset_epsilon(self):
        self.epsilon = self.epsilon_start
