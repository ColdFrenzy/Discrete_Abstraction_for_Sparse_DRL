from pettingzoo import ParallelEnv
from unified_planning.shortcuts import *
from unified_planning.model.multi_agent import *
import numpy as np


class BaseEnvironment(ParallelEnv, MultiAgentProblem):
    def __init__(
        self, width, height, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.9995
    ):
        super().__init__()
        self.grid_width = width
        self.grid_height = height
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = self.epsilon_start
        self.rewards = {agent.name: 0 for agent in self.agents}
        self.penalty_amount = 0
        self.active_agents = {agent.name: True for agent in self.agents}
        self.agent_fail = {agent.name: False for agent in self.agents}
        self.agent_steps = {agent.name: 0 for agent in self.agents}
        self.initialize_environment()

    def initialize_environment(self):
        # Inizializza l'ambiente, definisci lo stato iniziale, ecc.
        pass

    def reset(self, seed=None, options=None):
        self.initialize_environment()
        # Restituisci osservazioni iniziali e informazioni
        observations = self.generate_observations()
        infos = {agent.name: {} for agent in self.agents}
        return observations, infos

    def step(self, actions):
        # Applica azioni, aggiorna ambiente, calcola ricompense
        self.apply_actions(actions)
        self.update_environment()
        rewards = self.calculate_rewards()
        terminations, truncations, infos = self.check_terminations()
        infos = self.generate_infos()
        observations = self.generate_observations()
        return observations, rewards, terminations, truncations, infos

    # Implementa i seguenti metodi in base alle tue esigenze
    def apply_actions(self, actions):
        pass

    def update_environment(self):
        pass

    def calculate_rewards(self):
        pass

    def check_terminations(self):
        terminations = {a.name: False for a in self.agents}
        truncations = {a.name: False for a in self.agents}
        infos = {a.name: {} for a in self.agents}
        return terminations, truncations, infos

    def generate_observations(self):
        pass

    def generate_infos(self):
        pass
