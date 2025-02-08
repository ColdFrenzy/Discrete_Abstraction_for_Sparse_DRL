import random
from typing import List

import gymnasium as gym
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class StandardReplayBuffer:
    def __init__(self, env_params, capacity=512, sample_func=None):
        self.env_params = env_params
        self.capacity = capacity
        self.size = 0
        self.sample_func = sample_func
        self.buffer = {
            "observation": torch.empty([capacity, env_params["obs_dim"]]),
            "action": torch.empty([capacity, env_params["action_dim"]]),
            "reward": torch.empty([capacity, 1]),
            "done": torch.empty([capacity, 1]),
            "goal": torch.empty([capacity, env_params["goal_dim"]]),
            "achieved_goal": torch.empty([capacity, env_params["goal_dim"]]),
            "new_observation": torch.empty([capacity, env_params["obs_dim"]]),
        }

    def populate(self, env: gym.Env, start_steps: int = 1000) -> None:
        observation = env.reset()[0]
        observation = torch.as_tensor(observation, dtype=torch.float32).to(
            device
        )  # buffer expects tensor
        for i in range(start_steps):
            action = env.action_space.sample()
            action = torch.as_tensor(action, dtype=torch.float32).to(
                device
            )  # buffer expects tensor
            # actions right np.array([4., 0.]), left np.array([-4., 0.]), up np.array([0., 4.]), down np.array([0., -4.])
            new_observation, reward, terminated, truncated, _ = env.step(
                action.cpu().numpy()
            )
            done = terminated or truncated
            new_observation = torch.as_tensor(
                new_observation, dtype=torch.float32
            ).to(
                device
            )  # buffer expects tensor
            self.store(observation, action, reward, done, new_observation)
            observation = new_observation
            if terminated or truncated:
                observation = env.reset()[0]
                observation = torch.as_tensor(
                    observation, dtype=torch.float32
                ).to(
                    device
                )  # buffer expects tensor

    def store(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        done: bool,
        new_observation: torch.Tensor,
        goal: torch.Tensor,
    ) -> None:
        index = self.size % self.capacity
        self.buffer["observation"][index] = observation
        self.buffer["action"][index] = action
        self.buffer["reward"][index] = reward
        self.buffer["done"][index] = done
        self.buffer["goal"][index] = observation
        self.buffer["new_observation"][index] = new_observation
        self.size += 1

    def sample(self, batch_size=32) -> List[torch.Tensor]:
        max_batch_index = min(self.size, self.capacity - 1)
        if self.sample_func is not None:
            temp_buffer = {}
            for key in self.buffer.keys():
                temp_buffer[key] = self.buffer[key][:max_batch_index]
            temp_buffer['ag_next'] = temp_buffer["goal"][1:]
            return self.sample_func(temp_buffer, batch_size)
        sampled_indices = random.sample(range(max_batch_index), batch_size)
        observations = self.buffer["observation"][sampled_indices].to(device)
        actions = self.buffer["action"][sampled_indices].to(device)
        rewards = self.buffer["reward"][sampled_indices].to(device)
        dones = self.buffer["done"][sampled_indices].to(device)
        new_observations = self.buffer["new_observation"][sampled_indices].to(
            device
        )
        return [observations, actions, rewards, dones, new_observations]

