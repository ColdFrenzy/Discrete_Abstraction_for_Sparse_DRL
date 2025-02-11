import random
from typing import List

import gymnasium as gym
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EpisodicBuffer:
    def __init__(self, env_params, capacity=int(1e4), max_timesteps=300, sample_func=None):
        """Create a replay buffer.
        Args:
            env_params (dict): Environment parameters.
            capacity (int): The maximum number of episodes that the buffer can store.
            max_timesteps (int): The maximum number of timesteps that an episode can last.
            sample_func (function): Function to sample from the buffer.
        """
        self.env_params = env_params
        self.size = 0
        self.max_timesteps = max_timesteps
        self.capacity = capacity
        self.sample_func = sample_func # In this case, HER sampler
        self.buffer = {
            "observation": torch.empty([capacity, self.max_timesteps, env_params["obs_dim"]]),
            "action": torch.empty([capacity, self.max_timesteps, env_params["action_dim"]]),
            "reward": torch.empty([capacity, self.max_timesteps, 1]),
            "done": torch.empty([capacity,self.max_timesteps, 1]),
            "new_observation": torch.empty([capacity, self.max_timesteps, env_params["obs_dim"]]),
            "goal": torch.empty([capacity, self.max_timesteps, env_params["goal_dim"]]),
            "achieved_goal": torch.empty([capacity, self.max_timesteps, env_params["goal_dim"]]),
            "achieved_goal_next": torch.empty([capacity, self.max_timesteps, env_params["goal_dim"]]),
            "episode_len": torch.empty([capacity, 1]),
        }

    def store_episode(self, episode_batch) -> None:
        mb_obs, mb_actions, mb_reward, mb_done, mb_next_obs, mb_g, mb_ag, mb_ag_next = episode_batch
        ep_len = mb_actions.shape[0]
        idxs = self._get_storage_idx()
        # store the informations
        self.buffer['observation'][idxs, :ep_len] = mb_obs
        self.buffer['action'][idxs, :ep_len] = mb_actions
        self.buffer["reward"][idxs, :ep_len] = mb_reward
        self.buffer["done"][idxs, :ep_len] = mb_done
        self.buffer['new_observation'][idxs, :ep_len] = mb_next_obs
        self.buffer['goal'][idxs, :ep_len] = mb_g
        self.buffer['achieved_goal'][idxs, :ep_len] = mb_ag
        self.buffer['achieved_goal_next'][idxs, :ep_len] = mb_ag_next
        self.buffer['episode_len'][idxs] = ep_len



    def _get_storage_idx(self, inc=1):
        """Get the storage index.
        If the replay buffer is not full, return the next available index.
        If the replay buffer is full, replace old trajectories in a random way.
        """
        inc = inc or 1
        if self.size+inc <= self.capacity:
            idx = np.arange(self.size, self.size+inc)
        elif self.size < self.capacity:
            overflow = inc - (self.capacity - self.size)
            idx_a = np.arange(self.size, self.capacity)
            idx_b = np.random.randint(0, self.size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.capacity, inc)
        self.size = min(self.capacity, self.size+inc)
        if inc == 1:
            idx = idx[0]
        return idx


    def sample(self, num_episodes=32) -> List[torch.Tensor]:
        """Sample a batch of episodes from the buffer.
        Args:
            num_episodes (int): The number of episodes to sample.
        Returns:
            List[torch.Tensor]: A list of tensors containing the sampled episodes.
        """
        if self.sample_func is not None:
            temp_buffer = {}
            for key in self.buffer.keys():
                temp_buffer[key] = self.buffer[key][:self.size]
            return self.sample_func(temp_buffer, num_episodes)
        sampled_indices = random.sample(range(self.size), num_episodes)
        observations = self.buffer["observation"][sampled_indices].to(device)
        actions = self.buffer["action"][sampled_indices].to(device)
        rewards = self.buffer["reward"][sampled_indices].to(device)
        dones = self.buffer["done"][sampled_indices].to(device)
        new_observations = self.buffer["new_observation"][sampled_indices].to(
            device
        )
        goal = self.buffer['goal'][sampled_indices].to(device)
        achieved_goal = self.buffer['achieved_goal'][sampled_indices].to(device)
        next_achieved_goal = self.buffer['achieved_goal'][sampled_indices, 1:, :].to(device)
        n_steps = self.buffer['episode_len'][sampled_indices].to(device)
        return [observations, actions, rewards, dones, new_observations, goal, achieved_goal, next_achieved_goal, n_steps]

