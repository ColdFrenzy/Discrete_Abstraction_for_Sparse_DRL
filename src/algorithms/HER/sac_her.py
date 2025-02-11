import torch
import os
from tqdm import tqdm
from collections import deque
from datetime import datetime
import numpy as np
import gymnasium as gym

from src.algorithms.SAC.buffer import EpisodicBuffer
from src.algorithms.SAC.sac import SAC
from src.algorithms.HER.her import her_sampler


"""
SAC with HER
modified from https://github.com/TianhongDai/hindsight-experience-replay/
"""


class SAC_HER(SAC):
    """
    Soft Actor Critic with Hindsight Experience Replay
    in this implementation SAC has 2 critics and 2 target critics
    """
    def __init__(self,
        name,
        agent_name,
        env: gym.Env,
        window=100,
        polyak=0.995,
        pi_lr=0.0005,
        q_lr=0.0005,
        target_update_freq=2,
        value_update_freq=1,
        policy_update_freq=1,
        alpha_initial=4,
        alpha_final=0.05,
        batch_size=128,
        num_batches=40,
        gamma=0.1,
        max_episodes=200,
        max_ep_alpha_decay=200
    ):
        
        super().__init__(
            name,
            agent_name,
            env,
            window=window,
            polyak=polyak,
            pi_lr=pi_lr,
            q_lr=q_lr,
            target_update_freq=target_update_freq,
            value_update_freq=value_update_freq,
            policy_update_freq=policy_update_freq,
            alpha_initial=alpha_initial,
            alpha_final=alpha_final,
            batch_size=batch_size,
            num_batches=num_batches,
            gamma=gamma,
            max_episodes=max_episodes,
            max_ep_alpha_decay=max_ep_alpha_decay
        )

        # her sampler
        self.her_module = her_sampler("future", 4, self.env.reward_function_batch)
        # create the replay buffer
        self.memory = EpisodicBuffer(self.env_params, max_timesteps=self.env_params["max_steps"], sample_func=self.her_module.sample_her_transitions)
        # create the normalizer
        # self.o_norm = normalizer(size=env_params['obs'], default_clip_range=self.args.clip_range)
        # self.g_norm = normalizer(size=env_params['goal'], default_clip_range=self.args.clip_range)

    # do the evaluation
    def _eval_agent(self):
        total_success_rate = []
        for _ in range(self.args.n_test_rollouts):
            per_success_rate = []
            observation = self.env.reset()
            obs = observation['observation']
            g = observation['desired_goal']
            for _ in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    input_tensor = self._preproc_inputs(obs, g)
                    pi = self.actor_network(input_tensor)
                    # convert the actions
                    actions = pi.detach().cpu().numpy().squeeze()
                observation_new, _, _, info = self.env.step(actions)
                obs = observation_new['observation']
                g = observation_new['desired_goal']
                per_success_rate.append(info['is_success'])
            total_success_rate.append(per_success_rate)
        total_success_rate = np.array(total_success_rate)
        local_success_rate = np.mean(total_success_rate[:, -1])
        global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
        return global_success_rate / MPI.COMM_WORLD.Get_size()