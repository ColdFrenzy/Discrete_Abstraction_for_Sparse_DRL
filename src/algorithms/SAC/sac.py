import os
from collections import deque
from copy import deepcopy

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import wandb
from src.algorithms.SAC.buffer import StandardReplayBuffer
from src.algorithms.SAC.networks import Actor, Critic, SquashedGaussianActor
from src.utils.scheduler import (
    ExponentialDiscountScheduler,
    LinearDiscountScheduler,
    SigmoidDiscountScheduler
)
from src.utils.plot import save_reward_plots, save_steps_plot

# Seed
SEED = 13
torch.manual_seed(SEED)
torch.random.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SAC:
    def __init__(
        self,
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
        gamma=0.1,
        max_episodes=200,
        max_ep_alpha_decay=200
    ):
        """
        Initializes the Soft Actor-Critic (SAC) algorithm.

        Args:
            name (str): The name of the agent.
            env (gym.Env): The environment in which the agent will operate.
            window (int, optional): The window size for averaging rewards. Defaults to 100.
            polyak (float, optional): Interpolation factor in polyak averaging for target networks. Defaults to 0.995.
            pi_lr (float, optional): Learning rate for the policy network. Defaults to 0.0005.
            q_lr (float, optional): Learning rate for the Q-value networks. Defaults to 0.0005.
            target_update_freq (int, optional): Frequency of target network updates. Defaults to 2.
            value_update_freq (int, optional): Frequency of value network updates. Defaults to 1.
            policy_update_freq (int, optional): Frequency of policy network updates. Defaults to 1.
            alpha_initial (float, optional): Initial value of the entropy coefficient. Defaults to 4.
            alpha_final (float, optional): Final value of the entropy coefficient. Defaults to 0.05.
            batch_size (int, optional): Size of the mini-batch for updates. Defaults to 128.
            gamma (float, optional): Discount factor for future rewards. Defaults to 0.1.
            max_episodes (int, optional): Maximum number of episodes for training. Defaults to 200.
        """

        # Hyperparameters
        self.name = name
        self.agent_name = agent_name
        self.env = env
        self.window = window
        self.polyak = polyak
        self.pi_lr = pi_lr
        self.q_lr = q_lr
        self.target_update_freq = target_update_freq
        self.value_update_freq = value_update_freq
        self.policy_update_freq = policy_update_freq
        self.alpha = alpha_initial
        self.batch_size = batch_size
        self.gamma = gamma
        self.max_episodes = max_episodes
        self.max_ep_alpha_decay = max_ep_alpha_decay

        # Alpha discounting
        self.alpha_update = ExponentialDiscountScheduler(
            alpha_initial, alpha_final, self.max_ep_alpha_decay
        )

        # env params for networks and buffer
        observation = env.reset()[0]
        self.env_params = {
            "obs_dim": observation.shape[0],
            "action_dim": env.action_space.shape[0],
            "action_bound": env.action_space.high[0],
            "max_steps": env._max_episode_steps,
        }

        # Networks
        self.actor: SquashedGaussianActor = SquashedGaussianActor(
            self.env_params
        ).to(device)
        self.critic1: Critic = Critic(self.env_params).to(device)
        self.critic2: Critic = Critic(self.env_params).to(device)

        self.policy_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=pi_lr
        )
        self.value_optimizer1 = torch.optim.Adam(
            self.critic1.parameters(), lr=q_lr
        )
        self.value_optimizer2 = torch.optim.Adam(
            self.critic2.parameters(), lr=q_lr
        )
        self.value_loss_fn = nn.MSELoss()

        self.target_actor: Actor = deepcopy(self.actor).to(device)
        self.target_critic1: Critic = deepcopy(self.critic1).to(device)
        self.target_critic2: Critic = deepcopy(self.critic2).to(device)

        # Target networks must be updated not directly through the gradients but
        # with polyak averaging
        for param in self.target_critic1.parameters():
            param.requires_grad = False
        for param in self.target_critic2.parameters():
            param.requires_grad = False
        for param in self.target_actor.parameters():
            param.requires_grad = False

        # Experience Replay Buffer
        self.memory = StandardReplayBuffer(self.env_params)
        self.start_steps = batch_size

    def train(self):
        # Life stats
        self.ep = 1
        self.training = True
        self.reward_buffer = deque(maxlen=self.window)
        self.num_steps_buffer = deque(maxlen=self.window)
        agent_name = self.agent_name
        rewards_per_episode = {agent_name: []}
        total_steps = []

        # Populating the experience replay memory
        self.memory.populate(self.env, self.start_steps)

        with tqdm(total=self.max_episodes) as pbar:
            for i in range(self.max_episodes):
                # ep stats
                self.num_steps = 0
                self.ep_reward = 0

                # ep termination
                done = False

                # starting point
                observation = self.env.reset()[0]
                episode_mean_value_loss1, episode_mean_value_loss2, episode_mean_policy_loss, episode_mean_entropy_loss = 0, 0, 0, 0
                while not done:
                    new_observation, done, info = self.interaction_step(
                        observation
                    )
                    value_loss1_value, value_loss2_value , policy_loss_value, entropy_loss_value = self.learning_step()
                    episode_mean_value_loss1 += value_loss1_value
                    episode_mean_value_loss2 += value_loss2_value
                    episode_mean_policy_loss += policy_loss_value
                    episode_mean_entropy_loss += entropy_loss_value
                    observation = new_observation
                    self.num_steps += 1
                # Aggiungi la ricompensa totale per ogni agente
                # per ora NON è MULTI AGENT
                # for agent in self.env.agents:
                #     rewards_per_episode[agent.name].append(self.ep_reward)
                episode_mean_losses = {
                    "value_loss1": episode_mean_value_loss1 / self.num_steps,
                    "value_loss2": episode_mean_value_loss2 / self.num_steps,
                    "policy_loss": episode_mean_policy_loss / self.num_steps,
                    "entropy_loss": episode_mean_entropy_loss / self.num_steps,
                }
                rewards_per_episode[agent_name].append(self.ep_reward)
                total_steps.append(self.num_steps)

                # Salva i grafici delle ricompense all'ultimo episodio
                if self.ep == self.max_episodes: # se vuoi salvare ogni 100 episodi(self.ep % 100 == 0 or self.ep == self.max_episodes):
                    save_reward_plots(
                        rewards_per_episode,  # Passa il dizionario con le ricompense
                        save_path="plots",  # Percorso dove salvare
                    )
                    save_steps_plot(total_steps, save_path="plots")

                self.episode_update(pbar, info, episode_mean_losses)
                if self.ep % 1000 == 0:
                    self.env.render_episode(self)
                    self.env.save_episode(self.ep, name=f"uav_{self.agent_name}_cont")
                    self.save()

    def interaction_step(self, observation: np.ndarray) -> tuple:
        """
        Function responsible for the interaction of the agent with the
        environment. The action is selected by the policy network, then
        performed and the results stored in the replay buffer. It expects a
        numpy array as input.
        """
        observation = torch.as_tensor(observation, dtype=torch.float32).to(
            device
        )
        action = self.select_action(observation).to(device)
        new_observation, reward, terminated, truncated, info = self.env.step(
            action.cpu().numpy()
        )
        done = terminated or truncated
        if (
            truncated
        ):  # As if the episode was terminated (fell in a hole for example)
            self.num_steps = self.env._max_episode_steps
        new_observation = torch.as_tensor(
            new_observation, dtype=torch.float32
        ).to(
            device
        )  # buffer expects tensor
        self.memory.store(observation, action, reward, done, new_observation)
        self.ep_reward += reward
        return new_observation, done, info

    def select_action(self, observation: torch.Tensor) -> torch.Tensor:
        """
        This function selects an action from the policy network. It expects
        to receive a tensor in input.
        """
        with torch.no_grad():
            action, _ = self.actor(observation, with_logprob=False)
        return action

    def learning_step(self) -> bool:
        # Sampling of the minibatch
        batch = self.memory.sample(batch_size=self.batch_size)

        # Learning step
        if self.num_steps % self.value_update_freq == 0:
            value_loss1_value, value_loss2_value = self.value_learning_step(batch)
        if self.num_steps % self.policy_update_freq == 0:
            policy_loss_value, entropy_loss_value = self.policy_learning_step(batch)
        if self.num_steps % self.target_update_freq == 0:
            self.update_target_networks()
        
        return value_loss1_value, value_loss2_value , policy_loss_value, entropy_loss_value

    def value_learning_step(self, batch):
        observations, actions, rewards, dones, new_observations = batch

        self.value_optimizer1.zero_grad()
        self.value_optimizer2.zero_grad()

        # Computation of value estimates
        value_estimates1 = self.critic1(observations, actions)
        value_estimates2 = self.critic2(observations, actions)

        # Computation of value targets
        with torch.no_grad():
            actions, log_pi = self.actor(
                new_observations
            )  # (batch_size, action_dim)
            log_pi = log_pi.unsqueeze(1)  # hotfix
            target_values = torch.min(
                self.target_critic1(new_observations, actions),
                self.target_critic2(new_observations, actions),
            )
            targets = rewards + (1 - dones) * self.gamma * (
                target_values - self.alpha * log_pi
            )

        # MSBE
        value_loss1: torch.Tensor = self.value_loss_fn(
            value_estimates1, targets
        )
        value_loss2: torch.Tensor = self.value_loss_fn(
            value_estimates2, targets
        )
        value_loss1_value = value_loss1.item()
        value_loss1.backward()
        self.value_optimizer1.step()
        value_loss2_value = value_loss2.item()
        value_loss2.backward()
        self.value_optimizer2.step()
        return value_loss1_value, value_loss2_value

    def policy_learning_step(self, batch):
        observations, _, _, _, _ = batch
        self.policy_optimizer.zero_grad()

        # Don't waste computational effort
        for param in self.critic1.parameters():
            param.requires_grad = False
        for param in self.critic2.parameters():
            param.requires_grad = False

        # Policy Optimization
        estimated_actions, log_pi = self.actor(observations)
        estimated_values: torch.Tensor = torch.min(
            self.critic1(observations, estimated_actions),
            self.critic2(observations, estimated_actions),
        )
        policy_loss: torch.Tensor = (
            self.alpha * log_pi - estimated_values
        ).mean()  # perform gradient ascent
        entropy_loss_value = self.alpha * log_pi.mean().item()
        policy_loss_value = policy_loss.item()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Reactivate computational graph for critic
        for param in self.critic1.parameters():
            param.requires_grad = True
        for param in self.critic2.parameters():
            param.requires_grad = True
        return policy_loss_value, entropy_loss_value

    def update_target_networks(self, polyak=None):
        polyak = self.polyak if polyak is None else polyak
        with torch.no_grad():
            for target, online in zip(
                self.target_critic1.parameters(), self.critic1.parameters()
            ):
                target.data.mul(polyak)
                target.data.add((1 - polyak) * online.data)

            for target, online in zip(
                self.target_critic2.parameters(), self.critic2.parameters()
            ):
                target.data.mul(polyak)
                target.data.add((1 - polyak) * online.data)

            for target, online in zip(
                self.target_actor.parameters(), self.actor.parameters()
            ):
                target.data.mul(polyak)
                target.data.add((1 - polyak) * online.data)

    def episode_update(self, pbar: tqdm = None, info: dict = None, episode_mean_losses: dict = None):
        self.reward_buffer.append(self.ep_reward)
        self.num_steps_buffer.append(self.num_steps)
        mean_num_steps = np.mean(self.num_steps_buffer)
        mean_reward = np.mean(self.reward_buffer)

        wandb.log({
            f"{self.agent_name}/reward": self.ep_reward,
            f"{self.agent_name}/mean_reward": mean_reward,
            f"{self.agent_name}/num_steps": self.num_steps,
            f"{self.agent_name}/mean_num_steps": mean_num_steps,
            f"{self.agent_name}/alpha": self.alpha,
            f"{self.agent_name}/loss/value_loss1": episode_mean_losses["value_loss1"],
            f"{self.agent_name}/loss/value_loss2": episode_mean_losses["value_loss2"],
            f"{self.agent_name}/loss/policy_loss": episode_mean_losses["policy_loss"],
            f"{self.agent_name}/loss/entropy_loss": episode_mean_losses["entropy_loss"],
                }
        )

        if pbar is not None:
            pbar.set_description(
                f"Episode {self.ep} Alpha: {self.alpha:.2f} Mean Reward: {mean_reward:.2f} Ep_Reward: {self.ep_reward:.2f} Termination: {info['log']}"
            )
            pbar.update(1)
        self.ep += 1
        self.alpha = self.alpha_update()

    def evaluate(self, env=None, render: bool = True, num_ep=3):
        mean_reward = 0.0
        if env is None:
            env = self.env

        with tqdm(total=num_ep) as pbar:
            for i in range(num_ep):
                observation = torch.FloatTensor(env.reset()[0])

                terminated = False
                truncated = False
                total_reward = 0

                while not terminated and not truncated:
                    with torch.no_grad():
                        action, _ = self.actor(
                            observation, deterministic=True, with_logprob=False
                        )
                    observation, reward, terminated, truncated, info = (
                        env.step(action.cpu().numpy())
                    )
                    observation = torch.FloatTensor(observation)
                    total_reward += reward
                    if render:
                        self.env.render()

                mean_reward = mean_reward + (1 / (i + 1)) * (
                    total_reward - mean_reward
                )
                pbar.set_description(
                    f"Episode {i+1} Mean Reward: {mean_reward:.2f} Ep_Reward: {total_reward:.2f} Termination: {info['log']}"
                )
                pbar.update(1)
                env.save_episode(i)

        return mean_reward

    def save(self):
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, "models", self.agent_name, self.name)

        os.makedirs(path, exist_ok=True)

        torch.save(
            self.actor.state_dict(), open(os.path.join(path, f"actor_{self.ep}.pt"), "wb")
        )
        torch.save(
            self.critic1.state_dict(),
            open(os.path.join(path, f"critic1_{self.ep}.pt"), "wb"),
        )
        torch.save(
            self.critic2.state_dict(),
            open(os.path.join(path, f"critic2_{self.ep}.pt"), "wb"),
        )
        print("MODELS SAVED!")

    def load(self, ep=10000):
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, "models", self.agent_name, self.name)

        self.actor.load_state_dict(
            torch.load(
                open(os.path.join(path, f"actor_{ep}.pt"), "rb"),
                weights_only=True,
                map_location=device,
            )
        )
        self.critic1.load_state_dict(
            torch.load(
                open(os.path.join(path, f"critic1_{ep}.pt"), "rb"),
                weights_only=True,
                map_location=device,
            )
        )
        self.critic2.load_state_dict(
            torch.load(
                open(os.path.join(path, f"critic2_{ep}.pt"), "rb"),
                weights_only=True,
                map_location=device,
            )
        )
        print("MODELS LOADED!")
