import os
from collections import deque
from copy import deepcopy

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import wandb
from src.utils.paths import WEIGHTS_DIR
from src.algorithms.SAC.buffer import EpisodicBuffer
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
        num_batches=40,
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
            num_batches (int, optional): Number of mini-batches per update. Defaults to 40.
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
        self.num_batches = num_batches
        self.max_episodes = max_episodes
        self.max_ep_alpha_decay = max_ep_alpha_decay
        self.total_win = 0

        # Alpha discounting
        self.alpha_update = ExponentialDiscountScheduler(
            alpha_initial, alpha_final, self.max_ep_alpha_decay
        )

        # env params for networks and buffer
        observation = env.reset()[0]
        # observation and goal has the same shape so obs_dim is doubled
        self.env_params = {
            "obs_dim": observation.shape[0],
            "goal_dim": observation.shape[0],
            "obs_goal_dim": observation.shape[0] * 2,
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
        self.memory = EpisodicBuffer(self.env_params, max_timesteps=self.env_params["max_steps"])
        # self.start_steps = batch_size

    def train(self):
        # Life stats
        self.ep = 1
        self.training = True
        agent_name = self.agent_name

        # Populating the experience replay memory
        for _ in range(self.batch_size):
            observation = self.env.reset()[0]
            self.rollout_episode(observation)      
        # start to collect samples

        with tqdm(total=self.max_episodes) as pbar:
            for i in range(self.max_episodes):
                # reset the environment
                observation = self.env.reset()[0]
                # start to collect samples
                info = self.rollout_episode(observation)
                if info["log"] == "GOAL REACHED":
                    self.total_win += 1
                value_loss_1_mean, value_loss_2_mean, policy_loss_mean, entropy_loss_mean = [], [], [], []
                for _ in range(self.num_batches):
                    value_loss1_value, value_loss2_value , policy_loss_value, entropy_loss_value = self.learning_step()
                    value_loss_1_mean.append(value_loss1_value)
                    value_loss_2_mean.append(value_loss2_value)
                    policy_loss_mean.append(policy_loss_value)
                    entropy_loss_mean.append(entropy_loss_value)

                mean_losses = {
                    "value_loss1": np.mean(value_loss_1_mean),
                    "value_loss2": np.mean(value_loss_2_mean),
                    "policy_loss": np.mean(policy_loss_mean),
                    "entropy_loss": np.mean(entropy_loss_mean),
                    "total_win": self.total_win,
                }
                self.episode_update(pbar, info, mean_losses)
                if self.ep % 1000 == 0:
                    self.env.render_episode(self)
                    self.env.save_episode(self.ep, name=f"uav_{self.agent_name}_cont")
                    self.save()

    def rollout_episode(self, observation: np.ndarray) -> tuple:
        """
        Function responsible for the interaction of the agent with the
        environment. The action is selected by the policy network, then
        performed and the results stored in the replay buffer. It expects a
        numpy array as input.
        """
        done = False
        ep_obs, ep_actions, ep_rewards, ep_done, ep_next_obs, ep_g, ep_ag, ep_ag_next  = [], [], [], [], [], [], [], []
        self.num_steps = 0
        with torch.no_grad():
            while not done:
                observation = torch.as_tensor(observation, dtype=torch.float32).to(
                    device
                )
                achieved_goal = observation.clone()
                goal = torch.tensor(self.env.goal).to(device)
                obs_goal = torch.cat((observation, goal), dim=0)
                action = self.select_action(obs_goal).to(device)
                new_observation, reward, terminated, truncated, info = self.env.step(
                    action.cpu().numpy()
                )
                if (truncated): 
                    self.num_steps = self.env._max_episode_steps
                    done = truncated
                    continue
                ep_obs.append(observation)
                ep_rewards.append(torch.tensor(reward))
                ep_done.append(torch.tensor(terminated or truncated))
                ep_ag.append(achieved_goal)
                ep_g.append(goal)
                ep_actions.append(action)
                done = terminated or truncated
                new_observation = torch.as_tensor(
                    new_observation, dtype=torch.float32
                ).to(device)
                ep_next_obs.append(new_observation)
                ep_ag_next.append(new_observation.clone())
                self.num_steps += 1
                # self.memory.store(observation, action, reward, done, new_observation)
        
        ep_obs = torch.stack(ep_obs)
        ep_actions = torch.stack(ep_actions)
        ep_rewards = torch.stack(ep_rewards).unsqueeze(1)
        ep_done = torch.stack(ep_done).unsqueeze(1)
        ep_next_obs = torch.stack(ep_next_obs)
        ep_g = torch.stack(ep_g)
        ep_ag = torch.stack(ep_ag)
        ep_ag_next = torch.stack(ep_ag_next)
        assert ep_obs.shape[0] == ep_actions.shape[0] == ep_rewards.shape[0] == ep_done.shape[0] == ep_next_obs.shape[0] == ep_g.shape[0] == ep_ag.shape[0], "shape mismatch"
        self.memory.store_episode([ep_obs, ep_actions, ep_rewards, ep_done, ep_next_obs, ep_g, ep_ag, ep_ag_next])
        self.ep_mean_reward = ep_rewards.sum().item() / len(ep_rewards)
        return info

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
        if self.ep % self.value_update_freq == 0:
            value_loss1_value, value_loss2_value = self.value_learning_step(batch)
        if self.ep % self.policy_update_freq == 0:
            policy_loss_value, entropy_loss_value = self.policy_learning_step(batch)
        if self.ep % self.target_update_freq == 0:
            self.update_target_networks()
        
        return value_loss1_value, value_loss2_value , policy_loss_value, entropy_loss_value

    def value_learning_step(self, batch):
        observations= batch["observation"].squeeze(1)
        actions = batch["action"].squeeze(1)
        rewards = batch["reward"].squeeze(1).to(device)
        dones = batch["done"].squeeze(1).to(device)
        new_observations = batch["new_observation"].squeeze(1)
        goal = batch["goal"].squeeze(1)
        achieved_goal = batch["achieved_goal"].squeeze(1)
        next_achieved_goal = batch["achieved_goal_next"].squeeze(1)

        # cat the observation and the goal
        obs_goal = torch.cat((observations, goal), dim=1)
        self.value_optimizer1.zero_grad()
        self.value_optimizer2.zero_grad()

        # Computation of value estimates
        value_estimates1 = self.critic1(obs_goal, actions)
        value_estimates2 = self.critic2(obs_goal, actions)

        new_obs_goal = torch.cat((new_observations, goal), dim=1)
        # Computation of value targets
        with torch.no_grad():
            actions, log_pi = self.actor(
                new_obs_goal
            )  # (batch_size, action_dim)
            log_pi = log_pi.unsqueeze(1)  # hotfix
            target_values = torch.min(
                self.target_critic1(new_obs_goal, actions),
                self.target_critic2(new_obs_goal, actions),
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
        observations= batch["observation"].squeeze(1)
        actions = batch["action"].squeeze(1)
        rewards = batch["reward"].squeeze(1)
        dones = batch["done"].squeeze(1)
        new_observations = batch["new_observation"].squeeze(1)
        goal = batch["goal"].squeeze(1)
        achieved_goal = batch["achieved_goal"].squeeze(1)
        next_achieved_goal = batch["achieved_goal_next"].squeeze(1)
        self.policy_optimizer.zero_grad()

        # Don't waste computational effort
        for param in self.critic1.parameters():
            param.requires_grad = False
        for param in self.critic2.parameters():
            param.requires_grad = False

        obs_goal = torch.cat((observations, goal), dim=1)

        # Policy Optimization
        estimated_actions, log_pi = self.actor(obs_goal)
        estimated_values: torch.Tensor = torch.min(
            self.critic1(obs_goal, estimated_actions),
            self.critic2(obs_goal, estimated_actions),
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

        wandb.log({
            f"{self.agent_name}/mean_ep_reward": self.ep_mean_reward,
            f"{self.agent_name}/ep_num_steps": self.num_steps,
            f"{self.agent_name}/alpha": self.alpha,
            f"{self.agent_name}/loss/value_loss1": episode_mean_losses["value_loss1"],
            f"{self.agent_name}/loss/value_loss2": episode_mean_losses["value_loss2"],
            f"{self.agent_name}/loss/policy_loss": episode_mean_losses["policy_loss"],
            f"{self.agent_name}/loss/entropy_loss": episode_mean_losses["entropy_loss"],
                }
        )

        if pbar is not None:
            pbar.set_description(
                f"Episode {self.ep} Alpha: {self.alpha:.2f}  Ep_Mean_Reward: {self.ep_mean_reward:.2f} Termination: {info['log']}"
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
                        goal = torch.FloatTensor(env.goal)
                        obs_goal = torch.cat((observation, goal), dim=0)
 
                        action, _ = self.actor(
                            obs_goal, deterministic=True, with_logprob=False
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

    def save(self, name):
        here = WEIGHTS_DIR
        path = os.path.join(here, name, self.agent_name, self.name)

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

    def load(self, name, ep=10000):
        here = WEIGHTS_DIR
        path = os.path.join(here, self.agent_name, self.name)

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
