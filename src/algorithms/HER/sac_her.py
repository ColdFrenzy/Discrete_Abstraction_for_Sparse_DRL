from tqdm import tqdm
import gymnasium as gym

from algorithms.SAC.buffer import EpisodicBuffer
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
    def __init__(
        self,
        name,
        agent_name,
        env: gym.Env,
        window=100,
        polyak=0.995,
        pi_lr=0.0005,
        q_lr=0.0005,
        ep_update_freq=1,
        gradient_steps=1,
        alpha_initial=4,
        alpha_final=0.05,
        batch_size=128,
        gamma=0.1,
        max_episodes=200,
        max_ep_alpha_decay=200,
        buffer_capacity=512,
    ):
        super().__init__(
            name,
            agent_name,
            env,
            window=window,
            polyak=polyak,
            pi_lr=pi_lr,
            q_lr=q_lr,
            ep_update_freq=ep_update_freq,
            gradient_steps=gradient_steps,
            alpha_initial=alpha_initial,
            alpha_final=alpha_final,
            batch_size=batch_size,
            gamma=gamma,
            max_episodes=max_episodes,
            max_ep_alpha_decay=max_ep_alpha_decay,
            buffer_capacity=buffer_capacity,
        )

        # her sampler
        self.her_module = her_sampler("future", 4, self.env.reward_function_batch)
        # create the replay buffer
        self.memory = EpisodicBuffer(self.env_params, max_timesteps=self.env_params["max_steps"], sample_func=self.her_module.sample_her_transitions)

    def train(self):
        """
        train the network
        """
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
                # train the agent at the end of the episode
                value_loss1_value, value_loss2_value , policy_loss_value, entropy_loss_value = self.learning_step()
                mean_losses = {
                    "value_loss1": value_loss1_value,
                    "value_loss2": value_loss2_value,
                    "policy_loss": policy_loss_value,
                    "entropy_loss": entropy_loss_value,
                }
                self.episode_update(pbar, info, mean_losses)
            if self.ep % 1000 == 0:
                self.save()