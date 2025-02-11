import numpy as np
import torch
class her_sampler:
    def __init__(self, replay_strategy, replay_k, reward_func=None):
        """Initialize the HER sampler.
        Args:
            replay_strategy (str): Strategy for sampling HER transitions.
                Supported strategies are "future" which replaces goals with achieved goals in the future of the same episode,
            replay_k (int): Ratio between HER replays and regular replays.
            reward_func (function): Function to compute the reward.
        """
        self.replay_strategy = replay_strategy
        self.replay_k = replay_k
        # default is 4 so future_p = 0.8
        if self.replay_strategy == 'future':
            self.future_p = 1 - (1. / (1 + replay_k))
        else:
            self.future_p = 0
        self.reward_func = reward_func

    def sample_her_transitions(self, episode_batch, num_batch):
        """Sample HER transitions.
        Args:
            episode_batch (dict): number of episodes
            batch_size_in_transitions (int): size of the batch inside an episode.
        Returns:
            transitions (dict): Transitions with HER. size is [num_batch, dim]
        """
        T = episode_batch["episode_len"]
        # length of the EpisodicBuffer up to now
        rollout_batch_size = episode_batch['action'].shape[0]
        # number of episodes to take
        episode_idxs = torch.randint(rollout_batch_size, (num_batch,1))
        # number of samples from each episode
        t_samples = []
        for idx in episode_idxs:
            t_samples.append(torch.randint(int(T[idx]), (1,)))
        t_samples = torch.stack(t_samples)
        T = T[episode_idxs].squeeze(1)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].clone() for key in episode_batch.keys() if key != 'episode_len'}
        # her idx (episodes to modify)
        her_indexes = torch.where(torch.rand(num_batch) < self.future_p)
        # T - t_samples is the number of steps left in the episode
        # future_offset indicates how many steps in the future to replace the goal
        future_offset = torch.rand(num_batch).unsqueeze(1) * (T - t_samples)
        future_offset = future_offset.to(int)
        # future_t is the time step in the future to replace the goal
        future_t = (t_samples + future_offset)[her_indexes] # (t_samples + 1 + future_offset)[her_indexes]
        # replace goal with achieved goal from the original episode
        future_ag = episode_batch['achieved_goal'][episode_idxs[her_indexes], future_t]
        transitions['goal'][her_indexes] = future_ag
        # to get the params to re-compute reward
        # the reward function compute the distance between the achieved goal and the goal
        # in the case of sparse reward, the reward is 1 if the distance is less than a treshold and 0 otherwise
        # in the case of dense reward, the reward is the negative distance
        transitions['reward'] = self.reward_func(transitions['achieved_goal_next'].squeeze(1), transitions['goal'].squeeze(1), None).unsqueeze(1)
        transitions = {k: transitions[k].reshape(num_batch, *transitions[k].shape[1:]) for k in transitions.keys()}

        return transitions

    def sample_her_transitions_old(self, episode_batch, batch_size_in_transitions):
        def reward_fun(ag_next, g, info):
            return np.array([np.linalg.norm(ag_next[i] - g[i], axis=-1) < 0.05 for i in range(ag_next.shape[0])]).astype(np.float32)
        T = 100
        rollout_batch_size = episode_batch['actions'].shape[0]
        batch_size = batch_size_in_transitions
        # select which rollouts and which timesteps to be used
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy() for key in episode_batch.keys() if key != 'episode_len'}
        # her idx
        her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]
        # replace go with achieved goal
        future_ag = episode_batch['achieved_goal'][episode_idxs[her_indexes], future_t]
        transitions['goal'][her_indexes] = future_ag
        # to get the params to re-compute reward
        transitions['reward'] = np.expand_dims(reward_fun(transitions['achieved_goal_next'], transitions['goal'], None), 1)
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}

        return transitions



if __name__ == "__main__":
    # Test the HER sampler
    sampler = her_sampler(replay_strategy='future', replay_k=4)
    episode_batch = {
        'observation': np.random.rand(10, 100, 10),
        'achieved_goal': np.random.rand(10, 100, 3),
        'desired_goal': np.random.rand(10, 100, 3),
        'actions': np.random.rand(10, 100, 4),
        'ag_next': np.random.rand(10, 100, 3),
        'goal': np.random.rand(10, 100, 3),
        'episode_len': np.random.randint(0, 100, size=10),
        'achieved_goal_next': np.random.rand(10, 100, 3)
    }
    transitions = sampler.sample_her_transitions_old(episode_batch, 5)