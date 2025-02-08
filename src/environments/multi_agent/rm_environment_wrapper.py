from typing import List
from src.algorithms.agent_rl_cont import AgentRL
from src.environments.base_environment import BaseEnvironment


class RMEnvironmentWrapper:
    """
    A wrapper for an environment to manage interactions with Reward Machines (RM) for multiple agents.

    This wrapper intercepts the environment's step and reset functions to update the Reward Machines
    and their associated states and rewards for each agent.
    """

    def __init__(self, env: BaseEnvironment, agents: List[AgentRL]):
        """
        Initialize the RMEnvironmentWrapper.

        Args:
            env (BaseEnvironment): The environment to wrap.
            agents (list): A list of agents interacting with the environment.
        """
        self.env = env
        self.agents = agents  # List of agents

    def reset(self):
        """
        Reset the environment and the Reward Machines to their initial states.

        Returns:
            tuple: A tuple containing observations and infos dictionaries.
        """
        observations, infos = self.env.reset()
        for agent in self.agents:
            agent.get_reward_machine().reset_to_initial_state()
        return observations, infos

    def step(self, actions):
        """
        Perform a step in the environment and update the Reward Machines.

        Args:
            actions (dict): A dictionary mapping agent names to their respective actions.

        Returns:
            tuple: A tuple containing observations, rewards, terminations, truncations, and infos dictionaries.
        """
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        for agent in self.agents:
            rm = agent.get_reward_machine()
            # current_state = infos[agent.name]["prev_s"]
            # next_state_ = infos[agent.name]["s"]  
            # observations[agent.name] == next_state
            next_state = observations[agent.name]
            state_rm = rm.get_current_state()
            reward_rm = rm.step(next_state)  # Get the reward from the RM
            new_state_rm = rm.get_current_state()

            # print(reward_rm, next_state, new_state_rm, "oooooooo")
            # Aggiungi il reward shaping qui
            potentials = rm.potentials
            if potentials is not None:
                gamma = rm.gamma
                rs = gamma * potentials[new_state_rm] - potentials[state_rm]
                reward_rm += rs

            rewards[agent.name] += reward_rm
            # if not terminations[agent.name]:
                # rewards[agent.name] += -1.  # Penalità per ogni step
            infos[agent.name]["RQ"] = reward_rm  # Add RM reward information
            infos[agent.name]["prev_q"] = state_rm  # RM state before the event
            infos[agent.name]["q"] = new_state_rm  # RM state after the event
            # if (
            #     isinstance(agent.get_learning_algorithm(), QLearning)
            #     and agent.get_learning_algorithm().use_qrm
            # ):
            #     qrm_experiences = self._get_qrm_experiences(
            #         agent,
            #         # current_state,
            #         # next_state,
            #         actions[agent.name],
            #         rewards[agent.name],
            #         new_state_rm,
            #         terminations,
            #     )
            #     infos[agent.name]["qrm_experience"] = qrm_experiences

        terminations, truncations = self.check_terminations()
        return observations, rewards, terminations, truncations, infos

    def check_terminations(self):
        """
        Check for terminations and truncations in the environment and the Reward Machines.

        Returns:
            tuple: A tuple containing terminations and truncations dictionaries.
        """
        terminations, truncations = self.env.check_terminations()
        for agent in self.agents:
            rm = agent.get_reward_machine()
            if rm.get_current_state() == rm.get_final_state():
                terminations[agent.name] = True
                truncations[agent.name] = True
        return terminations, truncations

    def _get_qrm_experiences(
        self,
        agent,
        current_state,
        next_state,
        action,
        reward,
        next_rm_state,
        terminations,
    ):
        qrm_experiences = []
        action_index = agent.actions_idx(action)
        rm = agent.get_reward_machine()
        all_states = rm.get_all_states()[:-1]
        final_state = rm.get_final_state()
        for state_rm in all_states:
            event = rm.event_detector.detect_event(next_state)
            (
                hypothetical_next_state,
                hypothetical_reward,
            ) = rm.get_reward_for_non_current_state(state_rm, event)

            # Aggiungi ulteriori debug per vedere quale evento viene rilevato
            # print(f"State RM: {state_rm}, Event: {event}, Hypothetical Next State: {hypothetical_next_state}, Hypothetical Reward: {hypothetical_reward}")

            # Salta se lo stato ipotetico successivo è None
            if hypothetical_next_state is None:
                hypothetical_next_state = state_rm
            # print(current_state, state_rm, next_state, hypothetical_next_state, "weeeeeeeeeeeeee")
            encoded_state, _ = agent.encoder.encode(current_state, state_rm)
            encoded_next_state, _ = agent.encoder.encode(
                next_state, hypothetical_next_state
            )
            # print(f"QRM Experience: state_rm={state_rm}, event={event}, hypo_next_state={hypothetical_next_state}, hypo_reward={hypothetical_reward}")
            """if event != None:
                print("current_state:", current_state, "next_state:", next_state, "state_rm:", state_rm, "hypothetical_next_state:", hypothetical_next_state, "hypothetical_reward:", hypothetical_reward)"""

            # Verifica se lo stato ipotetico successivo è uno stato terminale
            # done = hypothetical_next_state == agent.get_reward_machine().get_final_state()
            rm_done = hypothetical_next_state == final_state
            env_done = terminations[
                agent.name
            ]  # or next_rm_state == agent.get_reward_machine().get_final_state()
            done = rm_done or env_done

            """if agent.ma_problem.agent_fail[agent.name] or agent.ma_problem.timestep > 1000:
                done = True"""

            # Calcolo del reward shaping
            potentials = rm.potentials
            if potentials is not None:
                rs = gamma * potentials[hypothetical_next_state] - potentials[state_rm]
                hypothetical_reward += rs

            qrm_experience = (
                encoded_state,
                action_index,
                hypothetical_reward,
                encoded_next_state,
                done,
            )
            qrm_experiences.append(qrm_experience)
            """if rm_done:
                print("qrm_experience:", qrm_experience, "env_done:", env_done, "rm_done", rm_done)
                breakpoint()"""
            # print(f"QRM Experience: state_rm={state_rm}, event={event}, hypo_next_state={hypothetical_next_state}, hypo_reward={hypothetical_reward}, encoded_state={encoded_state}, encoded_next_state={encoded_next_state}, done={done}")
        return qrm_experiences
