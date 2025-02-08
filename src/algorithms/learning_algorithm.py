from abc import ABC, abstractmethod

import numpy as np
import time, os
from tqdm import tqdm


class BaseLearningAlgorithm(ABC):
    def __init__(
        self, state_space_size, action_space_size, gamma=0.99, seed=1000, max_steps=100
    ):
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.requires_rm_event = False  # Imposta il valore predefinito
        self.episode = 0
        self.rleval = None
        self.max_steps = max_steps  # max steps per episode
        self.total_steps = 0
        self.verbose = 0
        self.seed = seed
        self.gamma = gamma
        self.runtime = 0
        self.user_quit = False
        self.agent_quit = False
        self.stopongoal = False
        self.rng = np.random.default_rng(seed=2020)
        self.tosave = ["episode", "total_steps", "runtime", "rng"]

    @abstractmethod
    def update(
        self, encoded_state, encoded_next_state, action, reward, rm_state, next_rm_state
    ):
        pass

    @abstractmethod
    def choose_action(self, encoded_state, **kwargs):
        pass

    def save_model(self, expname, aname, models_dir):
        filename = "%s/%s_%s_%06d.dat" % (models_dir, expname, aname, self.episode - 1)

        dict_save = {}
        for k in self.tosave:
            # dict_save[k] = (eval(f"self.{k}"), eval(f"type(self.{k})"))
            dict_save[k] = (self.__dict__[k], type(self.__dict__[k]))
        # print(dict_save)
        rleval_save = {}
        # save also RLeval structures
        if "rleval" in self.__dict__.keys() and self.rleval is not None:
            for k in self.rleval.tosave:
                rleval_save[k] = (
                    self.rleval.__dict__[k],
                    type(self.rleval.__dict__[k]),
                )

        np.savez(filename, dict_save=dict_save, rleval_save=rleval_save)

        print("Model saved on file %s\n" % filename)

    def auto_load_model(self, expname, aname, models_dir):
        m = -1
        for f in os.listdir(models_dir):
            if f[-8:] == ".dat.npz":
                v = f.split("_")
                if v[0] + "_" + v[1] == expname and v[2] == aname:
                    n = int(v[3].split(".")[0])
                    m = max(m, n)
        if m > 0:
            modelfile = "%s/%s_%s_%06d.dat.npz" % (models_dir, expname, aname, m)
            self.load_model(modelfile)
        else:
            print(f"No model found for {expname}_{aname}")

    def delete_models(self, expname, aname, models_dir):
        for f in os.listdir(models_dir):
            if f[-8:] == ".dat.npz":
                v = f.split("_")
                if v[0] + "_" + v[1] == expname and v[2] == aname:
                    os.remove(models_dir + "/" + f)

    def load_model(self, filename):
        try:
            data = np.load(filename, allow_pickle=True)
        except Exception as e:
            print(e)
            return
        dict_save = data["dict_save"].item()
        # print(dict_save)

        for k in dict_save.keys():
            # print(k)
            self.__dict__[k] = dict_save[k][0]
            assert (
                type(self.__dict__[k]) == dict_save[k][1]
            ), f"Loaded types for {k} mismatch {type(self.__dict__[k])} {dict_save[k][1]}"

        if "rleval" in self.__dict__.keys() and self.rleval is not None:
            rleval_save = data["rleval_save"].item()
            for k in rleval_save.keys():
                # print(k)
                self.rleval.__dict__[k] = rleval_save[k][0]
                assert (
                    type(self.rleval.__dict__[k]) == rleval_save[k][1]
                ), f"Loaded types for {k} mismatch {type(self.rleval.__dict__[k])} {rleval_save[k][1]}"

        print("Model loaded from file %s\n" % filename)

    def learn_init(self):
        pass

    def learn_init_episode(self):
        pass

    # def learn_update(self, obs, action: int, reward: float, terminated: bool, next_obs, info):
    #    pass

    def learn_done_episode(self):
        pass

    def learn_end(self):
        pass

    def learn_step(self, env):

        time0 = time.time()

        self.learn_init_episode()

        if self.episode == 0 and self.rleval is not None:  # eval very first episode
            self.rleval.add_mean_rewards()

        # print(f"episode {self.episode:6d}")

        obs, info = env.reset(seed=self.seed + self.episode)

        # play one episode
        t = 0
        done = False
        while not done and t < self.max_steps:
            action = self.choose_action(obs, info=info)
            next_obs, reward, terminated, truncated, info = env.step(action)
            self.total_steps += 1
            t += 1
            if t == self.max_steps:  # truncated = termination with failure
                if self.verbose > 2:
                    s, q = env.decode_SQ(obs)
                    s1, q1 = env.decode_SQ(next_obs)
                    print(
                        "Max steps reached: %d %d %d %d %d  re:%.6f"
                        % (
                            s,
                            q,
                            action,
                            s1,
                            q1,
                            self.sumR[s, action, s1] / self.nRSAS[s, action, s1],
                        )
                    )

            if not truncated:
                # update the agent
                self.update(obs, next_obs, action, reward, terminated, info=info)

            # update if the environment is done and the current obs
            done = terminated or truncated
            obs = next_obs

            # if self.verbose>1:
            #    print("Ep. %d Step %d" %(self.episode,t))

        if (
            self.rleval is not None
            and self.episode > 0
            and self.episode % self.rleval.eval_ep_interval == 0
        ):
            mr, sr, pg = self.rleval.add_mean_rewards()
            if pg > 0.4:
                self.count_goal_reached -= 1
                # print("goal reached.... ", self.count_goal_reached)
            else:
                self.count_goal_reached = self.target_count_goal_reached

            if self.verbose > 0:
                print(f" {self.episode:6d} | {mr:7.3f} +/- {sr:7.3f} | {pg:5.2f} ")

        # r,c = env.rc_state()
        # print(f" -- ep {self.episode:6d} - pos: {r} {c}")

        self.learn_done_episode()

        run_time = time.time() - time0
        self.runtime += run_time

        if self.stopongoal and self.count_goal_reached <= 0:
            return False
        if self.user_quit:
            return False
        if self.agent_quit:
            return False

        return True

    def learn(self, env, n_episodes, n_steps):

        tolearn = (n_episodes > 0 and self.episode <= n_episodes) or (
            n_steps > 0 and self.total_steps <= n_steps
        )

        if tolearn:  # something to do

            self.learn_init()

            self.target_count_goal_reached = (
                10  # stop after this number of evaluations with goal reached
            )
            self.count_goal_reached = (
                self.target_count_goal_reached
            )  # consecutive eval with goal reached

            ra = range(0)

            if n_episodes > 0:
                ia = self.episode
                ta = n_episodes
            elif n_steps > 0:
                ia = self.total_steps
                ta = n_steps
                last_steps = self.total_steps

            if self.verbose == 0:
                pbar = tqdm(initial=ia, total=ta)

            r = True
            while ia < ta and r:
                r = self.learn_step(env)
                self.episode += 1
                if n_episodes > 0:
                    ia = self.episode
                    pu = 1
                elif n_steps > 0:
                    ia = self.total_steps
                    pu = ia - last_steps
                    last_steps = self.total_steps
                if self.verbose == 0:
                    pbar.update(pu)

            if self.verbose == 0:
                pbar.close()

            self.learn_end()

            print(f"\nTraining time: {self.runtime:6.2f} s")

        return tolearn
