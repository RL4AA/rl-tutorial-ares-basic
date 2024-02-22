import pickle
from datetime import datetime

import gym
import numpy as np
from gym import spaces


class FilterAction(gym.ActionWrapper):
    def __init__(self, env, filter_indicies, replace="random"):
        super().__init__(env)

        self.filter_indicies = filter_indicies
        self.replace = replace

        self.action_space = spaces.Box(
            low=env.action_space.low[filter_indicies],
            high=env.action_space.high[filter_indicies],
            shape=env.action_space.low[filter_indicies].shape,
            dtype=env.action_space.dtype,
        )

    def action(self, action):
        if self.replace == "random":
            unfiltered = self.env.action_space.sample()
        else:
            unfiltered = np.full(
                self.env.action_space.shape,
                self.replace,
                dtype=self.env.action_space.dtype,
            )

        unfiltered[self.filter_indicies] = action

        return unfiltered


class NotVecNormalize(gym.Wrapper):
    """
    Normal Gym wrapper that replicates the functionality of Stable Baselines3's
    VecNormalize wrapper for non VecEnvs (i.e. `gym.Env`) in production.
    """

    def __init__(self, env, path):
        super().__init__(env)

        with open(path, "rb") as file_handler:
            self.vec_normalize = pickle.load(file_handler)

    def reset(self):
        observation = self.env.reset()
        return self.vec_normalize.normalize_obs(observation)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        observation = self.vec_normalize.normalize_obs(observation)
        reward = self.vec_normalize.normalize_reward(reward)
        return observation, reward, done, info


class PolishedDonkeyCompatibility(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        self.observation_space = spaces.Box(
            low=np.array(
                [
                    super().observation_space.low[4],
                    super().observation_space.low[5],
                    super().observation_space.low[7],
                    super().observation_space.low[6],
                    super().observation_space.low[8],
                    super().observation_space.low[9],
                    super().observation_space.low[11],
                    super().observation_space.low[10],
                    super().observation_space.low[12],
                    super().observation_space.low[0],
                    super().observation_space.low[2],
                    super().observation_space.low[1],
                    super().observation_space.low[3],
                ]
            ),
            high=np.array(
                [
                    super().observation_space.high[4],
                    super().observation_space.high[5],
                    super().observation_space.high[7],
                    super().observation_space.high[6],
                    super().observation_space.high[8],
                    super().observation_space.high[9],
                    super().observation_space.high[11],
                    super().observation_space.high[10],
                    super().observation_space.high[12],
                    super().observation_space.high[0],
                    super().observation_space.high[2],
                    super().observation_space.high[1],
                    super().observation_space.high[3],
                ]
            ),
        )

        self.action_space = spaces.Box(
            low=np.array([-30, -30, -30, -3e-3, -6e-3], dtype=np.float32) * 0.1,
            high=np.array([30, 30, 30, 3e-3, 6e-3], dtype=np.float32) * 0.1,
        )

    def reset(self):
        return self.observation(super().reset())

    def step(self, action):
        observation, reward, done, info = super().step(self.action(action))
        return self.observation(observation), reward, done, info

    def observation(self, observation):
        return np.array(
            [
                observation[4],
                observation[5],
                observation[7],
                observation[6],
                observation[8],
                observation[9],
                observation[11],
                observation[10],
                observation[12],
                observation[0],
                observation[2],
                observation[1],
                observation[3],
            ]
        )

    def action(self, action):
        return np.array(
            [
                action[0],
                action[1],
                action[3],
                action[2],
                action[4],
            ]
        )


class RecordEpisode(gym.Wrapper):
    """
    Wrapper for recording epsiode data such as observations, rewards, infos and actions.
    """

    def __init__(self, env):
        super().__init__(env)

        self.has_previously_run = False

    def reset(self):
        if self.has_previously_run:
            self.previous_observations = self.observations
            self.previous_rewards = self.rewards
            self.previous_infos = self.infos
            self.previous_actions = self.actions
            self.previous_t_start = self.t_start
            self.previous_t_end = datetime.now()
            self.previous_steps_taken = self.steps_taken

        observation = self.env.reset()

        self.observations = [observation]
        self.rewards = []
        self.infos = []
        self.actions = []
        self.t_start = datetime.now()
        self.t_end = None
        self.steps_taken = 0

        self.has_previously_run = True

        return observation

    def step(self, action):
        observation, reward, done, info = self.env.step(action)

        self.observations.append(observation)
        self.rewards.append(reward)
        self.infos.append(info)
        self.actions.append(action)
        self.steps_taken += 1

        return observation, reward, done, info
