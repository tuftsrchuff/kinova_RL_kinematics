from collections import namedtuple
import numpy as np
import abc
from stable_baselines3 import ppo

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state"))


class BaseAgent(object):
    @abc.abstractmethod
    def act(self, observation, reward, done):
        raise NotImplementedError


class RandomAgent(BaseAgent):
    def __init__(self, env):
        self.env = env

    def act(self, _observation, _reward, _done):
        return self.env.sample_action()


class SimpleAgent(BaseAgent):
    def __init__(self, env):
        self.env = env
        self.experience = {}
        self.exploration = 0.1
        self.last_action = self.env.sample_action()

    def act(self, observation, reward, done):
        hashable_observation = tuple(observation)
        if hashable_observation not in self.experience:
            self.experience[hashable_observation] = {}
        if self.last_action not in self.experience[hashable_observation]:
            self.experience[hashable_observation][self.last_action] = 0
        self.experience[hashable_observation][self.last_action] += reward
        if done:
            self.last_action = self.env.sample_action()
        else:
            if np.random.random() < self.exploration:
                self.last_action = self.env.sample_action()
            else:
                self.last_action = max(
                    self.experience[hashable_observation].items(), key=lambda x: x[1]
                )[0]
        return self.last_action


class StableBaselinesAgent(BaseAgent):
    def __init__(self, env):
        self.env = env
        self.model = ppo.PPO("MlpPolicy", env, verbose=1)
        self.model.learn(total_timesteps=10000)

    def act(self, observation, reward, done):
        return self.model.predict(observation)[0]
