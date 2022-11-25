import gym
import numpy as np

class NormalizedActionsHipConstrained(gym.ActionWrapper):
    def __init__(self, env: gym.Env, factor):
        super().__init__(env)
        low_bound   = self.action_space.low
        upper_bound = self.action_space.high
        self.action_space = gym.spaces.Box(low=np.full_like(low_bound, -1.), high=np.full_like(upper_bound, +1.))
        self.constraint_factor = factor
        self.hip_index = [0,3,6,9]

    def action(self, action):
        low_bound   = self.action_space.low * self.constraint_factor
        upper_bound = self.action_space.high * self.constraint_factor
        for index in self.hip_index:
        	low_bound[index] = low_bound[index] * 0.4
        	upper_bound[index] = upper_bound[index] * 0.4
        action = low_bound + (action + 1.0) * 0.5 * (upper_bound - low_bound)
        action = np.clip(action, low_bound, upper_bound)
        return action

    def reverse_action(self, action):
        low_bound   = self.action_space.low
        upper_bound = self.action_space.high

        action = 2 * (action - low_bound) / (upper_bound - low_bound) - 1
        action = np.clip(action, low_bound, upper_bound)

        return actions
