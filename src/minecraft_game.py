import warnings

import minerl
import gym
import logging

from gym.spaces import Discrete


class MinecraftGame:

    def __init__(self, id, log_level=None):
        """
        Creates an instance of the MineRL Minecraft gym environment with some helpful methods.
        :param id: ID of the environment to instantiate
        :param log_level: Either string of or logging log level of the logging level to set
        """
        if log_level is not None:
            if isinstance(log_level, str):
                log_level = log_level.lower()
                if log_level == "debug":
                    logging.basicConfig(level=logging.DEBUG)
                elif log_level == "critical":
                    logging.basicConfig(level=logging.CRITICAL)
                elif log_level == "fatal":
                    logging.basicConfig(level=logging.FATAL)
                elif log_level == "error":
                    logging.basicConfig(level=logging.ERROR)
                else:
                    print(f"Log level {log_level} is not supported!")
            else:
                logging.basicConfig(level=log_level)
        self._env = gym.make(id)

        self._observation = self.new_episode()
        self._curr_reward = None
        self.done = False

        self._action = None
        self._action_set_since_last_perform = False

    def get_action_space_size(self):
        """
        Returns an int with the number of actions that can be performed.
        :return: Size of the state space (1D)
        """
        s = 0
        for a in self._env.action_space.spaces.values():
            if isinstance(a, Discrete):
                s += 1
            else:
                s += a.shape[0]
        return s

    def get_frame_size(self):
        """
        Gets the size of the FOV observation frame
        :return: The size of the FOV observation frame (just the first 2 dimensions, not the number of channels)
        """
        return self._env.observation_space["pov"].shape[0:2]

    def new_episode(self):
        """
        Creates a new episode
        :return: None
        """
        self._observation = self._env.reset()
        return self._observation

    def get_state(self):
        """
        Gets the current state/observation. Environment must have been initialized to use this
        :return: The current state
        """
        if self._observation is None:
            raise ValueError("You have to initialize an episode before you can get the state!")
        return self._observation

    def get_frame(self):
        """
        Gets the current observation frame
        :return: The current observation frame in numpy array form
        """
        return self._observation["pov"]

    def is_episode_finished(self):
        """
        Queries if the current episode is finished
        :return: True if the current episode is done, false otherwise
        """
        return self.done

    def perform_action(self):
        """
        Performs the action. The action must be previously set or else a warning/error will be raised
        :return: None
        """
        if self._action is None:
            raise ValueError("Must set action first!")
        if not self._action_set_since_last_perform:
            warnings.warn("Action has not been updated since an action has last been performed!")

        self._action_set_since_last_perform = False
        print(self._action)
        self._observation, self._curr_reward, self.done, info = self._env.step(self._action)

    def set_action(self, target_idx):
        """
        Makes a new action and sets the given index to the one action that is performed. If it it the camera, then the
        appropriate axis will be moved by 18 degrees
        :param target_idx: Target index to set to true
        :return: The action created (and sets the action)
        """
        action = self._env.action_space.noop()
        curr_idx = 0

        for k, v in action.items():
            if isinstance(v, int):
                if curr_idx == target_idx:
                    action[k] = 1
                    break
                curr_idx += 1
            else:
                # Case for the camera, move it 5% of the space, or 18˚
                if curr_idx == target_idx:
                    action[k][0] = 18
                    break
                elif curr_idx + 1 == target_idx:
                    action[k][1] = 18
                    break
                curr_idx += 2

        self._action = action
        self._action_set_since_last_perform = True
        return action

    def get_last_reward(self):
        """
        Gets a reward granted after the last update of state, error if no action taken before this call
        :return: A reward granted after the last update of state
        """
        if self._curr_reward is None:
            raise ValueError("Action must be taken before reward is accessed!")
        return self._curr_reward

