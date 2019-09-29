# https://github.com/flyyufelix/VizDoom-Keras-RL/blob/master/ddqn.py

import random
from collections import deque

import numpy as np
import tensorflow as tf
from keras import backend as K

from src.ddqn.dqn import dqn


class DoubleDQN:
    """
    DDQN Agent, as described here https://arxiv.org/pdf/1509.06461.pdf
    """

    def __init__(self, state_size, action_size, flags):
        """
        Creates a DDQN with the given state and action size
        :param state_size: The size of a given state
        :param action_size: The size of the action space
        :param flags: TensorFlow flags object with all of the needed hyperparameters
        """

        self.state_size = state_size
        self.action_size = action_size

        # Hyperparameters
        self.gamma = flags.gamma
        self.learning_rate = flags.learning_rate
        self.epsilon = flags.epsilon
        self.initial_epsilon = flags.initial_epsilon
        self.final_epsilon = flags.final_epsilon
        self.batch_size = flags.batch_size
        self.explore = flags.explore
        self.update_target_freq = flags.update_target_freq
        self.timestep_per_train = flags.timestep_per_train

        # Create replay memory using deque
        self.memory = deque()
        self.max_memory = flags.max_memory

        # Define main model and target model
        self.model = dqn(self.state_size, self.action_size, self.learning_rate)
        self.target_model = dqn(self.state_size, self.action_size, self.learning_rate)

        # Performance Statistics
        self.stats_window_size = flags.stats_window_size
        self.mavg_score = []  # Moving Average of Survival Time
        self.var_score = []  # Variance of Survival Time
        self.mavg_ammo_left = []  # Moving Average of Ammo used
        self.mavg_kill_counts = []  # Moving Average of Kill Counts

    def update_target_model(self):
        """
        After some time interval update the target model to be same with model
        :return: None
        """
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state):
        """
        Get action from model using epsilon-greedy policy
        :param state: The current state
        :return: The action
        """
        if np.random.rand() <= self.epsilon:
            action_idx = random.randrange(self.action_size)
        else:
            q = self.model.predict(state)
            action_idx = np.argmax(q)
        return action_idx

    # Save trajectory sample <s,a,r,s'> to the replay memory
    def replay_memory(self, s_t, action_idx, r_t, s_t1, is_terminated, t):
        self.memory.append((s_t, action_idx, r_t, s_t1, is_terminated))
        if self.epsilon > self.final_epsilon:
            self.epsilon -= (self.initial_epsilon - self.final_epsilon) / self.explore

        if len(self.memory) > self.max_memory:
            self.memory.popleft()

        # Update the target model to be same with model
        if t % self.update_target_freq == 0:
            self.update_target_model()

    # Pick samples randomly from replay memory (with batch_size)
    def train_minibatch_replay(self):
        """
        Train on a single minibatch
        """
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        update_input = np.zeros(((batch_size,) + self.state_size))  # Shape 64, img_rows, img_cols, 4
        update_target = np.zeros(((batch_size,) + self.state_size))
        action, reward, done = [], [], []

        for i in range(batch_size):
            update_input[i, :, :, :] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            update_target[i, :, :, :] = mini_batch[i][3]
            done.append(mini_batch[i][4])

        target = self.model.predict(update_input)  # Shape 64, Num_Actions

        target_val = self.model.predict(update_target)
        target_val_ = self.target_model.predict(update_target)

        for i in range(self.batch_size):
            # like Q Learning, get maximum Q value at s'
            # But from target model
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                # the key point of Double DQN
                # selection of action is from model
                # update is from target model
                a = np.argmax(target_val[i])
                target[i][action[i]] = reward[i] + self.gamma * (target_val_[i][a])

        # make minibatch which includes target q value and predicted q value
        # and do the model fit!
        loss = self.model.train_on_batch(update_input, target)

        return np.max(target[-1]), loss

    # Pick samples randomly from replay memory (with batch_size)
    def train_replay(self):

        num_samples = min(self.batch_size * self.timestep_per_train, len(self.memory))
        replay_samples = random.sample(self.memory, num_samples)

        update_input = np.zeros(((num_samples,) + self.state_size))
        update_target = np.zeros(((num_samples,) + self.state_size))
        action, reward, done = [], [], []

        for i in range(num_samples):
            update_input[i, :, :, :] = replay_samples[i][0]
            action.append(replay_samples[i][1])
            reward.append(replay_samples[i][2])
            update_target[i, :, :, :] = replay_samples[i][3]
            done.append(replay_samples[i][4])

        target = self.model.predict(update_input)
        target_val = self.model.predict(update_target)
        target_val_ = self.target_model.predict(update_target)

        for i in range(num_samples):
            # like Q Learning, get maximum Q value at s'
            # But from target model
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                # the key point of Double DQN
                # selection of action is from model
                # update is from target model
                a = np.argmax(target_val[i])
                target[i][action[i]] = reward[i] + self.gamma * (target_val_[i][a])

        loss = self.model.fit(update_input, target, batch_size=self.batch_size, nb_epoch=1, verbose=0)

        return np.max(target[-1]), loss.history['loss']

    # load the saved model
    def load_model(self, name):
        self.model.load_weights(name)

    # save the model which is under training
    def save_model(self, name):
        self.model.save_weights(name)
