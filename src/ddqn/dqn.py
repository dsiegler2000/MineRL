# Refactored from https://github.com/flyyufelix/VizDoom-Keras-RL/blob/master/networks.py

from keras.layers import Convolution2D, Dense, Flatten
from keras.models import Sequential
from keras.optimizers import Adam


def dqn(input_shape, action_size, learning_rate):
    model = Sequential()

    # Conv layers
    model.add(Convolution2D(32, 8, 8, subsample=(4, 4), activation='relu', input_shape=input_shape))
    model.add(Convolution2D(64, 4, 4, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Flatten())

    # Dense output
    model.add(Dense(output_dim=512, activation='relu'))
    model.add(Dense(output_dim=action_size, activation='linear'))

    adam = Adam(lr=learning_rate)
    model.compile(loss='mse', optimizer=adam)

    return model
