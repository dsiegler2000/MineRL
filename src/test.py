import minerl
import gym
import logging

logging.basicConfig(level=logging.DEBUG)

env = gym.make("MineRLNavigateDense-v0")
obs = env.reset()

done = False

net_reward = 0

while not done:
    # Get the "nothing" action space
    action = env.action_space.noop()

    # Move to set the compass angle to 0 (aka going in the right direction)
    action["camera"] = [0, 0.03 * obs["compassAngle"]]
    action["back"] = 0
    action["forward"] = 1
    action["jump"] = 1
    action["attack"] = 1

    obs, reward, done, _ = env.step(action)
    net_reward += reward
