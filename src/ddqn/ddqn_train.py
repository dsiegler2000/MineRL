import tensorflow as tf
import keras.backend as K
import numpy as np

from src.ddqn.data_loader import preprocess_img
from src.ddqn.ddqn import DoubleDQN
from src.minecraft_game import MinecraftGame

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float("gamma", 0.99, "gamma")
tf.app.flags.DEFINE_float("learning_rate", 1e-4, "learning rate")
tf.app.flags.DEFINE_float("epsilon", 1.0, "epsilon random action rate (is decayed)")
tf.app.flags.DEFINE_float("initial_epsilon", 1.0, "initial epsilon")
tf.app.flags.DEFINE_float("final_epsilon", 1e-4, "final epsilon")
tf.app.flags.DEFINE_integer("batch_size", 16, "batch size")
tf.app.flags.DEFINE_integer("explore", 100, "how many tics at the beginning to explore")
tf.app.flags.DEFINE_integer("update_target_freq", 20, "update the model every x ticks")
tf.app.flags.DEFINE_integer("timestep_per_train", 1, "number of timesteps between training interval")
tf.app.flags.DEFINE_integer("max_memory", 50000, "number of previous transitions to remember")
tf.app.flags.DEFINE_integer("stats_window_size", 50, "window size for computing rolling statistics")
tf.app.flags.DEFINE_integer("img_channels", 4, "how many images to stack up (i.e. the last 4 frames stacked)")
tf.app.flags.DEFINE_string("id", "MineRLNavigateDense-v0", "id of the scenario to do")

# TODO take into account compass angle

if __name__ == "__main__":
    # Avoid Tensorflow eating up (v)GPU memory
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    game = MinecraftGame(FLAGS.id, log_level="debug")
    print("Game environment created")

    # TODO make this right
    # game_state = game.get_state()
    # misc = game.get_game_variables()  # [KILLCOUNT, AMMO, HEALTH]
    # prev_misc = misc

    action_size = game.get_action_space_size()

    state_size = game.get_frame_size() + (FLAGS.img_channels,)
    agent = DoubleDQN(state_size, action_size, FLAGS)

    x_t = game.get_frame()
    x_t = preprocess_img(x_t, size=game.get_frame_size())
    s_t = np.stack(([x_t] * 4), axis=2)  # 64x64x4 (repeating the first frame 4 times because we don't have history)
    s_t = np.expand_dims(s_t, axis=0)  # 1x64x64x4

    done = game.is_episode_finished()

    # Start training
    GAME = 0
    t = 0
    max_life = 0  # Maximum episode life (Proxy for agent performance)
    life = 0

    # Buffer to compute rolling statistics
    life_buffer, ammo_buffer, kills_buffer = [], [], []

    while not game.is_episode_finished():
        loss = 0
        Q_max = 0

        # Epsilon Greedy
        action_idx = agent.get_action(s_t)  # We only take 1 action per frame

        game.set_action(action_idx)
        game.perform_action()

        game_state = game.get_state()  # Observe again after we take the action
        done = game.is_episode_finished()

        r_t = game.get_last_reward()  # each frame we get reward of 0.1, so 4 frames will be 0.4

        if done:
            # TODO change with reward
            if life > max_life:
                max_life = life
            GAME += 1
            print("Episode Finish ")
            game.new_episode()
            game_state = game.get_state()
            x_t1 = game.get_frame()

        x_t1 = game.get_frame()

        x_t1 = preprocess_img(x_t1, size=game.get_frame_size())
        x_t1 = np.reshape(x_t1, (1, game.get_frame_size()[0], game.get_frame_size()[1], 1))
        s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)

        # r_t = DoubleDQN.shape_reward(r_t, misc, prev_misc, t) I think this just rewarded certain actions

        if done:
            life = 0
        else:
            life += 1

        # Save the sample <s, a, r, s'> to the replay memory and decrease epsilon
        agent.replay_memory(s_t, action_idx, r_t, s_t1, done, t)

        # Do the training
        if t % agent.timestep_per_train == 0:
            Q_max, loss = agent.train_replay()

        s_t = s_t1
        t += 1

        # Save progress every 10000 iterations
        if t % 10000 == 0:
            print("Saving the model")
            agent.model.save_weights("models/ddqn.h5", overwrite=True)

        print(t)
        print(action_idx)
        if done:
            print(f"Time: {t}")
            print(f"Game: {GAME}")
            print(f"Epsilon: {agent.epsilon}")
            print(f"Final reward: {r_t}")
            print(f"Loss: {loss}")

            # TODO deal with this
            # Save Agent's Performance Statistics
            if GAME % agent.stats_window_size == 0:
                print("Update Rolling Statistics")
                agent.mavg_score.append(np.mean(np.array(life_buffer)))
                agent.var_score.append(np.var(np.array(life_buffer)))
                agent.mavg_ammo_left.append(np.mean(np.array(ammo_buffer)))
                agent.mavg_kill_counts.append(np.mean(np.array(kills_buffer)))

                # Reset rolling stats buffer
                life_buffer, ammo_buffer, kills_buffer = [], [], []

                # Write Rolling Statistics to file
                with open("statistics/ddqn_stats.txt", "w") as stats_file:
                    stats_file.write('Game: ' + str(GAME) + '\n')
                    stats_file.write('Max Score: ' + str(max_life) + '\n')
                    stats_file.write('mavg_score: ' + str(agent.mavg_score) + '\n')
                    stats_file.write('var_score: ' + str(agent.var_score) + '\n')
                    stats_file.write('mavg_ammo_left: ' + str(agent.mavg_ammo_left) + '\n')
                    stats_file.write('mavg_kill_counts: ' + str(agent.mavg_kill_counts) + '\n')
