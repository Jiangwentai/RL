import multiprocessing
import time
from multiprocessing import Pool
import numpy as np
from gym.wrappers.breakout_wrap import make_atari, wrap_deepmind
from tensorflow.keras import layers
import tensorflow as tf
import tensorflow.keras as keras
import datetime
# tf.config.experimental.set_memory_growth(tf.config.list_physical_devices()[1], True)


class acagent():

    def __init__(self):

        self.gamma = 0.99
        self.learning_rate = 0.001
        inputs = layers.Input(shape=(84, 84, 4,))

        layer1 = layers.Conv2D(16, [8, 8], strides=4, activation=tf.nn.relu,
                               kernel_initializer=keras.initializers.RandomNormal)(inputs)
        layer2 = layers.Conv2D(32, [4, 4], strides=2, activation=tf.nn.relu,
                               kernel_initializer=keras.initializers.RandomNormal)(layer1)
        layer3 = layers.Flatten()(layer2)

        layer4 = layers.Dense(256, activation=tf.nn.relu,
                              kernel_initializer=keras.initializers.RandomNormal)(layer3)

        critic = layers.Dense(1, activation="linear",
                              kernel_initializer=keras.initializers.RandomNormal)(layer4)

        action = layers.Dense(3, activation="softmax")(layer4)
        self.model = keras.Model(inputs=inputs, outputs=[action, critic])

        self.opt = keras.optimizers.RMSprop(self.learning_rate, 0.99, clipnorm=40)
        self.print_action_prob = 0
        self.TRAIN_MODEL = 1

    def choose_action(self, states):
        states = states.reshape(1, 84, 84, 4)
        action, critic = self.model(states)
        actions_prob = action
        if self.TRAIN_MODEL == 1:
            action = np.random.choice(len(actions_prob[0]), p=np.array(actions_prob)[0])
        else:
            action = np.argmax(actions_prob[0])
        self.print_action_prob = actions_prob
        return action + 1


"""
## Train
"""


def worker(e, weights_list, experience_list):
    e.clear()

    env = make_atari("PongNoFrameskip-v4")
    # Warp the frames, grey scale, stake four frame and scale to smaller ratio
    env = wrap_deepmind(env, frame_stack=True, scale=True, episode_life=True)
    agent = acagent()
    done = 0
    state = env.reset()
    state = np.reshape(state, [1, 84, 84, 4])

    while done == 0:

        agent.model.set_weights(weights_list[0])
        statelist = []
        rewardlist = []
        actionlist = []
        donelist = []
        nextstatelist = []
        tstepcount = 0
        for t in range(200000):
            at = agent.choose_action(state)

            next_state, reward, done, _ = env.step(at)
            next_state = np.reshape(next_state, [1, 84, 84, 4])
            statelist.append(state)
            actionlist.append(at)
            nextstatelist.append(next_state)

            rewardlist.append(reward)

            donelist.append(done)
            state = next_state
            tstepcount = tstepcount + 1

            if tstepcount >= 100:
                action, critic = agent.model(statelist[-1])
                R = critic
                break
            if done:
                R = 0.0
                break

        tlist = list(range(len(statelist)))
        tlist.reverse()
        experience = []
        for i in tlist:
            R = rewardlist[i] + agent.gamma * R
            experience.append((actionlist[i], statelist[i], donelist[i], R))

        experience_list.append(experience)
        #     #
        #     # with tf.GradientTape(persistent=True) as tape:
        #     #     actor, critic = agent.model(statelist[i])
        #     #
        #     #     y_predpi = tf.math.log(tf.maximum(actor[0][actionlist[i]-1], 1e-6)) * (-critic[0] + R)
        #     #     y_predv = tf.square(tf.math.subtract(R, critic[0]))
        #     #
        #     #     total_loss = -y_predpi + 0.5 * y_predv + entropymulti * tf.reduce_sum(
        #     #         tf.math.log(tf.maximum(actor[0], 1e-6)) * tf.maximum(actor[0], 1e-6))
        #     #
        #     # #
        #     # d_theta = np.add(d_theta, tape.gradient(total_loss, agent.model.trainable_variables))
        #
        # finallist.append(d_theta)
        e.wait()


def run2(finallist):
    env1 = make_atari("PongNoFrameskip-v4")
    env1 = wrap_deepmind(env1, frame_stack=True, scale=True, episode_life=True)
    globalagent1 = acagent()
    while True:
        if len(finallist) > 0:
            globalagent1.model.set_weights(finallist[0])
            state = env1.reset()
            state = np.reshape(state, [1, 84, 84, 4])
            for i in range(2000000):
                env1.render()
                at = globalagent1.choose_action(state)
                next_state, reward, done, _ = env1.step(at)
                next_state = np.reshape(next_state, [1, 84, 84, 4])
                state = next_state
                time.sleep(0.02)
                if done:
                    print(globalagent1.print_action_prob)
                    break


if __name__ == "__main__":

    process_number = 5

    p = Pool(process_number, maxtasksperchild=200)

    weights_list = multiprocessing.Manager().list()
    experience_list = multiprocessing.Manager().list()
    e = multiprocessing.Manager().Event()
    globalagent = acagent()

    weights_list.append(globalagent.model.get_weights())

    q1 = multiprocessing.Process(target=run2, args=(weights_list,))
    q1.start()

    initial_hour = datetime.datetime.now().hour

    for i in range(1000000):
        p.apply_async(worker, args=(e, weights_list, experience_list))

    episode_count = 0

    entropy_coef = 0.01

    while True:
        # start_time = time.time()
        if len(experience_list) == process_number:
            episode_count = episode_count + 1
            print(episode_count)

            total_lost_list = []
            with tf.GradientTape(persistent=True) as tape:

                for expericence in experience_list:

                    for action, state, done, R in expericence:

                        actor, critic = globalagent.model(state)

                        actor = tf.maximum(actor[0], 1e-6)

                        log_prob = tf.math.log(actor[action -1])

                        policy_cost = log_prob * (R - critic)

                        value_cost = tf.math.square(R - critic)

                        entropy_cost = -tf.reduce_sum(log_prob) * actor

                        total_lost = -policy_cost + 0.5 * value_cost - entropy_coef * entropy_cost

                        total_lost_list.append(total_lost)

                model_lost = tf.math.reduce_sum(total_lost_list)

            globalagent.opt.minimize(model_lost, globalagent.model.trainable_variables,tape=tape)

            weights_list[0] = globalagent.model.get_weights()

            experience_list[:] = []

            e.set()
