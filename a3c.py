import multiprocessing
import time

from multiprocessing import Pool

import numpy as np
from gym.wrappers.breakout_wrap import make_atari, wrap_deepmind
from tensorflow.keras import layers
import tensorflow as tf
import tensorflow.keras as keras
import datetime


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

        action = layers.Dense(6, activation="softmax")(layer4)
        self.model = keras.Model(inputs=inputs, outputs=[action, critic])

        self.opt = keras.optimizers.RMSprop(self.learning_rate, 0.99)
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
        return action


"""
## Train
"""


def run1(q, lock, finallist):

    env = make_atari("PongNoFrameskip-v4")
    # Warp the frames, grey scale, stake four frame and scale to smaller ratio
    env = wrap_deepmind(env, frame_stack=True, scale=True, episode_life=True)

    # Warp the frames, grey scale, stake four frame and scale to smaller ratio

    agent = acagent()
    entropymulti = 0.01

    done = 0

    state = env.reset()
    state = np.reshape(state, [1, 84, 84, 4])

    while done == 0:
        # start_time = time.time()

        d_theta = tf.convert_to_tensor(0)
        # d_thetav = tf.convert_to_tensor(0.0)

        agent.model.set_weights(finallist[0])
        agent.opt._create_all_weights( agent.model.trainable_variables)
        agent.opt.set_weights(finallist[1])

        statelist = []
        rewardlist = []
        actionlist = []
        donelist = []
        nextstatelist = []
        tstepcount = 0
        for t in range(200000):
            at = agent.choose_action(state)
            for h in range(4):

                next_state, reward, done, _ = env.step(at)
                next_state = np.reshape(next_state, [1, 84, 84, 4])
                statelist.append(state)
                actionlist.append(at)
                nextstatelist.append(next_state)

                rewardlist.append(reward)

                donelist.append(done)
                state = next_state
                tstepcount = tstepcount + 1
                if done:
                    break

            if tstepcount >= 5:
                action, critic = agent.model(statelist[-1])
                R = critic
                break
            if done:
                # print(np.sum(rewardlist))
                # print(np.average(np.sum(rewardlist)))
                R = 0.0

                break

        # tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(agent.PImodel(statelist[i])[0], agent.PImodel(statelist[i])[0])
        tlist = list(range(len(statelist)))
        tlist.reverse()

        for i in tlist:
            R = rewardlist[i] + agent.gamma * R

            with tf.GradientTape(persistent=True) as tape:
                actor, critic = agent.model(statelist[i])

                y_predpi = tf.math.log(tf.maximum(actor[0][actionlist[i]], 1e-6)) * (-critic[0] + R)
                y_predv = tf.square(tf.math.subtract(R, critic[0]))

                total_loss = -y_predpi + 0.5 * y_predv + entropymulti * tf.reduce_sum(
                    tf.math.log(tf.maximum(actor[0], 1e-6)) * tf.maximum(actor[0], 1e-6))

            #
            d_theta = np.add(d_theta, tape.gradient(total_loss, agent.model.trainable_variables))

        d_theta=[ tf.clip_by_norm(h,40) for h in d_theta ]
        agent.opt.apply_gradients(zip(d_theta,agent.model.trainable_variables))

        theta=agent.model.get_weights()
        theta_opt=agent.opt.get_weights()
        lock.acquire()

        finallist[0]=theta
        finallist[1]=theta_opt

        lock.release()



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
                print(globalagent1.print_action_prob)
                next_state, reward, done, _ = env1.step(at)
                next_state = np.reshape(next_state, [1, 84, 84, 4])
                state = next_state
                # time.sleep(0.02)
                if done:
                    break


if __name__ == "__main__":

    process_number = 18

    p = Pool(process_number, maxtasksperchild=200)

    finallist = multiprocessing.Manager().list()



    q = []

    globalagent = acagent()
    globalagent.opt._create_all_weights(globalagent.model.trainable_variables)
    finallist.append(globalagent.model.get_weights())
    finallist.append(globalagent.opt.get_weights())

    lock = multiprocessing.Manager().Lock()


    initial_hour = datetime.datetime.now().hour

    for i in range(1000000):
        p.apply_async(run1, args=(q, lock, finallist))

    q1 = multiprocessing.Process(target=run2, args=(finallist,))
    q1.start()
    q1.join()