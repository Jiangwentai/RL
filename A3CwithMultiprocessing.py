from keras import *
import keras.layers
import tensorflow as tf
import numpy as np
import random
import datetime
from collections import deque
import threading
from tensorflow.keras.optimizers import *
import gym
import multiprocessing
from multiprocessing.managers import BaseManager
import threading
from multiprocessing import Process, Queue
import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)
import os
import copy
#
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import threading
import time
from threading import RLock
import multiprocessing
from multiprocessing.managers import BaseManager


class acagent:
    def __init__(self):

        self.state_size = 4
        self.action_size = 2
        self.memory = deque(maxlen=5000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.00001
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0001

        self.vopt = tf.optimizers.Adam(learning_rate=self.learning_rate)
        self.piopt = tf.optimizers.Adam(learning_rate=self.learning_rate)

        self.vmodel = keras.Sequential(
            [layers.Dense(256, input_shape=(4,), activation='relu'),
             layers.Dense(512, 'relu' ),
             layers.Dense(512, 'relu'),

             layers.Dense(1, activation="linear")]
        )
        # self.vmodel.compile(loss='mse',
        #                     optimizer=Adam(lr=self.learning_rate))

        self.pimodel = keras.Sequential(
            [layers.Dense(256, input_shape=(4,), activation='relu'),
             layers.Dense(512, 'relu'),
             layers.Dense(512, 'relu'),

             layers.Dense(2, activation="softmax")]
        )
        self.piRfinal = 0
        self.vRfinal = 0

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, states):
        states = states.reshape(1, 4)

        action_logits = self.pimodel.predict(states)

        actions_prob = action_logits

        action = np.random.choice(len(actions_prob[0]), p=np.array(actions_prob)[0])

        return action

def run1(q, lock, finallist):
    env = gym.make('CartPole-v1')
    agent=acagent()
    Tmax = 10
    T = 0
    t = 1
    for T in range(Tmax):
        d_theta = tf.convert_to_tensor(0.0)
        d_thetav = tf.convert_to_tensor(0.0)
        thetav, theta = q[-1]
        agent.pimodel.set_weights(theta)
        agent.vmodel.set_weights(thetav)
        state = env.reset()
        state = np.reshape(state, [1, 4])
        statelist = []
        logproblist = []
        rewardlist = []
        actionlist = []
        donelist = []
        nextstatelist = []
        for t in range(200000):
            at = agent.choose_action(state)
            next_state, reward, done, _ = env.step(at)
            next_state = np.reshape(next_state, [1, 4])
            statelist.append(state)
            actionlist.append(at)
            nextstatelist.append(next_state)
            rewardlist.append(reward)
            donelist.append(done)
            state = next_state
            if done:
                print(np.sum(rewardlist))
                finallist.append(np.sum(rewardlist))
                break

        R =0.0
        tlist = list(range(len(statelist) - 1))
        tlist.reverse()

        for i in tlist:
            R = rewardlist[i]+agent.gamma*R
            with tf.GradientTape(persistent=True) as tapepi:
                y_predpi = tf.math.log(agent.pimodel(statelist[i])[0][actionlist[i]])*(agent.vmodel(statelist[i])[0]-R)

            d_theta=np.add(d_theta,tapepi.gradient(y_predpi,agent.pimodel.trainable_variables))


            with tf.GradientTape(persistent=True) as tapev:
                y_predv =  tf.square(tf.math.subtract(R, agent.vmodel(statelist[i])[0]))

            d_thetav=np.add(d_thetav,tapev.gradient( y_predv,agent.vmodel.trainable_variables))

        agent.piopt.apply_gradients(zip(d_theta, agent.pimodel.trainable_variables))
        agent.vopt.apply_gradients(zip(d_thetav, agent.vmodel.trainable_variables))
        v = agent.vmodel.get_weights()
        pi = agent.pimodel.get_weights()
        lock.acquire()
        q[-1]=(v, pi)
        lock.release()
        if T != 0 and T % 20 == 0:

            break

    #             d_thetav
    #
    # #
    # globalV, globalPI = q[-1]
    # d_theta=0
    # d_thetav=0
    #(state))[0][action]
    # agent = acagent()
    # import matplotlib.pyplot as plt
    # plt.ion()
    #
    # env = gym.make('CartPole-v1')
    # agent.pimodel.set_weights(globalPI)
    # agent.vmodel.set_weights(globalV)
    #
    # episodes = 20000000
    # finalresult=[]
    #
    #
    # for e in range(episodes):
    #     statelist = []
    #     logproblist=[]
    #     rewardlist = []
    #     actionlist = []
    #     donelist = []
    #     nextstatelist = []
    #
    #     # reset state in the beginning of each game
    #
    #     state = env.reset()
    #     state = np.reshape(state, [1, 4])
    #
    #
    #     # time_t represents each frame of the game
    #     # Our goal is to keep the pole upright as long as possible until score of 500
    #     # the more time_t the more score
    #     for time_t in range(2000000):
    #         # turn this on if you want to render
    #         # env.render()
    #
    #         # Decide action
    #         action = agent.choose_action(states=state)
    #         next_state, reward, done, _ = env.step(action)
    #         next_state = np.reshape(next_state, [1, 4])
    #         # statelist.append(state)
    #         # actionlist.append(action)
    #         # nextstatelist.append(next_state)
    #         # rewardlist.append(reward)
    #         # donelist.append(done)
    #         # agent.memorize(state, action, reward, next_state, done)
    #         rewardlist.append(reward)
    #
    #
    #         # reward = (reward - np.mean(reward)) / (np.std(reward) + 1e-9)
    #
    #
    #         # TDerror=tf.math.subtract(tf.math.add(reward[i],qtplus1),qt)
    #         with tf.GradientTape() as tape:
    #             qt = agent.vmodel(state)[0]
    #             qtplus1 = agent.vmodel(next_state)[0]
    #             y_pre = tf.square(tf.math.subtract(tf.math.add(reward, qtplus1), qt))
    #
    #
    #
    #
    #         grad = tape.gradient(y_pre, agent.vmodel.trainable_variables)
    #         grad=np.multiply(grad,0.5)
    #
    #         tf.clip_by_global_norm(list(grad), clip_norm=1.0)
    #
    #         agent.vopt.apply_gradients(zip(grad, agent.vmodel.trainable_variables))
    #
    #
    #         advantage = np.subtract(np.add(reward, agent.vmodel(next_state)[0]), agent.vmodel(state)[0])
    #
    #         with tf.GradientTape() as tape:
    #             y_pre2 = tf.math.log(agent.pimodel(state))[0][action]
    #
    #         # advantage=np.subtract(np.add(reward[i],  self.vmodel(next_state[i])[0])),self.vmodel(state[i])[0])
    #
    #         dtheta = tape.gradient(y_pre2, agent.pimodel.trainable_variables)
    #
    #         grad2 = np.multiply(np.multiply(dtheta, advantage), -1)
    #         tf.clip_by_global_norm(list(grad2), clip_norm=1.0)
    #         agent.piopt.apply_gradients(zip(grad2, agent.pimodel.trainable_variables))
    #         v = agent.vmodel.get_weights()
    #         pi = agent.pimodel.get_weights()
    #
    #
    #         # make next_state the new current state for the next frame.
    #         state = next_state
    #         if done:
    #
    #             finallist.append(sum(rewardlist))
    #             print(sum(rewardlist))
    #             # plt.clf()
    #             # plt.plot(finalresult)
    #             # plt.pause(0.001)
    #             break
    #     if e!=0 and e%100==0:
    #         lock.acquire()
    #         q.append((v, pi))
    #         lock.release()
    #         if len(q) > 40:
    #             q = q[-30:-1]
    #
    # exit()


if __name__ == "__main__":

    finallist = multiprocessing.Manager().list()

    manager = multiprocessing.Manager()

    q = manager.list()
    agent = acagent()
    q.append((agent.vmodel.get_weights(), agent.pimodel.get_weights()))
    lock = manager.Lock()
    # p = Process(target=run1, args=(q, lock))
    # p.start()
    # time.sleep(100)
    # print(len(q))
    # time.sleep(1000)
    # print(len(q))
    #
    #
    # p.join()
    p = multiprocessing.Pool(20)
    for i in range(200000):
        p.apply_async(run1, args=(q, lock, finallist))

    import matplotlib.pyplot as plt
    while True:
        plt.clf()
        B = copy.deepcopy(finallist)
        A = list(range(len(B)))
        plt.scatter(A, B, s=2, c=B, cmap=plt.cm.get_cmap("gist_rainbow"))
        plt.pause(0.001)

    p.close()
    p.join()
