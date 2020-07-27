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
        self.gamma = 0.95  # discount rate
        self.learning_rate = 0.0001

        self.vopt = tf.optimizers.Adam(learning_rate=self.learning_rate)
        self.piopt = tf.optimizers.Adam(learning_rate=self.learning_rate)

        self.vmodel = keras.Sequential(
            [layers.Dense(128, input_shape=(4,), activation='relu',kernel_initializer =tf.compat.v1.truncated_normal_initializer(stddev=0.02)),
             layers.Dense(256, 'relu',kernel_initializer =tf.compat.v1.truncated_normal_initializer(stddev=0.02)),
             layers.Dense(512, 'relu', kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02)),
             layers.Dense(1024, 'relu', kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02)),

             layers.Dense(1, activation="linear")]
        )
        self.vmodel.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))



        self.pimodel = keras.Sequential(
            [layers.Dense(128, input_shape=(4,), activation='relu',kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02)),
             layers.Dense(256, 'relu',kernel_initializer =tf.compat.v1.truncated_normal_initializer(stddev=0.02)),
             layers.Dense(512, 'relu', kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02)),
             layers.Dense(1024, 'relu', kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02)),


             layers.Dense(2, activation="softmax")]
        )

    # def memorize(self, state, action, reward, next_state, done):
    #     self.memory.append((state, action, reward, next_state, done))



    def choose_action(self, states):
        states = states.reshape(1, 4)


        action_logits = self.pimodel.predict(states)

        actions_prob = tf.nn.softmax(action_logits)


        action = np.random.choice(len(actions_prob.numpy()[0]), p=actions_prob.numpy()[0])


        return action


    def train(self, statelist, actionlist,nextstatelist, rewardlist):

        for i,j in enumerate(statelist):

            with tf.GradientTape() as tape:
                qt = self.vmodel(statelist[i])[0]
                qtplus1 = self.vmodel(nextstatelist[i])[0]
                y_pre = tf.square(tf.math.subtract(tf.math.add(rewardlist[i], qtplus1), qt))




            grad = tape.gradient(y_pre, self.vmodel.trainable_variables)
            grad=np.multiply(grad,0.5)

            tf.clip_by_global_norm(list(grad), clip_norm=1.0)

            self.vopt.apply_gradients(zip(grad, self.vmodel.trainable_variables))


            advantage = np.subtract(np.add(rewardlist[i], self.vmodel(nextstatelist[i])[0]), self.vmodel(statelist[i])[0])

            with tf.GradientTape() as tape:
                y_pre2 = tf.math.log(self.pimodel(statelist[i]))[0][actionlist[i]]

            # advantage=np.subtract(np.add(reward[i],  self.vmodel(next_state[i])[0])),self.vmodel(state[i])[0])

            dtheta = tape.gradient(y_pre2, self.pimodel.trainable_variables)

            grad2 = np.multiply(np.multiply(dtheta, advantage), -1)
            self.piopt.apply_gradients(zip(grad2, self.pimodel.trainable_variables))






def run1():


    agent = acagent()
    import matplotlib.pyplot as plt
    plt.ion()

    env = gym.make('CartPole-v1')

    episodes = 1000000
    finalresult=[]


    for e in range(episodes):
        statelist = []
        logproblist=[]
        rewardlist = []
        actionlist = []
        donelist = []
        nextstatelist = []

        # reset state in the beginning of each game

        state = env.reset()
        state = np.reshape(state, [1, 4])


        # time_t represents each frame of the game
        # Our goal is to keep the pole upright as long as possible until score of 500
        # the more time_t the more score
        for time_t in range(2000000):
            # turn this on if you want to render
            # env.render()

            # Decide action
            action = agent.choose_action(states=state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, 4])
            statelist.append(state)
            actionlist.append(action)
            nextstatelist.append(next_state)
            rewardlist.append(reward)
            donelist.append(done)
            # agent.memorize(state, action, reward, next_state, done)
            rewardlist.append(reward)




            # TDerror=tf.math.subtract(tf.math.add(reward[i],qtplus1),qt)



            # make next_state the new current state for the next frame.
            state = next_state
            if done:

                agent.train(statelist,actionlist,nextstatelist,rewardlist)


                print(sum(rewardlist))
                # plt.clf()
                # plt.plot(finalresult)
                # plt.pause(0.001)
                break


if __name__ == "__main__":
    run1()

    # finallist = multiprocessing.Manager().list()
    #
    # manager = multiprocessing.Manager()
    #
    # q = manager.list()
    # agent = acagent()
    # q.append((agent.vmodel.get_weights(), agent.pimodel.get_weights()))
    # lock = manager.Lock()
    # # p = Process(target=run1, args=(q, lock))
    # # p.start()
    # # time.sleep(100)
    # # print(len(q))
    # # time.sleep(1000)
    # # print(len(q))
    # #
    # #
    # # p.join()
    # p = multiprocessing.Pool(20)
    # for i in range(200000):
    #     p.apply_async(run1, args=(q, lock,finallist))
    #
    # import matplotlib.pyplot as plt
    # while True:
    #     c = []
    #
    #     for i, j in enumerate(finallist):
    #         if i != 0 and i % 50 == 0:
    #             a = np.average(finallist[i - 50:i])
    #             c.append(a)
    #
    #
    #     plt.clf()
    #     plt.plot(c)
    #     plt.pause(10)
    # p.close()
    # p.join()
