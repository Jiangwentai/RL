from keras import  *
from keras.layers import *
from keras.optimizers import Adam
from collections import deque
import numpy as np
import random
import gym
import gym.envs
import tensorflow as tf
import matplotlib.pyplot as plt
import copy

learning_rate = 0.00001
epsilon = 1  # exploration rate
epsilon_min = 0.01
epsilon_decay = 0.995
gamma=0.95


minibatch=deque(maxlen=200000)



env = gym.make('CartPole-v1')
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
model = tf.keras.models.Sequential([

tf.keras.layers.Dense(32, input_shape=(4,), activation='relu'),
  tf.keras.layers.Dense(64, activation='relu'),
    # tf.keras.layers.Dense(64, activation='relu'),


    # tf.keras.layers.Dense(256, activation='relu'),

    tf.keras.layers.Dense(2, activation='linear')

])

model.compile(loss='mse',
                    optimizer=Adam(lr=learning_rate))

targetmodel = tf.keras.models.Sequential([

tf.keras.layers.Dense(32, input_shape=(4,), activation='relu'),
  tf.keras.layers.Dense(64, activation='relu'),
    # tf.keras.layers.Dense(64, activation='relu'),



    # tf.keras.layers.Dense(256, activation='relu'),

    tf.keras.layers.Dense(2, activation='linear')

])
# targetmodel.set_weights(model.get_weights())
#
opt=tf.optimizers.Adam(learning_rate=learning_rate)
finallist=[]
c=0
for epoch in range(2000000):

    state = env.reset()
    state = np.reshape(state, [1, 4])
    accreward=0
    at = np.random.randint(0, 2)
    inited=False
    for time in range(200000):

        if np.random.uniform() < epsilon:
            at = np.random.randint(0, 2)

        else:
            qt = model(state)[0]
            at = np.argmax(qt)
        epsilon = epsilon * epsilon_decay



        next_state, reward, done, _ = env.step(at)
        next_state=np.reshape(next_state,[1,4])
        minibatch.append((state,next_state,at,reward*0.01,done))
        # with tf.GradientTape() as tape:
        #     qt = model(state)[0][at]
        #     yt = reward + np.multiply(gamma, np.amax(targetmodel(next_state)[0]))
        #     y_pred = tf.square(tf.math.subtract(qt,yt))
        # grad = tape.gradient(y_pred, model.trainable_variables)
        # grad = np.multiply(grad, 0.5)
        # opt.apply_gradients(zip(grad, model.trainable_variables))


        state=next_state
        accreward=accreward+reward
        c=c+1
        if done ==True:
            if len(minibatch) >129:
                for _state, _next_state, _at, _reward, _done in random.sample(minibatch, 32):

                    with tf.GradientTape() as tape:
                        qt = model(_state)[0][_at]
                        yt = tf.convert_to_tensor(_reward*0.01) + np.multiply(gamma, np.amax(targetmodel(_next_state)[0]))
                        y_pred = tf.square(tf.math.subtract(yt,qt))

                    grad = tape.gradient(y_pred, model.trainable_variables)
                    grad = np.multiply(grad, 0.5)
                    opt.apply_gradients(zip(grad, model.trainable_variables))
            if c!=0 and c%1000==0:
                targetmodel.set_weights(model.get_weights())
            break





    print(accreward)
    finallist.append(accreward)
    plt.clf()
    B = copy.deepcopy(finallist)
    A = list(range(len(B)))
    plt.scatter(A, B, s=2, c=B, cmap=plt.cm.get_cmap("rainbow"))
    plt.pause(0.001)



