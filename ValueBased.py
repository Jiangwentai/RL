from keras import  *
from keras.layers import *
from keras.optimizers import Adam
from collections import deque
import numpy as np
import random
import gym
import gym.envs
import tensorflow as tf
learning_rate = 0.001
epsilon = 1  # exploration rate
epsilon_min = 0.01
epsilon_decay = 0.995
gamma=0.95

env = gym.make('CartPole-v0')
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
model = tf.keras.models.Sequential([

tf.keras.layers.Dense(24, input_shape=(4,), activation='relu'),
  tf.keras.layers.Dense(24, activation='relu'),
  tf.keras.layers.Dense(2, activation='linear')

])
model.compile(loss='mse',
              optimizer=Adam(lr=learning_rate))

opt=tf.optimizers.Adam(learning_rate=learning_rate)
for epoch in range(200000):

    state = env.reset()
    state = np.reshape(state, [1, 4])
    accreward=0
    at = np.random.randint(0, 2)
    for time in range(200000):
        qt=model.predict(state)[0][at]

        # with tf.GradientTape() as tape
        #     Q=

        if np.random.uniform() < epsilon:

            at = np.argmax(qt)
        else:
            at = np.random.randint(0, 2)

        qt=model.predict(state)[0][at]
        with tf.GradientTape() as tape:
            y_pred=model(state)[0][at]
        dt=tape.gradient(y_pred,model.trainable_variables)


        next_state, reward, done, _ = env.step(at)
        next_state=np.reshape(next_state,[1,4])

        yt=reward+np.multiply(gamma,np.amax(model.predict(next_state)[0]))
        grad=np.multiply(np.subtract(qt,yt),dt)
        opt.apply_gradients(zip(grad,model.trainable_variables))




        if done ==True:
            if epsilon > epsilon_min:
                epsilon *= epsilon_decay
            break


        state=next_state
        accreward=accreward+reward


    print(accreward)




