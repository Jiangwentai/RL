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

        action = layers.Dense(3, activation="softmax")(layer4)
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
        return action+1


"""
## Train
"""


def run1(e, lock,finallist):

    env = make_atari("PongNoFrameskip-v4")
    # Warp the frames, grey scale, stake four frame and scale to smaller ratio
    env = wrap_deepmind(env, frame_stack=True, scale=True, episode_life=True)

    # Warp the frames, grey scale, stake four frame and scale to smaller ratio

    agent = acagent()
    entropymulti = 0.01
    e.clear()
    done = 0
    state = env.reset()
    state = np.reshape(state, [1, 84, 84, 4])

    while done == 0:
        # start_time = time.time()
        e.clear()

        d_theta = tf.convert_to_tensor(0)
        # d_thetav = tf.convert_to_tensor(0.0)

        agent.model.set_weights(finallist[0])

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

            if tstepcount >= 30:
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

                y_predpi = tf.math.log(tf.maximum(actor[0][actionlist[i]-1], 1e-4)) * (-critic[0] + R)
                y_predv = tf.square(tf.math.subtract(R, critic[0]))

                total_loss = -y_predpi + 0.5 * y_predv + entropymulti * tf.reduce_sum(
                    tf.math.log(tf.maximum(actor[0], 1e-4)) * tf.maximum(actor[0], 1e-4))

            #
            d_theta = np.add(d_theta, tape.gradient(total_loss, agent.model.trainable_variables))

        lock.acquire()
        finallist.append(d_theta)
        lock.release()
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
                # print(globalagent1.model.get_weights())
                at = globalagent1.choose_action(state)
                next_state, reward, done, _ = env1.step(at)
                next_state = np.reshape(next_state, [1, 84, 84, 4])
                state = next_state
                # time.sleep(0.02)
                if done:
                    print(globalagent1.print_action_prob)
                    break


if __name__ == "__main__":

    process_number = 15

    p = Pool(process_number, maxtasksperchild=200)

    finallist = multiprocessing.Manager().list()


    e=multiprocessing.Manager().Event()


    globalagent = acagent()

    finallist.append(globalagent.model.get_weights())

    lock = multiprocessing.Manager().Lock()

    q1 = multiprocessing.Process(target=run2, args=(finallist,))
    q1.start()

    initial_hour = datetime.datetime.now().hour

    for i in range(1000000):
        p.apply_async(run1, args=(e, lock,finallist))

    plotlist = []
    record = []
    # logging.basicConfig(filename=__name__ + '.log',
    #                     format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', level=logging.NOTSET,
    #                     filemode='a+', datefmt='%Y-%m-%d%I:%M:%S %p')

    # if datetime.datetime.now().hour != initial_hour:
    #     globalagent.model.save_weights(str(datetime.datetime.now().day) + str(datetime.datetime.now().hour))
    #     initial_hour = datetime.datetime.now().hour
    episode_count=0

    while True:
        # start_time = time.time()
        if  len(finallist) >= process_number+1:

            print(episode_count)

            episode_count=episode_count+1
            sumgrad = 0

            for grad in finallist[1:]:
                sumgrad = np.add(sumgrad, grad)

            grads = sumgrad / (len(finallist) - 1)

            grads = [tf.clip_by_norm(grad, 40) for grad in grads]
            globalagent.opt.apply_gradients(zip(grads, globalagent.model.trainable_variables))
            finallist[:]=[]
            finallist.append(globalagent.model.get_weights())
            e.set()
