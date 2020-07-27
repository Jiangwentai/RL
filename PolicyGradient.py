from keras import *
import keras.layers
import  tensorflow as tf
import numpy as np

from tensorflow.keras.optimizers import SGD
import gym
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

ACTION_SPACE_SIZE=2
OBSERVATION_SPACE_SIZE=4
GAMMA=0.95


class Network(keras.Model):
  def __init__(self):
    super().__init__()

    self.model = keras.Sequential(
        [layers.Dense(32, 'relu',kernel_initializer=tf.random_normal_initializer()),
         layers.Dense(64, 'relu',kernel_initializer=tf.random_normal_initializer()),
         layers.Dense(128, 'relu',kernel_initializer=tf.random_normal_initializer()),

         layers.Dense(ACTION_SPACE_SIZE,activation="softmax")]
    )

  def call(self, x):
    out = self.model(x)

    return out

class PGAgent:

    def __init__(self):
        self.learningrate=0.0001
        self.network = Network()
        self.network.build(input_shape=(None, OBSERVATION_SPACE_SIZE))
        self.opt=keras.optimizers.Adam(learning_rate=0.00001)


    def choose_action(self, states):
        states = states.reshape(1, OBSERVATION_SPACE_SIZE)

        action_logits = self.network.predict(states)
        actions_prob = tf.nn.softmax(action_logits)
        action = np.random.choice(len(actions_prob.numpy()[0]), p=actions_prob.numpy()[0])
        return action
    #
    # def total_loss(self, model,statelist, actionlist, rewardlist):
    #     y_pred = self.network(states[i])
    #     for i in range(len(statelist) - 1):
    #         rewards = np.array(rewardlist[i])
    #         states = np.array(statelist[i])
    #         actions = np.array(actionlist[i])
    #         loglist = []
    #         for i, j in enumerate(rewards):
    #             loglist.append(tf.math.log(self.network(states[i])[0, actions[i]]))
    #         logtprob.append(tf.math.reduce_sum(tf.math.multiply(tf.reduce_sum(rewards), loglist)))

    def train(self, statelist, actionlist, rewardlist):

        rewards=self.discount_rewards(rewardlist)
        rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-9)
        for i,j in enumerate(statelist):

            with tf.GradientTape() as tape:
                y_pred=tf.math.log(self.network(statelist[i])[0][actionlist[i]])

            grad=tape.gradient(y_pred,self.network.trainable_variables)
            g=np.multiply(grad,-rewards[i])
            self.opt.apply_gradients(zip(g,self.network.trainable_variables))






        #
        #
        # Rgradient=0
        # optimizer = keras.optimizers.Adam(0.1)
        # gradslist=[]
        # logtprob = []
        # opt=SGD(learning_rate=self.learningrate)
        #
        #
        # for l in range(len(statelist)-1):
        #     rewards=np.array(rewardlist[l])
        #     rewards=self.discount_rewards(rewards)
        #     states=np.array(statelist[l])
        #     actions=np.array(actionlist[l])
        #     for i ,j in enumerate(states):
        #
        #         with tf.GradientTape() as tape:
        #             y_pred=tf.math.log(self.network(states[i])[0][actions[i]])
        #         grad=tape.gradient(y_pred,self.network.trainable_variables)
        #         grad=np.multiply(grad,rewards[i])
        #         if i ==0:
        #             sumlogprob=grad
        #         else:
        #
        #             sumlogprob=np.add(sumlogprob,grad)
        #     sumlogprob=np.subtract(sumlogprob,25)
        #     sumlogprob=np.multiply(-1,sumlogprob)
        #     if l==0:
        #         Rfinal=sumlogprob
        #     else:
        #         Rfinal=np.add(Rfinal,sumlogprob)
        # weight1=self.network.get_weights()
        #
        # weight2=np.add(np.multiply(self.learningrate,Rfinal),weight1)
        # self.network.set_weights(weight2)
        #
        # opt.apply_gradients(zip(Rfinal,self.network.trainable_variables))

    def discount_rewards(self, rewards):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * GAMMA + rewards[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        # discounted_ep_rs -= np.mean(discounted_ep_rs)
        # discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs



# initialize gym environment and the agent
env = gym.make('CartPole-v1')
agent = PGAgent()
episodes=20000000



# Iterate the game
i=0
import matplotlib.pyplot as plt

plt.ion()
accreward = []
finallist=[]

for e in range(episodes):

    # reset state in the beginning of each game
    state = env.reset()
    state = np.reshape(state, [1, 4])
    statelist = []
    rewardlist = []
    actionlist = []
    donelist = []
    # statelist.append([])
    # actionlist.append([])
    # rewardlist.append([])
    # donelist.append([])
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
        rewardlist.append(reward)
        donelist.append(done)
        # make next_state the new current state for the next frame.
        state = next_state
        if done ==True:
            agent.train(statelist,actionlist,rewardlist)
            accreward.append(np.sum(rewardlist))
            print(np.average(accreward))
            break
    plt.clf()
    plt.plot(accreward)
    plt.pause(0.001)







