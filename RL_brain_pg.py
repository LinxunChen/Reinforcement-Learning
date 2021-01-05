import random
import numpy as np
import tensorflow as tf

from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.regularizers import l2


class PolicyGradient:
    def __init__(self, n_actions, n_features, learning_rate=0.01, reward_decay=0.9):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate  # 学习率
        self.gamma = reward_decay  # reward 递减率
        self.states, self.actions, self.rewards, self.ep_rewards, self.discount_rewards = [], [], [], [], []
        self.l2 = 0.01
        self._build_net()

    def _build_net(self):
        inputs = Input(shape=(self.n_features,))
        x = Dense(16, activation='relu', kernel_regularizer=l2(self.l2))(inputs)
        x = Dense(16, activation='relu', kernel_regularizer=l2(self.l2))(x)
        output = Dense(self.n_actions, activation='softmax', kernel_regularizer=l2(self.l2))(x)
        self.model = Model(inputs=inputs, outputs=output)
        self.model.compile(optimizer=Adam(learning_rate=self.lr), loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                           metrics=['accuracy'])

    def choose_action(self, observation, is_train_mode=True):
        observation = np.array(observation)
        observation = observation[np.newaxis, :]
        action_probs = self.model.predict(observation)
        print('action_probs', action_probs)
        if is_train_mode:
            action = np.random.choice(range(action_probs.shape[1]), p=np.squeeze(action_probs))  # 加入了随机性
        else:
            action = np.squeeze(np.argmax(action_probs, axis=1))
        print('action', action)
        return action

    def store_transition(self, s, a, r):
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)
        self.ep_rewards.append(r)

    def learn(self):
        # print('learn: sample length-',len(self.actions))
        # print('learn: states-',self.states)
        # print('learn: actions-',self.actions)
        # print('learn: discount_rewards-',self.discount_rewards)
        history = self.model.fit(x=np.array(self.states), y=np.array(self.actions),
                                 sample_weight=np.array(self.discount_rewards),
                                 verbose=2, batch_size=64,
                                 epochs=2)
        loss_mean = np.mean(history.history['loss'])
        self.states, self.actions, self.rewards, self.discount_rewards = [], [], [], []
        return loss_mean

    def discount_and_norm_ep_rewards(self):
        '''
        一次episode结束后计算discounted reward
        :return:
        '''
        discount_rewards = np.zeros_like(self.ep_rewards)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rewards))):
            running_add = running_add * self.gamma + self.ep_rewards[t]
            discount_rewards[t] = running_add

        # normalize episode rewards
        discount_rewards -= np.mean(discount_rewards)
        discount_rewards /= np.std(discount_rewards)

        self.ep_rewards = []

        # plt.figure()
        # plt.plot(discount_rewards)
        # plt.xlabel('episode steps')
        # plt.ylabel('normalized reward')
        # plt.show()
        return list(discount_rewards)
