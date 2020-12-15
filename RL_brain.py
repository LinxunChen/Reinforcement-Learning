import random
import numpy as np
import tensorflow as tf

from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.losses import SparseCategoricalCrossentropy
from tensorflow.python.keras.optimizer_v2.adam import Adam
from collections import deque
import keras.backend as K


class DQN:
    def _build_net(self):
        eval_inputs = Input(shape=(self.n_features,))
        x = Dense(64, activation='relu')(eval_inputs)
        x = Dense(32, activation='relu')(x)
        eval_output = Dense(self.n_actions)(x)
        self.model_eval = Model(inputs=eval_inputs, outputs=eval_output)
        self.model_eval.compile(optimizer=Adam(learning_rate=self.lr), loss='mean_squared_error', metrics=['accuracy'])

        target_inputs = Input(shape=(self.n_features,))
        x = Dense(64, activation='relu')(target_inputs)
        x = Dense(32, activation='relu')(x)
        target_output = Dense(self.n_actions)(x)
        self.model_target = Model(inputs=target_inputs, outputs=target_output)
        self.model_target.compile(optimizer=Adam(learning_rate=self.lr), loss='mean_squared_error',
                                  metrics=['accuracy'])

    def __init__(self, n_actions, n_features, learning_rate=0.01, reward_decay=0.9, epsilon=1.0, epsilon_min=0.01,
                 epsilon_decay=0.995, replace_target_iter=300, memory_size=2000, batch_size=64):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.replace_target_iter = replace_target_iter
        # self.memory_size = memory_size
        self.batch_size = batch_size
        self.learn_step_counter = 0
        # self.memory = np.zeros((self.memory_size, n_features * 2 + 2))
        self._build_net()
        # 经验池
        self.memory_buffer = deque(maxlen=memory_size)

    def target_net_replace_op(self):
        w1 = self.model_eval.get_weights()
        self.model_target.set_weights(w1)
        # print("params has changed")

    def store_transition(self, state, action, reward, next_state, done):
        '''
        向经验池添加数据
        :param state:状态
        :param action:动作
        :param reward:回报
        :param next_state:下一个状态
        :param done:游戏结束标志
        :return:
        '''
        item = (state, action, reward, next_state, done)
        self.memory_buffer.append(item)

        # if not hasattr(self, 'memory_counter'):
        #     self.memory_counter = 0
        # transition = np.hstack((s, a, r, s_))
        # index = self.memory_counter % self.memory_size
        # self.memory[index, :] = transition
        # self.memory_counter += 1

    def learn(self):
        # 每N步更新target模型的参数
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net_replace_op()

        # 从经验池中随机采样一个batch
        data = random.sample(self.memory_buffer, self.batch_size)
        # 生成Q_target。
        states = np.array([d[0] for d in data])
        next_states = np.array([d[3] for d in data])

        y = self.model_eval.predict(states)
        q = self.model_target.predict(next_states)

        for i, (_, action, reward, _, done) in enumerate(data):
            target = reward
            if not done:
                target += self.gamma * np.max(q[i])
            y[i][action] = target
        history = self.model_eval.fit(x=states, y=y, verbose=0, batch_size=32, epochs=10)

        # # 从memory里选取一部分进行样本
        # batch_memory = self.memory[np.random.randint(0, min(self.memory_counter, self.memory_size), self.batch_size), :]
        #
        # # 构造q_target
        # q_next, q_eval = self.model_target.predict(batch_memory[:, -self.n_features:]), self.model_eval.predict(
        #     batch_memory[:, :self.n_features])
        # q_target = q_eval.copy()
        # batch_index = np.arange(self.batch_size, dtype=np.int32)
        # eval_act_index = batch_memory[:, self.n_features].astype(int)
        # reward = batch_memory[:, self.n_features + 1]
        # q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)
        #
        # history = self.model_eval.fit(x=batch_memory[:, 0:self.n_features], y=q_target, verbose=0, batch_size=32,
        #                               epochs=10)

        loss_mean = np.mean(history.history['loss'])  # 记录每个step的平均loss

        # update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.learn_step_counter += 1

        return loss_mean

    def choose_action(self, observation, is_train_mode=True):
        observation = np.array(observation)
        observation = observation[np.newaxis, :]

        if not is_train_mode or np.random.uniform() >= self.epsilon:
            action_vals = self.model_eval.predict(observation)
            action = np.squeeze(np.argmax(action_vals, axis=1))
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    # def plot_cost(self):
    #     print('end,cost_his size:%d' % len(self.each_epoch_cost_his))
    #     plt.clf()
    #     plt.plot(np.arange(len(self.each_epoch_cost_his)), self.each_epoch_cost_his)
    #     plt.ylabel('Cost')
    #     plt.xlabel('training steps')  # step代表每个epoch（self.model_eval.fit的入参）
    #     plt.show()


#
class PolicyGradient:
    def __init__(self, n_actions, n_features, learning_rate=0.01, reward_decay=0.9):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate  # 学习率
        self.gamma = reward_decay  # reward 递减率
        self.states, self.actions, self.rewards, self.ep_rewards, self.discount_rewards = [], [], [], [], []
        self._build_net()

    def _build_net(self):
        inputs = Input(shape=(self.n_features,))
        x = Dense(16, activation='relu')(inputs)
        x = Dense(16, activation='relu')(x)
        output = Dense(self.n_actions, activation='softmax')(x)
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
        print(self.states)
        print(self.actions)
        print(self.discount_rewards)
        history = self.model.fit(x=np.array(self.states), y=np.array(self.actions),
                                 sample_weight=np.array(self.discount_rewards),
                                 verbose=2, batch_size=32,
                                 epochs=10)
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
        return list(discount_rewards)
