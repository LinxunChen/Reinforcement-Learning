import random
import numpy as np

from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Dense, Lambda, Subtract, Add
from tensorflow.python.keras.optimizer_v2.adam import Adam
from collections import deque
import keras.backend as K
from tensorflow.python.keras.regularizers import l2


class DQN:
    def _build_net(self):
        inputs = Input(shape=(self.n_features,))
        x = Dense(64, activation='relu', kernel_regularizer=l2(self.l2))(inputs)
        x = Dense(32, activation='relu', kernel_regularizer=l2(self.l2))(x)
        if not self.dueling:
            output = Dense(self.n_actions, kernel_regularizer=l2(self.l2))(x)
        else:
            v = Dense(1, kernel_regularizer=l2(self.l2))(x)
            a = Dense(self.n_actions, kernel_regularizer=l2(self.l2))(x)
            mean = Lambda(lambda x: K.mean(x, axis=1, keepdims=True))(a)
            # advantage = Lambda(lambda x, y: x - y)([a, mean])
            # output = Lambda(lambda x, y: x + y)([v, advantage])
            advantage = Subtract()([a, mean])
            output = Add()([v, advantage])

        model = Model(inputs=inputs, outputs=output)
        model.compile(optimizer=Adam(learning_rate=self.lr), loss='mean_squared_error', metrics=['accuracy'])
        return model

    def __init__(self, n_actions, n_features, learning_rate=0.001, reward_decay=0.9, epsilon=1.0, epsilon_min=0.01,
                 epsilon_decay=0.998, replace_target_iter=300, memory_size=2000, batch_size=64, l2=0.001, double_q=True,
                 dueling=True):
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
        self.l2 = l2
        self.double_q = double_q
        self.dueling = dueling
        # self.memory = np.zeros((self.memory_size, n_features * 2 + 2))
        self.model_eval = self._build_net()
        self.model_target = self._build_net()
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
        if not self.double_q:
            q = self.model_target.predict(next_states)
            for i, (_, action, reward, _, done) in enumerate(data):
                target = reward
                if not done:
                    target += self.gamma * np.max(q[i])
                y[i][action] = target
        else:
            tmp = self.model_eval.predict(next_states)
            act_next = np.argmax(tmp, axis=1)
            q = self.model_target.predict(next_states)
            for i, (_, action, reward, _, done) in enumerate(data):
                target = reward
                if not done:
                    target += self.gamma * q[i, act_next[i]]
                y[i][action] = target

        history = self.model_eval.fit(x=states, y=y, verbose=0, batch_size=64, epochs=5)

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
            action = int(np.squeeze(np.argmax(action_vals, axis=1)))
        else:
            action = int(np.random.randint(0, self.n_actions))
        return action

    # def plot_cost(self):
    #     print('end,cost_his size:%d' % len(self.each_epoch_cost_his))
    #     plt.clf()
    #     plt.plot(np.arange(len(self.each_epoch_cost_his)), self.each_epoch_cost_his)
    #     plt.ylabel('Cost')
    #     plt.xlabel('training steps')  # step代表每个epoch（self.model_eval.fit的入参）
    #     plt.show()
