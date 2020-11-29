import numpy as np

from tensorflow.python import keras
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.optimizer_v2.adam import Adam
import matplotlib.pyplot as plt


class DQN:
    def _build_net(self):
        eval_inputs = Input(shape=(self.n_features,))
        x = Dense(64, activation='relu')(eval_inputs)
        x = Dense(64, activation='relu')(x)
        eval_output = Dense(self.n_actions)(x)
        self.model_eval = Model(inputs=eval_inputs, outputs=eval_output)
        self.model_eval.compile(optimizer=Adam(learning_rate=self.lr), loss='mean_squared_error', metrics=['accuracy'])

        target_inputs = Input(shape=(self.n_features,))
        x = Dense(64, activation='relu')(target_inputs)
        x = Dense(64, activation='relu')(x)
        target_output = Dense(self.n_actions)(x)
        self.model_target = Model(inputs=target_inputs, outputs=target_output)
        self.model_target.compile(optimizer=Adam(learning_rate=self.lr), loss='mean_squared_error',
                                  metrics=['accuracy'])

    def __init__(self, n_actions, n_features, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9,
                 replace_target_iter=300, memory_size=500, batch_size=32, e_greedy_increment=None, output_graph=False):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.learn_step_counter = 0
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))
        self._build_net()
        self.cost_his = []

    def target_net_replace_op(self):
        w1 = self.model_eval.get_weights()
        self.model_target.set_weights(w1)
        print("params has changed")

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, a, r, s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # 每N步更新target模型的参数
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net_replace_op()

        # 从memory里选取一部分进行样本
        batch_memory = self.memory[np.random.randint(0, min(self.memory_counter, self.memory_size), self.batch_size), :]

        # 构造q_target
        # q_next, q_eval 包含所有 action 的值，而我们需要的只是已经选择好的 action 的值, 其他的并不需要。
        # 所以我们将其他的 action 值全变成 0, 将用到的 action 误差值 反向传递回去, 作为更新凭据。
        # 将 q_eval 全部赋值给 q_target, 这时 q_target-q_eval 全为 0,
        # 不过 我们再根据 batch_memory 当中的 action 这个 column 来给 q_target 中的对应的 memory-action 位置来修改赋值.
        # 使新的赋值为 reward + gamma * maxQ(s_), 这样 q_target-q_eval 就可以变成我们所需的样子.
        q_next, q_eval = self.model_target.predict(batch_memory[:, -self.n_features:]), self.model_eval.predict(
            batch_memory[:, :self.n_features])
        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]
        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        history = self.model_eval.fit(x=batch_memory[:, 0:self.n_features], y=q_target, verbose=0, batch_size=64, epochs=10)
        self.cost_his.extend(history.history['loss'])

        self.learn_step_counter += 1

    def choose_action(self, observation):
        observation = np.array(observation)
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            action_vals = self.model_eval.predict(observation)
            action = np.squeeze(np.argmax(action_vals, axis=1))
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def plot_cost(self):
        print('end,cost_his size:%d' % len(self.cost_his))
        plt.clf()
        # plt.legend()
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()
