import numpy as np

from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.optimizer_v2.adam import Adam


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
        # self.each_epoch_cost_his = []
        self.each_step_cost_his = []

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
        q_next, q_eval = self.model_target.predict(batch_memory[:, -self.n_features:]), self.model_eval.predict(
            batch_memory[:, :self.n_features])
        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]
        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        history = self.model_eval.fit(x=batch_memory[:, 0:self.n_features], y=q_target, verbose=0, batch_size=32,
                                      epochs=10)
        # self.each_epoch_cost_his.extend(history.history['loss'])
        self.each_step_cost_his.append(np.mean(history.history['loss']))  # 记录每个step的平均loss

        self.learn_step_counter += 1

    def choose_action(self, observation):
        observation = np.array(observation)
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            action_vals = self.model_eval.predict(observation)
            action = np.squeeze(np.argmax(action_vals, axis=1))
        else:
            action = np.random.randint(0, self.n_actions)
        # print('action is ' + action)
        return action

    # def plot_cost(self):
    #     print('end,cost_his size:%d' % len(self.each_epoch_cost_his))
    #     plt.clf()
    #     plt.plot(np.arange(len(self.each_epoch_cost_his)), self.each_epoch_cost_his)
    #     plt.ylabel('Cost')
    #     plt.xlabel('training steps')  # step代表每个epoch（self.model_eval.fit的入参）
    #     plt.show()
