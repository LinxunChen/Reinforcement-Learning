import random
import numpy as np
import tensorflow as tf

from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.regularizers import l2
import keras.backend as K
from tensorflow.python.keras.saving import model_from_json


class PPO:
    def __init__(self, n_actions, n_features, actor_lr=0.01, critic_lr=0.01, reward_decay=0.9, l2=0.01,
                 loss_clipping=0.1, target_update_alpha=0.9):
        self.n_actions = n_actions
        self.n_features = n_features
        self.actor_lr = actor_lr  # 学习率
        self.critic_lr = critic_lr
        self.gamma = reward_decay  # reward 递减率
        self.states, self.actions, self.rewards, self.states_, self.dones, self.v_by_trace = [], [], [], [], [], []  # V(s)=r+g*V(s_)
        self.l2 = l2
        self.loss_clipping = loss_clipping
        self.target_update_alpha = target_update_alpha #模型参数平滑因子
        self._build_critic()
        self._build_actor()
        self.dummy_advantage = np.zeros((1, 1))
        self.dummy_old_prediction = np.zeros((1, self.n_actions))

    def _build_critic(self):
        inputs = Input(shape=(self.n_features,))
        x = Dense(16, activation='relu', kernel_regularizer=l2(self.l2))(inputs)
        x = Dense(16, activation='relu', kernel_regularizer=l2(self.l2))(x)
        output = Dense(1, kernel_regularizer=l2(self.l2))(x)
        self.critic = Model(inputs=inputs, outputs=output)
        self.critic.compile(optimizer=Adam(learning_rate=self.critic_lr), loss='mean_squared_error',
                            metrics=['accuracy'])

    def _build_actor(self):
        state = Input(shape=(self.n_features,), name="state")
        advantage = Input(shape=(1,), name="Advantage")
        old_prediction = Input(shape=(self.n_actions,), name="Old_Prediction")
        x = Dense(16, activation='relu', kernel_regularizer=l2(self.l2))(state)
        x = Dense(16, activation='relu', kernel_regularizer=l2(self.l2))(x)
        policy = Dense(self.n_actions, activation='softmax', kernel_regularizer=l2(self.l2))(x)
        self.actor = Model(inputs=[state, advantage, old_prediction], outputs=policy)
        self.actor.compile(optimizer=Adam(learning_rate=self.actor_lr), loss=self.proximal_policy_optimization_loss(
            advantage=advantage, old_prediction=old_prediction),
                           metrics=['accuracy'])
        self.actor_old = self.build_network_from_copy(self.actor)

    def build_network_from_copy(self, actor_network):
        network_structure = actor_network.to_json()
        network_weights = actor_network.get_weights()
        network = model_from_json(network_structure)
        network.set_weights(network_weights)
        network.compile(optimizer=Adam(lr=self.actor_lr), loss="mse")  # loss随便写，只用于预测不更新参数
        return network

    def proximal_policy_optimization_loss(self, advantage, old_prediction):
        def loss(y_true, y_pred):
            prob = y_true * y_pred
            old_prob = y_true * old_prediction
            r = prob / (old_prob + 1e-10)
            return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - self.loss_clipping,
                                                           max_value=1 + self.loss_clipping) * advantage))
            # + entropy_loss * (prob * K.log(prob + 1e-10)))

        return loss

    def choose_action(self, observation, is_train_mode=True):
        observation = np.array(observation)
        observation = observation[np.newaxis, :]
        action_probs = self.actor.predict([observation, self.dummy_advantage, self.dummy_old_prediction])
        print('action_probs', action_probs)
        if is_train_mode:
            action = np.random.choice(range(action_probs.shape[1]), p=np.squeeze(action_probs))  # 加入了随机性
        else:
            action = np.squeeze(np.argmax(action_probs, axis=1))
        print('action', action)
        return action

    def store_transition(self, s, a, r, s_, d):
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)
        self.states_.append(s_)
        self.dones.append(d)

    def learn(self):
        # print('learn: sample length-',len(self.actions))
        # print('learn: states-',self.states)
        # print('learn: actions-',self.actions)
        self.cal_v_by_traceback()
        b_s, b_a, b_vt = np.array(self.states), np.array(self.actions), np.array(self.v_by_trace)
        b_v = self.get_v(b_s)
        b_adv = b_vt - b_v
        b_old_prediction = self.get_old_prediction(b_s)
        b_a_onehot = np.zeros(b_a.shape[0], self.n_actions)
        b_a_onehot[:, b_a.flatten()] = 1

        history = self.actor.fit(x=[b_s, b_adv, b_old_prediction], y=b_a_onehot, verbose=0)
        actor_loss_mean = np.mean(history.history['loss'])

        self.critic.fit(x=b_s, y=b_vt, epochs=2, verbose=0) #？todo 参考：https://github.com/liziniu/RL-PPO-Keras/blob/9fcfa63843c7d3a4d0b67c31de3405ccf5ce4212/ppo.py#L65

        self.states, self.actions, self.rewards, self.states_, self.dones, self.v_by_trace = [], [], [], [], [], []
        self.update_target_network()
        return actor_loss_mean

    def update_target_network(self):
        self.actor_old.set_weights(self.target_update_alpha * np.array(self.actor.get_weights())
                                           + (1 - self.target_update_alpha) * np.array(self.actor_old.get_weights()))

    def get_old_prediction(self, s):
        s = np.reshape(s, (-1, self.n_features))
        v = np.squeeze(self.actor_old.predict(s))
        return v

    def get_v(self, s):
        s = np.reshape(s, (-1, self.n_features))
        v = np.squeeze(self.critic.predict(s))
        return v

    def cal_v_by_traceback(self):
        '''
        截断后或episode结束后，通过回溯计算V(s)=r+g*V(s_)
        :return:
        '''
        self.v_by_traceback = np.zeros_like(self.rewards)
        if self.dones[-1]:
            v = 0
        else:
            s = np.array(self.states_[-1])
            v = self.get_v(s)

        for t in reversed(range(0, len(self.rewards))):
            v = v * self.gamma + self.rewards[t]
            self.v_by_traceback[t] = v

        # self.rewards = []

        # plt.figure()
        # plt.plot(discount_rewards)
        # plt.xlabel('episode steps')
        # plt.ylabel('normalized reward')
        # plt.show()
