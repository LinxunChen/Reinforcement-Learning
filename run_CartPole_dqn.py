import random

import gym
import os
import matplotlib.pyplot as plt
import numpy as np
from RL_brain_dqn import DQN
import tensorflow as tf
from tensorflow.python.keras import backend as K

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def plot_reward_and_cost(i_episode, history):
    if i_episode <= 1:
        plt.figure()
        plt.ion()
    # elif i_episode >= size - 1:
    #     plt.ioff()
    #     # plt.show()
    #     return
    plt.clf()
    plt.subplot(3, 1, 1)
    plt.plot(np.arange(len(history['Episode_reward'])), history['Episode_reward'])
    plt.title('each_episode_reward')
    plt.subplot(3, 1, 2)
    plt.plot(np.arange(len(history['Loss'])), history['Loss'])
    plt.title('each_episode_cost')
    plt.subplot(3, 1, 3)
    plt.plot(np.arange(len(history['Val_reward'])) * 5, history['Val_reward'])
    plt.title('each_val_reward')
    plt.pause(0.00000001)


def train(episodes):
    total_steps = 0  # 记录步数
    history = {'episode': [], 'Episode_reward': [], 'Loss': [], 'Val_reward': []}

    for i_episode in range(episodes):
        observation = env.reset()
        reward_sum = 0
        loss = np.infty
        action_cnt = {}
        while True:
            env.render()
            action = rl.choose_action(observation)
            if action not in action_cnt:
                action_cnt[action] = 1
            else:
                action_cnt[action] = action_cnt[action] + 1
            observation_, reward, done, info = env.step(action)

            # x 是车的水平位移, 所以 r1 是车越偏离中心, 分越少
            # theta 是棒子离垂直的角度, 角度越大, 越不垂直. 所以 r2 是棒越垂直, 分越高
            x, x_dot, theta, theta_dot = observation_
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            reward = r1 + r2  # 总 reward 是 r1 和 r2 的结合, 既考虑位置, 也考虑角度, 这样学习更有效率

            reward_sum += reward
            rl.store_transition(observation, action, reward, observation_, done)

            if total_steps > 1000:
                loss = rl.learn()

            if done:
                break

            observation = observation_
            total_steps += 1

        history['episode'].append(i_episode)
        history['Episode_reward'].append(reward_sum)
        history['Loss'].append(loss)
        print(
            'Episode: {}/{} | Action cnt: {} | Episode reward: {} | loss: {:.6f} | e:{:.2f}'.format(i_episode, episodes,
                                                                                                    action_cnt,
                                                                                                    reward_sum,
                                                                                                    loss, rl.epsilon))

        if i_episode % 5 == 0:
            validate(history)
            plot_reward_and_cost(i_episode, history)
    return history
    # ENV = ENV.unwrapped
    # RL.plot_cost()


def validate(history):
    print('-----------Validate Start-----------')
    observation = env.reset()
    i_episode = 0
    total_episodes = 6
    turns_count = 0
    reward_sum = 0
    action_cnt = {}

    while i_episode < total_episodes:
        # env.render()
        action = rl.choose_action(observation, False)
        if action not in action_cnt:
            action_cnt[action] = 1
        else:
            action_cnt[action] = action_cnt[action] + 1
        observation, reward, done, info = env.step(action)
        turns_count += 1
        reward_sum += reward
        if done:
            i_episode += 1
            observation = env.reset()
    # env.close()
    print('VAL—>action_cnt:{}, reward_sum:{}, turns_count:{}'.format(action_cnt, reward_sum, turns_count))
    history['Val_reward'].append(reward_sum)
    print('-----------Validate End-----------')


def play():
    """使用训练好的模型测试游戏.
    """
    observation = env.reset()
    i_episode = 0
    total_episodes = 10
    turns_count = 0
    reward_sum = 0

    while i_episode < total_episodes:
        env.render()
        action = rl.choose_action(observation, False)
        observation, reward, done, info = env.step(action)

        turns_count += 1
        reward_sum += reward

        if done:
            print("Reward for this episode was: {}, turns was: {}".format(reward_sum, turns_count))
            i_episode += 1
            reward_sum = 0
            turns_count = 0
            observation = env.reset()

    env.close()


def model_reproducible():
    # 模型可复现
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(1)
    random.seed(1)
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                                  inter_op_parallelism_threads=1)
    tf.set_random_seed(1)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)


if __name__ == '__main__':
    # 玩500回合，边玩边产生样本，边训练
    tf.disable_eager_execution()
    env = gym.make('CartPole-v0')
    env.seed(1)
    model_reproducible()

    print(env.action_space)  # 显示可用 action
    print(env.observation_space)  # 显示可用 state 的 observation
    print(env.observation_space.high)  # 显示 observation 最高值
    print(env.observation_space.low)  # 显示 observation 最低值

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    rl = DQN(action_size, state_size)

    EPISODES = 500
    his = train(EPISODES)
    play()
    plt.ioff()
    plt.show()
