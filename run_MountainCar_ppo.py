import random

import gym
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K

from RL_brain_ppo import PPO

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def plot_reward_and_cost(i_episode, size, history):
    if i_episode <= 1:
        plt.figure()
        plt.ion()
    # elif i_episode >= size - 1:
    #     plt.ioff()
    #     # plt.show()
    #     return
    plt.clf()
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(len(history['Episode_reward'])), history['Episode_reward'])
    plt.title('each_episode_reward')
    plt.subplot(2, 1, 2)
    plt.plot(np.arange(len(history['Loss'])), history['Loss'])  # 1个episode训练step次，每次训练遍历样本epoch次
    plt.title('each_episode_cost')
    plt.pause(0.00000001)


def train():
    history = {'episode': [], 'Episode_reward': [], 'Loss': []}

    for i_episode in range(EP_MAX):
        observation = env.reset()
        ep_reward_sum = 0
        loss = np.infty
        for t in range(EP_LEN):
            env.render()
            action = rl.choose_action(observation)
            observation_, reward, done, info = env.step(action)

            # 车开得越高 reward 越大
            position, velocity = observation_
            reward = abs(position - (-0.5))

            ep_reward_sum += reward
            rl.store_transition(observation, action, reward, observation_, done)

            if (t + 1) % BATCH == 0 or t == EP_LEN - 1:
                loss = rl.learn()
            observation = observation_

        if i_episode != 0 and i_episode % 1 == 0:
            history['episode'].append(i_episode)
            history['Episode_reward'].append(ep_reward_sum)
            history['Loss'].append(loss)
            print(
                'Episode: {}/{} | Episode reward: {} | loss: {:.6f}'.format(i_episode, EP_MAX, ep_reward_sum, loss))
            plot_reward_and_cost(i_episode, EP_MAX, history)
    return history


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
    env = gym.make('MountainCar-v0')
    env.seed(1)
    model_reproducible()

    print(env.action_space)  # 显示可用 action
    print(env.observation_space)  # 显示可用 state 的 observation
    print(env.observation_space.high)  # 显示 observation 最高值
    print(env.observation_space.low)  # 显示 observation 最低值
    # DISPLAY_REWARD_THRESHOLD = -2000
    # RENDER = False  # rendering wastes time

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    rl = PPO(
        n_actions=action_size,
        n_features=state_size,
    )

    EP_MAX = 2000
    EP_LEN = 200
    BATCH = 32
    his = train()
    # play()
    plt.ioff()
    plt.show()
