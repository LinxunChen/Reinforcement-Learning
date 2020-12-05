import gym
import os
import matplotlib.pyplot as plt
import numpy as np

from RL_brain import DQN


def plot_reward_and_cost(i_episode, size):
    if i_episode == 0:
        plt.figure()
        plt.ion()
    elif i_episode >= size - 1:
        plt.ioff()
        plt.show()
        return
    plt.clf()
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(len(step_reward_list)), step_reward_list)
    plt.title('each_step_reward')
    plt.subplot(2, 1, 2)
    plt.plot(np.arange(len(RL.each_step_cost_his)), RL.each_step_cost_his)  # 1个episode训练step次，每次训练遍历样本epoch次
    plt.title('each_step_cost')
    plt.pause(0.00000001)


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

env = gym.make('MountainCar-v0')
RL = DQN(3, 2)

## 玩100回合，边玩边产生样本，边训练
step_reward_list = []  # 记录每个step的reward
episode_cnt = 300
total_steps = 0  # 记录步数

for i_episode in range(episode_cnt):
    print('episode:%d' % i_episode)
    observation = env.reset()
    while True:
        env.render()
        action = RL.choose_action(observation)
        observation_, reward, done, info = env.step(action)

        position, velocity = observation_
        # 车开得越高 reward 越大
        reward = abs(position - (-0.5))

        RL.store_transition(observation, action, reward, observation_)

        if total_steps > 1000:
            step_reward_list.append(reward)
            RL.learn()

        if done:
            break

        observation = observation_
        total_steps += 1
    plot_reward_and_cost(i_episode, episode_cnt)

env = env.unwrapped
# RL.plot_cost()
