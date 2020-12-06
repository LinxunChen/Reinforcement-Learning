import gym
import os
import matplotlib.pyplot as plt
import numpy as np
from RL_brain import DQN

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def plot_reward_and_cost(i_episode, size, history):
    if i_episode == 0:
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


def train(episodes):
    # total_steps = 0  # 记录步数
    history = {'episode': [], 'Episode_reward': [], 'Loss': []}

    for i_episode in range(episodes):
        observation = env.reset()
        reward_sum = 0
        loss = np.infty
        while True:
            env.render()
            action = rl.choose_action(observation)
            observation_, reward, done, info = env.step(action)

            # 车开得越高 reward 越大
            position, velocity = observation_
            reward = abs(position - (-0.5))

            reward_sum += reward
            rl.store_transition(observation, action, reward, observation_, done)

            if done:
                loss = rl.learn()
                break

            observation = observation_
            # total_steps += 1

        if i_episode % 5 == 0:
            history['episode'].append(i_episode)
            history['Episode_reward'].append(reward_sum)
            history['Loss'].append(loss)
            print(
                'Episode: {}/{} | Episode reward: {} | loss: {:.6f} | e:{:.2f}'.format(i_episode, episodes, reward_sum,
                                                                                       loss, rl.epsilon))
            plot_reward_and_cost(i_episode, episodes, history)
    return history
    # ENV = ENV.unwrapped
    # RL.plot_cost()


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


if __name__ == '__main__':
    # 玩500回合，边玩边产生样本，边训练
    env = gym.make('MountainCar-v0')
    env.seed(1)
    print(env.action_space)  # 显示可用 action
    print(env.observation_space)  # 显示可用 state 的 observation
    print(env.observation_space.high)  # 显示 observation 最高值
    print(env.observation_space.low)  # 显示 observation 最低值
    # DISPLAY_REWARD_THRESHOLD = -2000
    # RENDER = False  # rendering wastes time

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    rl = PolicyGradient(
        n_actions=env.action_space.n,
        n_features=env.observation_space.shape[0],
        learning_rate=0.01,
        reward_decay=0.995,
    )

    EPISODES = 500
    his = train(EPISODES)
    play()
    plt.ioff()
    plt.show()
