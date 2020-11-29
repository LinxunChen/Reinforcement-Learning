import gym
from RL_brain import DQN
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

env = gym.make('MountainCar-v0')
RL = DQN(3, 2)
total_steps = 0  # 记录步数

## 玩100回合，边玩边产生样本，边训练
reward_list = []
for i_episode in range(1000):
    print('episode:%d' % i_episode)
    observation = env.reset()
    reward_sum = 0
    while True:
        env.render()
        action = RL.choose_action(observation)
        observation_, reward, done, info = env.step(action)

        position, velocity = observation_
        # 车开得越高 reward 越大
        reward = abs(position - (-0.5))

        RL.store_transition(observation, action, reward, observation_)
        reward_sum += reward

        if total_steps > 1000:
            RL.learn()

        if done:
            reward_list.append(reward_sum)
            break

        observation = observation_
        total_steps += 1
    print('reward_list:%s' % reward_list)

env = env.unwrapped
RL.plot_cost()

## 游戏预览
## 参考文档：https://github.com/openai/gym/blob/master/gym/envs/classic_control/mountain_car.py
# for i_episode in range(3):
#     observation = env.reset() ##状态初始化，初始化一个高度，速度置为0
#     for t in range(1000):
#         env.render() ## 画面刷新
#         print(observation)
#         action = env.action_space.sample()
#         print(action)
#         observation, reward, done, info = env.step(action)
#         print(reward)
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             break
# env.close()
