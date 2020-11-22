import gym
from RL_brain import DQN

env = gym.make('MountainCar-v0')
RL = DQN(3, 2)
total_steps = 0 # 记录步数

## 玩100回合，边玩边产生样本，边训练
for i_episode in range(100):
    observation = env.reset()

    while True:
        env.render()
        action = RL.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        RL.store_transition(observation, action, reward, observation_)

        if total_steps > 1000:
            RL.learn()

        if done:
            break

        observation = observation_
        total_steps += 1

RL.plot_cost()
env = env.unwrapped


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