import gym
import numpy as np
import matplotlib.pyplot as plot
import math

LEADER_SPEED = 0.012


class MultiAgentEnv(gym.Env):
    def __init__(self):
        super(MultiAgentEnv, self).__init__()

        # 定义智能体的数量
        self.n_agents = 3
        self.dirct = 1

        # 初始化智能体的状态
        self.states = np.array([
            [-4.5, -0.09],
            [5.5, 0.12],
            [0.0, LEADER_SPEED]
        ])

        # 定义通信拓扑
        self.adjacency_matrix = np.array([
            [0, 0, 1],
            [1, 0, 1],
        ])
        # 初始化时间步
        self.time_step = 0

        self.positions_history = []
        self.speed_history = []

    def step(self, actions):
        self.positions_history.append(self.states[:, 0].copy())  # 在每一步保存所有智能体的位置
        self.speed_history.append(self.states[:, 1].copy())

        # 更新智能体的状态
        for i in range(self.n_agents - 1):
            action = 0.02 * actions[i][0]
            # action = action * (200 / (300 + self.time_step))
            self.states[i][1] += action
            if self.states[i][1] > 0.50:
                self.states[i][1] = 0.50
            else:
                if self.states[i][1] < -0.5:
                    self.states[i][1] = -0.5

        # self.states[4][1] = LEADER_SPEED - (self.time_step/(5 + self.time_step))*LEADER_SPEED
        # 更新智能体的位置
        for i in range(self.n_agents):
            self.states[i][0] += self.states[i][1]

        n_obs = []

        for i in range(self.n_agents - 1):
            if i < self.dirct:
                obs = [self.states[i][0], self.states[self.n_agents - 1][0], self.states[i][1],
                       self.states[self.n_agents - 1][1]]
            else:
                neighbors = self.adjacency_matrix[i].nonzero()[0]  # 找到智能体的邻居
                neighbor_states = self.states[neighbors]  # 获取邻居的状态

                # 计算位置和速度的均值
                position_mean = neighbor_states[:, 0].mean()

                speed_mean = neighbor_states[:, 1].mean()

                # 将统计量与智能体自己的状态一起作为输入传递给ActorNetwork
                obs = [self.states[i][0], position_mean, self.states[i][1], speed_mean]
            n_obs.append(obs)

        # 计算奖励
        errors = 0.0
        rewards = []
        for i in range(self.n_agents - 1):
            error = -math.sqrt((self.states[i][0] - self.states[self.n_agents - 1][0]) *
                               (self.states[i][0] - self.states[self.n_agents - 1][0]))*0.98
            
            error -= math.sqrt((self.states[i][1] - self.states[self.n_agents - 1][1]) *
                               (self.states[i][1] - self.states[self.n_agents - 1][1]))*0.02
            
            errors += error
            #error -= math.sqrt((n_obs[i][0] - n_obs[i][1]) * (n_obs[i][0] - n_obs[i][1]))
            #error = -(reward)*0.98
            #error -= 0.02* math.sqrt((n_obs[i][2] - n_obs[i][3]) * (n_obs[i][2] - n_obs[i][3]))
            rewards.append(error)

        # 检查是否结束
        done = self.time_step > 300

        # 更新时间步
        self.time_step += 1

        return n_obs, rewards, done, errors/(self.n_agents - 1), {}

    def render(self, mode='human'):
        # 画出每个智能体的位置随时间的变化
        positions_history = np.array(self.positions_history)  # (time_step, n_agents, 2)
        for i in range(self.n_agents - 1):
            if i < self.dirct:
                plot.plot(positions_history[:, i], 'r-', linewidth=2.0)  # 颜色是蓝色
            else:
                plot.plot(positions_history[:, i], 'b-', linewidth=2.0)  # 颜色是红色
        plot.plot(positions_history[:, self.n_agents - 1], 'k-', linewidth=2.0)  # leader 颜色是黑色
        plot.show()

        speed_history = np.array(self.speed_history)  # (time_step, n_agents, 2)
        for i in range(self.n_agents - 1):
            if i < self.dirct:
                plot.plot(speed_history[:, i], 'r-', linewidth=2.0)  # 颜色是红色
            else:
                plot.plot(speed_history[:, i], 'b-', linewidth=2.0)  # 颜色是蓝色
        plot.plot(speed_history[:, self.n_agents - 1], 'k-', linewidth=2.0)  # leader 颜色是黑色
        # plot.ylim([-0.4, 0.4])
        plot.show()

    def reset(self):
        # 重置智能体的状态
        self.states = np.array([
            [-4.5, -0.09],
            [5.5, 0.12],
            [0.0, LEADER_SPEED]
        ])

        n_obs = []

        for i in range(self.n_agents - 1):
            if i < self.dirct:
                obs = [self.states[i][0], self.states[self.n_agents - 1][0], self.states[i][1],
                       self.states[self.n_agents - 1][1]]
            else:
                neighbors = self.adjacency_matrix[i].nonzero()[0]  # 找到智能体的邻居
                neighbor_states = self.states[neighbors]  # 获取邻居的状态

                # 计算位置和速度的均值和标准差
                position_mean = neighbor_states[:, 0].mean()

                speed_mean = neighbor_states[:, 1].mean()

                # 将统计量与智能体自己的状态一起作为输入传递给ActorNetwork
                obs = [self.states[i][0], position_mean, self.states[i][1], speed_mean]
            n_obs.append(obs)

        # 重置时间步
        self.time_step = 0

        self.positions_history = []
        self.speed_history = []

        return n_obs
