import math
import random
from typing import List

import numpy as np
from shapely import Polygon
from concurrent.futures import ProcessPoolExecutor, as_completed

from algos.hungarian import assign_tasks, assign_task_with_submatrix
from algos.informedRRT import InformedRRTStar
from envs.channel import Channel
from envs.jobs import Job
from envs.map import Map
from envs.obstacle import ObstacleMap
from envs.uav import UAV
from parameter.paramEnv import args_env
from utils.common import set_rand_seed
from utils.normal import Normalize, RewardScaling

"""
固定alpha，环境状态S_t先采用匈牙利算法决策卸载O，然后更新环境状态S'，输入RL算法，
"""


class Environment_1(object):
    def __init__(self):
        # Slot
        self.slot = 4   # 时隙长度
        self.t = 0  # 当前时隙t
        self.alpha_ref = 0.5  # 计算潜在奖励值
        self.eta = 0.5  # 前瞻项奖励值系数
        self.reward_norm = Normalize(shape=3)
        # self.reward_norm = RewardScaling(shape=1, gamma=0.9)
        self.reward_weight = np.array([0.4, 0.4, 0.2])
        self.use_potential_reward = args_env.use_potential_reward
        if self.use_potential_reward:
            self.n_feature = 8  # 状态特征长度
        else:
            self.n_feature = 6

        # Map
        self.map = Map()
        self.grid_load = np.zeros(self.map.n_grids)

        # UAVs
        self.n_uav = args_env.n_uav
        self.uav = [UAV(uav_id) for uav_id in range(self.n_uav)]
        self.r_u2u = np.zeros((self.n_uav, self.n_uav), dtype=np.float32)
        self.cur_energy_loss = np.zeros(self.n_uav, dtype=np.float32)   # 当前时刻UAV的能耗损耗量
        self.uav_free_time = np.zeros(self.n_uav, dtype=np.float32)     # UAV空闲的时刻
        self.prev_theta = np.zeros(self.n_uav, dtype=np.float32)

        # Jobs
        self.n_jobs = args_env.n_jobs
        self.jobs = [Job(job_id) for job_id in range(self.n_jobs)]
        self.r_j2u = np.zeros((self.n_jobs, self.n_uav), dtype=np.float32)
        self.job_decision_time = np.zeros(self.n_jobs, dtype=np.float32)  # Job被决策的时刻

        # Channel
        self.channel = Channel()
        self.uav_max_distance = args_env.max_distance

        # Hop Matrix
        self.H = np.zeros((self.n_jobs, self.n_uav), dtype=np.float32)

        # Cost Matrix
        self.cost = np.zeros((self.n_jobs, self.n_uav), dtype=np.float32)
        self.cost_time = 0  # 当前时刻的时间成本
        self.cost_energy = 0  # 当前时刻的能耗成本
        self.cost_penalty = 0
        self.reward_task_success = 0
        self.t_tx = np.zeros(self.n_jobs, dtype=np.float32)
        # self.offload = np.zeros((self.n_jobs, self.n_uav), dtype=np.float32)

        # State
        self.state = np.zeros((self.n_uav, self.n_feature),dtype=np.float32)
        self.uav_feats = np.zeros((self.n_uav, self.n_feature), dtype=np.float32)
        self.job_feats = np.zeros((self.n_jobs, self.n_feature), dtype=np.float32)

    def reset(self):
        self.t = 0
        # Reset UAV Property
        for uav_id in range(self.n_uav):
            self.uav[uav_id].reset_uav(uav_id)
        self.get_u2u()

        # Reset Jobs Property
        for job_id in range(self.n_jobs):
            self.jobs[job_id].reset_jobs(job_id)
            grid_id = self.jobs[job_id].lab
            # Load Distribution
            self.grid_load[grid_id] = self.grid_load[grid_id] + 1

        if any(job.status for job in self.jobs):
            # 匈牙利算法卸载，先计算成本矩阵
            cost_matrix = self.get_cost()
            offload, cost_value = assign_task_with_submatrix(cost_matrix)
            state = self.update_state_hungarian(offload)
        else:
            state = self.get_state()
        # 计算Reward：
        # 归一化，求Reward值
        reward_set = np.array([-self.cost_time, -self.cost_energy, -self.cost_penalty])
        reward_set = self.reward_norm(reward_set)
        reward_value = round(np.sum(reward_set * self.reward_weight), 2)
        return state, reward_value, -self.cost_time, -self.cost_energy, -self.cost_penalty

    def step_asy(self, action):
        alpha = 0.5  # 时隙比例
        theta = action  # UAV轨迹
        self.cost_energy = 0
        self.cost_time = 0
        self.cost_penalty = 0
        self.reward_task_success = 0
        self.t += 1

        # 先计算Reward值，更新UAV状态
        for u in range(self.n_uav):
            # 更新 UAV 剩余能耗
            # 悬停能耗
            e_hov = self.uav[u].hov_energy_coef * alpha * self.slot
            # 飞行能耗
            p_fly = self.uav[u].fly_energy_coef * 0.5 * (self.uav[u].vels ** 3)
            direction_factor = 1.0 + 0.5 * (1.0 - math.cos(theta[u]))
            e_fly = p_fly * direction_factor * (1 - alpha) * self.slot
            # 计算转向能耗
            d_theta = theta[u] - self.prev_theta[u]
            d_theta = (d_theta + math.pi) % (2 * math.pi) - math.pi
            e_turn = self.uav[u].turn_energy_coef * (1 - math.cos(d_theta))
            e_total = e_fly + e_hov + e_turn
            self.cost_energy += e_total
            # print(f"UAV{u}飞行过程产生能耗{e_fly + e_hov}")
            # print(f"当前损耗总能耗为：{self.cost_energy}")
            self.uav[u].current_energy -= e_total
            # 更新 UAV 位置
            pos_x = self.uav[u].pos[0] + self.uav[u].vels * (1 - alpha) * self.slot * np.cos(theta[u])
            pos_y = self.uav[u].pos[1] + self.uav[u].vels * (1 - alpha) * self.slot * np.sin(theta[u])
            pos_z = self.uav[u].pos[2]
            self.uav[u].pos = np.array([pos_x, pos_y, pos_z])
            # 更新 UAV 空闲状态
            if self.uav_free_time[u] > self.t:
                self.uav[u].free = False
            elif self.uav_free_time[u] == self.t:
                self.uav[u].free = True
            # 上一个时刻的移动轨迹（动作）
            self.uav[u].move_x = self.uav[u].vels * np.cos(theta[u])
            self.uav[u].move_y = self.uav[u].vels * np.sin(theta[u])
        # 更新上个轨迹方向
        self.prev_theta = theta
        # 更新U2U通信
        self.get_u2u()

        # 更新Job状态
        for j in range(self.n_jobs):
            # 当前时刻job的task还未执行完
            if self.job_decision_time[j] > self.t:
                self.jobs[j].status = False
            # 当前时刻job的task已执行完，准备决策下一个task
            elif self.job_decision_time[j] == self.t:
                i = self.jobs[j].curr_task_id
                if i < (self.jobs[j].n_task - 1):
                    self.jobs[j].status = True  # 更新job执行状态
                    self.jobs[j].curr_task_id = i + 1  # 更新job当前准备决策的任务
                    self.jobs[j].start_time[i + 1] = self.t  # 更新job当前决策任务的开始时间点
                    self.jobs[j].uav_request[i + 1] = self.jobs[j].uav_offload[i]  # 更新job当前决策任务的发送UAV
                else:
                    # Job的所有task都执行完，则更新状态为False
                    self.jobs[j].status = False
                    self.jobs[j].finish = True
            # 任务完成奖励值
            if self.jobs[j].finish and self.jobs[j].deadline >= 0:
                self.reward_task_success += 1
                self.cost_penalty += -10
            elif self.jobs[j].finish:
                self.cost_penalty += -5
            else:
                # 更新deadline（无论是否被执行，deadline都要减1）
                self.jobs[j].deadline -= self.slot
                if self.jobs[j].deadline < 0:
                    self.cost_penalty += np.abs(self.jobs[j].deadline)

            # 软约束--任务传输时长超过Phase A
            time_a = self.t_tx[j] - alpha * self.slot
            self.cost_penalty += max(0, time_a)  # 欠传时长

        # 下一个时隙t+1的卸载决策，匈牙利算法卸载。
        cost_matrix = self.get_cost()  # 先计算成本矩阵
        if np.all(np.isinf(cost_matrix)):
            next_state = self.get_state()
        else:
            offload, cost_value = assign_task_with_submatrix(cost_matrix)
            next_state = self.update_state_hungarian(offload)

        # 下一个时隙t+1的潜在奖励值 potential_next_cost_value
        is_all_status_false = not any(job.status for job in self.jobs)
        is_all_finished = all(job.curr_task_id == job.n_task - 1 for job in self.jobs)
        if is_all_finished and is_all_status_false:
            isterminal = True
        else:
            isterminal = False

        # 计算Reward：
        # # 归一化，求Reward值
        reward_set = np.array([-self.cost_time, -self.cost_energy, -self.cost_penalty])
        # reward_set = self.reward_norm(reward_set)
        reward_set = np.array(reward_set)
        reward_value = round(np.sum(reward_set * self.reward_weight), 2)
        # reward_value = -self.cost_time - self.cost_penalty

        return reward_value, -self.cost_time, -self.cost_energy, -self.cost_penalty, self.reward_task_success, next_state, isterminal

    # 更新匈牙利算法之后的状态，并返回卸载的成本（所有任务执行的时间成本、能耗成本）
    def update_state_hungarian(self, offload):
        self.t_tx = np.zeros(self.n_jobs, dtype=np.float32)
        for u in range(self.n_uav):
            for j in range(self.n_jobs):
                # 判断任务在此时刻是否被卸载
                if offload[j][u] == 1:
                    i = self.jobs[j].curr_task_id
                    req_u = self.jobs[j].uav_request[i]
                    if i == 0 and req_u == -1:
                        self.jobs[j].start = True
                        job_lab = self.jobs[j].lab
                        for v in range(self.n_uav):
                            if self.uav[v].lab == job_lab:
                                req_u = v
                                break
                    self.jobs[j].assigned_uav[i] = u
                    self.jobs[j].uav_offload[i] = u  # 更新task的卸载UAV

                    # 任务 计算时间+传输时间
                    t_comp = self.jobs[j].workload[i] / self.uav[u].p_cm
                    if u == req_u:
                        t_tx = 0
                    else:
                        t_tx = (self.jobs[j].data_size[i] * 1024 * 8) / (self.r_u2u[req_u][u] * 1e6)
                    self.t_tx[j] = t_tx
                    t_exe = math.ceil((t_tx + t_comp) / self.slot)
                    self.jobs[j].finish_time[i] = self.t + t_exe  # 更新任务的完成时间点
                    self.job_decision_time[j] = self.jobs[j].finish_time[i]   # 更新job下一个可决策的时间点
                    self.uav_free_time[u] = self.jobs[j].finish_time[i]       # 更新UAV的空闲时间点，与job下一个可决策时间点一致
                    self.cost_time += t_exe * self.slot

                    # UAV 计算能耗+传输能耗
                    e_comp = self.uav[u].comp_energy_coef * self.uav[u].p_cm ** 3 * t_comp
                    e_tx = self.uav[req_u].send_energy_coef * self.uav[req_u].p_tx * t_tx
                    self.uav[u].current_energy -= e_comp
                    self.uav[req_u].current_energy -= e_tx
                    self.cost_energy += e_comp + e_tx
                    # print(f"UAV{u}执行任务{j}产生能耗{e_comp + e_tx}")
                    # print(f"当前损耗总能耗为：{self.cost_energy}")
        return self.get_state()

    def get_state(self):
        for uav_id in range(self.n_uav):
            self.uav_feats[uav_id, 0] = self.uav[uav_id].pos[0]
            self.uav_feats[uav_id, 1] = self.uav[uav_id].pos[1]
            self.uav_feats[uav_id, 2] = self.uav[uav_id].current_energy
            self.uav_feats[uav_id, 3] = np.sum(self.r_u2u[uav_id, :] > 0)
            self.uav_feats[uav_id, 4] = 1 if self.uav[uav_id].free else 0
            self.uav_feats[uav_id, 5] = sum(1 for job in self.jobs if job.uav_request[job.curr_task_id] == uav_id)
            if self.use_potential_reward:
                self.uav_feats[uav_id, 6] = self.uav[uav_id].move_x
                self.uav_feats[uav_id, 7] = self.uav[uav_id].move_y

        for job_id in range(self.n_jobs):
            curr_task_id = self.jobs[job_id].curr_task_id
            # print(f"job_id = {job_id}, status = {self.jobs[job_id].status}")
            if self.jobs[job_id].status:
                self.job_feats[job_id, 0] = round((curr_task_id + 1) / self.jobs[job_id].n_task, 2)
                self.job_feats[job_id, 1] = round(self.jobs[job_id].workload[curr_task_id] / args_env.workload_max, 2)
                self.job_feats[job_id, 2] = round(self.jobs[job_id].data_size[curr_task_id] / args_env.data_size_max, 2)
                self.job_feats[job_id, 3] = round(self.jobs[job_id].deadline / self.jobs[job_id].deadline_max, 2)

        self.state[:self.n_uav, :] = self.uav_feats
        # self.state[self.n_uav:, :] = job_feats
        return self.state

    # 计算匈牙利算法的成本矩阵（不考虑alpha的成本，只有时间和能耗成本）
    def get_cost(self):
        # 初始化成本矩阵和权重
        cost_matrix = np.full((self.n_jobs, self.n_uav), np.inf, dtype=np.float32)

        for j in range(self.n_jobs):
            # 判断当前时刻job是否有task执行
            if self.jobs[j].status:
                i = self.jobs[j].curr_task_id
                req_u = self.jobs[j].uav_request[i]
                if i == 0 and req_u == -1:
                    job_lab = self.jobs[j].lab
                    for v in range(self.n_uav):
                        if self.uav[v].lab == job_lab:
                            req_u = v
                            break
                for u in range(self.n_uav):
                    # 先判断当前时刻UAV是否空闲
                    if self.uav[u].free:
                        # 判断发送UAV与接收UAV之间是否有通信链路，如果没有则成本为inf
                        if self.r_u2u[req_u][u] != 0:
                            # 任务计算时间
                            t_comp = round(self.jobs[j].workload[i] / self.uav[u].p_cm, 2)
                            # 传输时间
                            if u == req_u:
                                t_tx = 0
                            else:
                                t_tx = round((self.jobs[j].data_size[i] * 1024 * 8) / (self.r_u2u[req_u][u] * 1e6), 2)
                            # 总执行时间
                            t_exe = math.ceil((t_tx + t_comp) / self.slot)

                            # 计算能耗
                            e_comp = self.uav[u].comp_energy_coef * self.uav[u].p_cm ** 3 * t_comp
                            # 传输能耗
                            e_tx = self.uav[req_u].send_energy_coef * self.uav[req_u].p_tx * t_tx

                            # 时间成本
                            cost_t = t_exe * self.slot
                            # 能耗成本
                            cost_e = e_comp + e_tx
                            # 软约束--任务传输时长超过Phase A
                            time_a = t_tx - 0.5 * self.slot
                            cost_penalty = max(0, time_a)  # 欠传时长

                            # 总成本
                            cost_set = [cost_t, cost_e, cost_penalty]
                            # cost_set = self.reward_norm(cost_set)
                            cost_set = np.array(cost_set)
                            cost_matrix[j][u] = round(np.sum(cost_set * self.reward_weight), 2)
                            if cost_matrix[j][u] == np.nan:
                                raise ValueError("Cost is NaN!")
        return cost_matrix

    def get_u2u(self):
        for uav_id_i in range(self.n_uav):
            for uav_id_j in range(self.n_uav):
                if uav_id_i == uav_id_j:
                    self.r_u2u[uav_id_i][uav_id_j] = np.inf
                elif uav_id_i > uav_id_j:
                    self.r_u2u[uav_id_i][uav_id_j] = self.r_u2u[uav_id_j][uav_id_i]
                else:
                    self.r_u2u[uav_id_i][uav_id_j] = self.channel.get_u2u_rate(self.uav[uav_id_i], self.uav[uav_id_j])


# 示例用法（若作为独立脚本运行）
if __name__ == '__main__':
    for e in range(2):
        seed = 128
        set_rand_seed(seed)

        args_env.ranges_x = 300
        args_env.ranges_y = 300
        args_env.n_uav = 9
        args_env.n_jobs = 30

        env = Environment_1()
        state = env.reset()

        for t in range(10):
            print(f"----------------------------第{e}次，第{t}步--------------------------------------")
            print(np.array2string(state, formatter={'float_kind': lambda x: f"{x:.2f}"}))

            action = np.random.uniform(low=0, high=2 * np.pi, size=env.n_uav+1)
            action[0] = 0.5  # 固定alpha
            print(f"action = {action}")

            reward, next_state, isterminal = env.step(action)
            print(f"Step = {t}, Reward = {reward}")

            state = next_state


