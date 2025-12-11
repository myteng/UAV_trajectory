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
from utils.normal import Normalize


class Environment(object):
    def __init__(self):
        # Slot
        self.slot = 2   # 时隙长度
        self.t = 0  # 当前时隙t
        self.alpha_ref = 0.5  # 计算潜在奖励值
        self.eta = 0.5  # 前瞻项奖励值系数
        self.reward_norm = Normalize(shape=3)
        self.reward_weight = np.array([0.4, 0.4, 0.2])
        self.use_potential_reward = args_env.use_potential_reward
        if self.use_potential_reward:
            self.n_feature = 8  # 状态特征长度
        else:
            self.n_feature = 6
        # Reward
        self.cost_energy = 0
        self.cost_time = 0
        self.cost_penalty = 0
        self.reward_task_success = 0

        # Map
        self.map = Map()
        self.grid_load = np.zeros(self.map.n_grids)

        # 障碍物地图，用于碰撞检测（墙/建筑等）
        self.obstacle_map = ObstacleMap()

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

        # 约束相关惩罚系数（可按需调整）
        self.collision_penalty = 100.0   # UAV 发生碰撞（越界或撞障碍）时的惩罚
        self.timeout_penalty = 20.0      # 任务首次超时的惩罚

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

        return self.get_state()

    def step_evaluate(self, action):
        alpha = 0.5  # 时隙比例
        theta = action  # UAV轨迹
        self.cost_energy = 0
        self.cost_time = 0
        self.cost_penalty = 0
        self.reward_task_success = 0

        # Reward
        cost_matrix = self.get_cost(alpha)  # 先计算成本矩阵
        if np.all(np.isinf(cost_matrix)):
            self.get_reward_no_offload(alpha, theta)
        else:
            offload, cost_value = assign_task_with_submatrix(cost_matrix)

            self.get_reward(offload, alpha, theta)

        # Update State
        next_state, reward_value, r_time, r_energy, r_penalty, r_task_success = self.update_state(alpha, theta)

        # is_terminal
        is_all_status_false = not any(job.status for job in self.jobs)
        is_all_finished = all(job.curr_task_id == job.n_task - 1 for job in self.jobs)
        if is_all_finished and is_all_status_false:
            isterminal = True
        else:
            isterminal = False

        return reward_value, r_time, r_energy, r_penalty, r_task_success, next_state, isterminal

    def step(self, action):
        alpha = 0.5  # 时隙比例
        theta = action  # UAV轨迹
        self.cost_energy = 0
        self.cost_time = 0
        self.cost_penalty = 0
        self.reward_task_success = 0

        # Reward
        cost_matrix = self.get_cost(alpha)
        if np.all(np.isinf(cost_matrix)):
            # (1) 无任务卸载
            self.get_reward_no_offload(alpha, theta)
            potential_cost_value = 0
        else:
            # (2) 有任务卸载
            offload, cost_value = assign_task_with_submatrix(cost_matrix)
            # (2.1) 当前时隙潜在奖励值 potential_cost_value
            potential_cost_matrix = self.get_cost(self.alpha_ref)
            if self.use_potential_reward:
                if np.all(np.isinf(potential_cost_matrix)):
                    potential_cost_value = 0
                else:
                    potential_offload, potential_cost = assign_task_with_submatrix(potential_cost_matrix)
                    potential_cost_value = - potential_cost
            else:
                potential_cost_value = 0
            # (2.2) 即时奖励值
            self.get_reward(offload, alpha, theta)

        # Update State
        next_state, reward_value, r_time, r_energy, r_penalty, r_task_success = self.update_state(alpha, theta)

        # (3) 下一个时隙t+1的潜在奖励值 potential_next_cost_value
        is_all_status_false = not any(job.status for job in self.jobs)
        is_all_finished = all(job.curr_task_id == job.n_task - 1 for job in self.jobs)
        if is_all_finished and is_all_status_false:
            isterminal = True
        elif is_all_status_false:
            potential_next_cost_value = 0
            # (4) 计算即时奖励：包含前瞻项
            reward_value = reward_value + self.eta * (potential_next_cost_value - potential_cost_value)
            isterminal = False
        else:
            potential_next_cost_matrix = self.get_cost(self.alpha_ref)
            if self.use_potential_reward:
                if np.all(np.isinf(potential_next_cost_matrix)):
                    potential_next_cost_value = 0
                else:
                    potential_next_offload, potential_next_cost = assign_task_with_submatrix(potential_next_cost_matrix)
                    potential_next_cost_value = - potential_next_cost
            else:
                potential_next_cost_value = 0
            # (4) 即时奖励：包含前瞻项
            reward_value = reward_value + self.eta * (potential_next_cost_value - potential_cost_value)
            isterminal = False

        return reward_value, next_state, isterminal

    # 计算奖励
    def get_reward(self, offload, alpha, theta):
        self.cur_energy_loss = np.zeros(self.n_uav, dtype=np.float32)
        # 悬停+飞行阶段
        e_hov = np.zeros(self.n_uav, dtype=np.float32)
        e_fly = np.zeros(self.n_uav, dtype=np.float32)
        e_turn = np.zeros(self.n_uav, dtype=np.float32)

        # 执行任务阶段
        t_tx = np.zeros(self.n_jobs, dtype=np.float32)
        t_comp = np.zeros(self.n_jobs, dtype=np.float32)
        t_exe = np.zeros(self.n_jobs, dtype=np.float32)
        penalty = np.zeros(self.n_jobs, dtype=np.float32)
        e_tx = np.zeros(self.n_uav, dtype=np.float32)
        e_comp = np.zeros(self.n_uav, dtype=np.float32)

        for u in range(self.n_uav):
            # 悬停能耗
            e_hov[u] = self.uav[u].hov_energy_coef * alpha * self.slot
            # 飞行能耗
            p_fly = self.uav[u].fly_energy_coef * 0.5 * (self.uav[u].vels ** 3)
            direction_factor = 1.0 + 0.5 * (1.0 - math.cos(theta[u]))
            e_fly[u] = p_fly * direction_factor * (1 - alpha) * self.slot
            # 计算转向能耗
            d_theta = theta[u] - self.prev_theta[u]
            d_theta = (d_theta + math.pi) % (2 * math.pi) - math.pi
            e_turn[u] = self.uav[u].turn_energy_coef * (d_theta ** 2)
            for j in range(self.n_jobs):
                # 判断任务在此时刻是否被卸载
                if offload[j][u] == 1:
                    i = self.jobs[j].curr_task_id
                    req_u = self.jobs[j].uav_request[i]
                    if i == 0 and req_u == -1:
                        job_lab = self.jobs[j].lab
                        for v in range(self.n_uav):
                            if self.uav[v].lab == job_lab:
                                req_u = v
                                break
                    self.jobs[j].assigned_uav[i] = u
                    self.jobs[j].uav_offload[i] = u  # 更新task的卸载UAV

                    # 任务 计算时间+传输时间
                    t_comp[j] = self.jobs[j].workload[i] / self.uav[u].p_cm
                    if u == req_u:
                        t_tx[j] = 0
                    else:
                        t_tx[j] = (self.jobs[j].data_size[i] * 1024 * 8) / (self.r_u2u[req_u][u] * 1e6)
                    t_exe[j] = math.ceil((t_tx[j] + t_comp[j]) / self.slot)
                    self.jobs[j].finish_time[i] = self.t + t_exe[j]  # 更新任务的完成时间点
                    self.job_decision_time[j] = self.jobs[j].finish_time[i]   # 更新job下一个可决策的时间点
                    self.uav_free_time[u] = self.jobs[j].finish_time[i]       # 更新UAV的空闲时间点，与job下一个可决策时间点一致
                    # cost_time += round(t_exe[j] / self.jobs[j].deadline, 2)
                    self.cost_time += t_exe[j] * self.slot

                    # 软约束--任务传输时长超过Phase A
                    time_a = t_tx[j] - alpha * self.slot
                    penalty[j] = max(0, time_a)  # 欠传时长
                    # cost_penalty += round(penalty[j] / (alpha * self.slot), 2)
                    self.cost_penalty += penalty[j]

                    # UAV 计算能耗+传输能耗
                    e_comp[u] = self.uav[u].comp_energy_coef * self.uav[u].p_cm ** 3 * t_comp[j]
                    e_tx[req_u] = self.uav[req_u].send_energy_coef * self.uav[req_u].p_tx * t_tx[j]
                    self.cur_energy_loss[u] += e_comp[u]
                    self.cur_energy_loss[req_u] += e_tx[req_u]
                    self.cost_energy += e_comp[u] + e_tx[req_u]
                    # print(f"UAV{u}执行任务{j}产生的能耗为{e_comp[u] + e_tx[req_u]}")

            self.cur_energy_loss[u] += (e_hov[u] + e_fly[u] + e_turn[u])
            self.cost_energy += (e_hov[u] + e_fly[u] + e_turn[u])

    def get_reward_no_offload(self, alpha, theta):
        self.cur_energy_loss = np.zeros(self.n_uav, dtype=np.float32)
        # 悬停+飞行阶段
        e_hov = np.zeros(self.n_uav, dtype=np.float32)
        e_fly = np.zeros(self.n_uav, dtype=np.float32)
        e_turn = np.zeros(self.n_uav, dtype=np.float32)
        # 计算成本值
        cost_energy = 0
        for u in range(self.n_uav):
            # 悬停能耗
            e_hov[u] = self.uav[u].hov_energy_coef * alpha * self.slot
            # 飞行能耗
            p_fly = self.uav[u].fly_energy_coef * 0.5 * (self.uav[u].vels ** 3)
            direction_factor = 1.0 + 0.5 * (1.0 - math.cos(theta[u]))
            e_fly[u] = p_fly * direction_factor * (1 - alpha) * self.slot
            # 计算转向能耗
            d_theta = theta[u] - self.prev_theta[u]
            d_theta = (d_theta + math.pi) % (2 * math.pi) - math.pi
            e_turn[u] = self.uav[u].turn_energy_coef * (1 - math.cos(d_theta))
            # UAV 飞行+悬停能耗
            self.cur_energy_loss[u] += (e_hov[u] + e_fly[u] + e_turn[u])
            # cost_energy += round((e_hov[u] + e_fly[u]) / self.uav[u].current_energy, 2)
            self.cost_energy += (e_hov[u] + e_fly[u] + e_turn[u])

    def update_state(self, alpha, theta):
        # 更新当前时刻
        self.t += 1
        # 更新UAV状态
        for u in range(self.n_uav):
            # UAV 位置（先计算候选新位置）
            prev_pos = self.uav[u].pos.copy()
            pos_x = prev_pos[0] + self.uav[u].vels * (1 - alpha) * self.slot * np.cos(theta[u])
            pos_y = prev_pos[1] + self.uav[u].vels * (1 - alpha) * self.slot * np.sin(theta[u])
            pos_z = prev_pos[2]

            # 边界碰撞检测：是否飞出地图范围
            out_of_bounds = (
                pos_x < 0 or pos_x > args_env.ranges_x or
                pos_y < 0 or pos_y > args_env.ranges_y
            )

            # 障碍物碰撞检测：飞行路径是否被任意障碍阻挡
            hit_obstacle = self.obstacle_map.is_path_blocked_3d(
                tuple(prev_pos), (pos_x, pos_y, pos_z)
            )

            if out_of_bounds or hit_obstacle:
                # 碰撞：增加惩罚，并将位置保持在上一步（不穿过障碍/边界）
                self.cost_penalty += self.collision_penalty
                # 也可以选择裁剪到边界，这里保持上一位置更安全
                self.uav[u].pos = prev_pos
                pos_x, pos_y, pos_z = prev_pos[0], prev_pos[1], prev_pos[2]
            else:
                # 正常更新位置
                self.uav[u].pos = np.array([pos_x, pos_y, pos_z])
            # UAV label
            x_lab = math.ceil(pos_x / args_env.range_pos) - 1
            y_lab = math.ceil(pos_y / args_env.range_pos) - 1
            n_grids = args_env.ranges_x / args_env.range_pos
            self.uav[u].lab = x_lab + y_lab * n_grids
            # UAV 剩余能耗
            self.uav[u].current_energy -= self.cur_energy_loss[u]
            # UAV 空闲状态
            if self.uav_free_time[u] > self.t:
                self.uav[u].free = False
            elif self.uav_free_time[u] == self.t:
                self.uav[u].free = True
            # 上一个时刻的移动轨迹（动作）
            self.uav[u].move_x = self.uav[u].vels * np.cos(theta[u])
            self.uav[u].move_y = self.uav[u].vels * np.sin(theta[u])

        self.get_u2u()

        # 更新Job状态
        for j in range(self.n_jobs):
            # 更新Job的执行状态
            # print(f"job_id = {j}, job_decision_time = {self.job_decision_time[j]}, next_time_slot = {self.t}, curr_task_id = {self.jobs[j].curr_task_id}, n_task = {self.jobs[j].n_task}")
            if self.job_decision_time[j] > self.t:
                # 当前时刻job的task还未执行完
                self.jobs[j].status = False
            elif self.job_decision_time[j] == self.t:
                # 当前时刻job的task已执行完，准备决策下一个task
                i = self.jobs[j].curr_task_id
                if i < (self.jobs[j].n_task - 1):
                    self.jobs[j].status = True   # 更新job执行状态
                    self.jobs[j].curr_task_id = i + 1  # 更新job当前准备决策的任务
                    self.jobs[j].start_time[i + 1] = self.t  # 更新job当前决策任务的开始时间点
                    self.jobs[j].uav_request[i + 1] = self.jobs[j].uav_offload[i]  # 更新job当前决策任务的发送UAV
                else:
                    # Job的所有task都执行完，则更新状态为False
                    # print(f"job {j} 执行完，status 为{self.jobs[j].status}")
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
                # 任务超时惩罚：仅在首次超时越界时施加一次固定惩罚
                if self.jobs[j].deadline < 0:
                    # 使用动态属性标记是否已经对该 job 施加超时惩罚
                    if not hasattr(self.jobs[j], 'timeout_flag') or not self.jobs[j].timeout_flag:
                        self.cost_penalty += self.timeout_penalty
                        self.jobs[j].timeout_flag = True

        # 手工加权 Reward：碰撞/超时惩罚 > 按时完成任务 > 能耗/时间 + 少量角度多样性奖励
        # 说明：
        #   - self.cost_penalty：包含碰撞、超时、欠传等惩罚（越大越差）
        #   - self.reward_task_success：按时完成任务数量（越大越好）
        #   - self.cost_time：执行时间成本
        #   - self.cost_energy：能耗成本
        #   - theta：本时刻各 UAV 的方向角（来自动作），方差越大表示方向越分散
        penalty_term = self.cost_penalty
        success_term = self.reward_task_success
        time_term = self.cost_time
        energy_term = self.cost_energy
        angle_diversity = float(np.var(theta))

        # 原始 reward（未缩放），数值可能较大
        raw_reward = (
            -1.0 * penalty_term        # 强烈惩罚碰撞/超时/欠传等
            +0.5 * success_term        # 按时完成任务的正向奖励
            -0.3 * time_term           # 时间越长越差
            -0.1 * energy_term         # 能耗次要，但也希望越小越好
            +0.1 * angle_diversity     # 适度鼓励各 UAV 方向角不要完全一致
        )

        # 为了让数值落在一个较稳定的范围内，这里做一个简单缩放
        reward_value = float(raw_reward / 50.0)

        return self.get_state(), reward_value, -self.cost_time, -self.cost_energy, -self.cost_penalty, self.reward_task_success

    def get_state(self):
        for uav_id in range(self.n_uav):
            # self.uav_feats[uav_id, 0] = round(self.uav[uav_id].pos[0] / args_env.ranges_x, 2)
            # self.uav_feats[uav_id, 1] = round(self.uav[uav_id].pos[1] / args_env.ranges_y, 2)
            # self.uav_feats[uav_id, 2] = round(self.uav[uav_id].current_energy / args_env.max_energy_uav, 2)
            # self.uav_feats[uav_id, 3] = round(np.sum(self.r_u2u[uav_id, :] == 0) / self.n_uav, 2)
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

    # 计算匈牙利算法的成本矩阵
    def get_cost(self, alpha):
        # 初始化成本矩阵和权重
        cost_weight = [0.4, 0.4, 0.2]
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
                            # if self.jobs[j].deadline == 0:
                            #     cost_t = 2
                            # else:
                            #     cost_t = round(t_exe / self.jobs[j].deadline, 2)
                            cost_t = t_exe * self.slot
                            # 能耗成本
                            cost_e_tx = round(e_tx / self.uav[u].current_energy, 2)
                            cost_e_comp = round(e_comp / self.uav[u].current_energy, 2)
                            cost_e = e_comp + e_tx

                            # A段未传完惩罚项
                            time_a = t_tx - alpha * self.slot
                            delta = max(0, time_a)  # 欠传时长
                            # cost_fail = round(delta / time_a, 2)  # 惩罚项
                            cost_fail = delta

                            # 总成本
                            cost_set = [cost_t, cost_e, cost_fail]
                            # cost_set = self.reward_norm(cost_set)
                            cost_set = np.array(cost_set)
                            cost_matrix[j][u] = round(np.sum(cost_set * cost_weight), 2)
                            if cost_matrix[j][u] == np.nan:
                                raise ValueError("Cost is NaN!")
                            # cost_matrix[j][u] = cost_weight[0] * cost_t + cost_weight[1] * (cost_e_tx + cost_e_comp) + cost_weight[2] * cost_fail
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
        # print(self.r_u2u)


# 示例用法（若作为独立脚本运行）
if __name__ == '__main__':
    for e in range(1):
        seed = 128
        set_rand_seed(seed)

        args_env.ranges_x = 300
        args_env.ranges_y = 300
        args_env.n_uav = 9
        args_env.n_jobs = 30

        env = Environment()
        state = env.reset()
        # for j in range(env.n_jobs):
        #     print(f"job_id = {j}, job_lab = {env.jobs[j].lab}")

        for t in range(10):
            print(f"----------------------------第{e}次，第{t}步--------------------------------------")
            print(np.array2string(state, formatter={'float_kind': lambda x: f"{x:.2f}"}))

            action = np.random.uniform(low=0, high=2 * np.pi, size=env.n_uav+1)
            action[0] = 0.5  # 固定alpha
            print(f"action = {action}")

            reward, next_state, isterminal = env.step_evaluate(action)
            print(f"Step = {t}, Reward = {reward}")

            state = next_state


