import math
import random
from typing import List

import numpy as np
from shapely import Polygon
from concurrent.futures import ProcessPoolExecutor, as_completed

from algos.hungarian import assign_tasks, assign_task_with_submatrix
from algos.informedRRT import InformedRRTStar
from envs.channel import Channel
from envs.hotspots import compute_hotspots_from_jobs, assign_hotspots_to_uav
from envs.jobs import Job
from envs.map import Map
from envs.obstacle import ObstacleMap
from envs.uav import UAV
from parameter.paramEnv import args_env
from utils.common import set_rand_seed
from utils.normal import Normalize, RewardScaling
from utils.save_txt import TXT_FILE

"""
固定alpha，环境状态S_t先采用匈牙利算法决策卸载O，然后更新环境状态S'，输入RL算法，
"""


class Environment_1(object):
    def __init__(self):
        # Slot
        self.slot = 4   # 时隙长度
        self.t = 0  # 当前时隙t
        self.alpha_ref = 0.5  # 计算潜在奖励值
        self.alpha = 0.5
        self.eta = 0.5  # 前瞻项奖励值系数
        self.reward_norm = Normalize(shape=3)
        # self.reward_norm = RewardScaling(shape=1, gamma=0.9)
        self.reward_weight = np.array([0.4, 0.4, 0.2])
        self.use_potential_reward = args_env.use_potential_reward
        # if self.use_potential_reward:
        #     self.n_feature = 8  # 状态特征长度
        # else:
        #     self.n_feature = 6
        if self.use_potential_reward:
            self.n_feature = 9  # 状态特征长度
        else:
            self.n_feature = 7
        
        self.save_txt = TXT_FILE()

        # Map
        self.map = Map()
        self.grid_load = np.zeros(self.map.n_grids)
        # 动态任务负载（每步更新，反映剩余任务分布）
        self.grid_load_dynamic = np.zeros(self.map.n_grids)

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

        # Hotspots
        self.n_hotspots = 1
        self.hotspots = [None for h in range(self.n_hotspots)]
        self.uav_hotspots = [None for u in range(self.n_uav)]
        self.uav_pre_pos = []

        # Cost Matrix
        self.cost = np.zeros((self.n_jobs, self.n_uav), dtype=np.float32)
        self.cost_time = 0  # 当前时刻的时间成本
        self.cost_energy = 0  # 当前时刻的能耗成本
        self.cost_penalty = 0
        self.penalty_trajectory = 0
        self.penalty_energy = 0
        self.penalty_task = 0
        self.reward_job_success = 0
        self.reward_task_num = 0
        self.reward_eer = 0
        self.reward_hotspots = 0
        self.reward_deadline_margin = 0.0
        self.reward_coverage = 0.0

        self.t_tx = np.zeros(self.n_jobs, dtype=np.float32)
        # self.offload = np.zeros((self.n_jobs, self.n_uav), dtype=np.float32)

        # State
        self.state = np.zeros((self.n_uav, self.n_feature),dtype=np.float32)
        self.uav_feats = np.zeros((self.n_uav, self.n_feature), dtype=np.float32)
        self.job_feats = np.zeros((self.n_jobs, self.n_feature), dtype=np.float32)

    def reset(self):
        self.t = 0
        self.cost_time = 0  # 当前时刻的时间成本
        self.cost_energy = 0  # 当前时刻的能耗成本
        self.cost_penalty = 0
        self.penalty_trajectory = 0
        self.penalty_energy = 0
        self.penalty_task = 0
        self.reward_job_success = 0
        self.reward_task_num = 0
        self.reward_eer = 0
        self.reward_hotspots = 0
        self.reward_deadline_margin = 0.0
        self.reward_coverage = 0.0

        # Reset UAV Property
        for uav_id in range(self.n_uav):
            self.uav[uav_id].reset_uav(uav_id)
        self.get_u2u()

        # Reset Jobs Property
        self.grid_load = np.zeros(self.map.n_grids)  # 重置静态负载
        self.grid_load_dynamic = np.zeros(self.map.n_grids)  # 重置动态负载
        for job_id in range(self.n_jobs):
            self.jobs[job_id].reset_jobs(job_id)
            grid_id = self.jobs[job_id].lab
            # Load Distribution（初始任务分布）
            self.grid_load[grid_id] = self.grid_load[grid_id] + 1
            self.grid_load_dynamic[grid_id] = self.grid_load_dynamic[grid_id] + 1

        # Hotspots
        self.hotspots = compute_hotspots_from_jobs(self.jobs, self.n_hotspots)
        assigned_idx, self.uav_hotspots = assign_hotspots_to_uav(self.uav, self.hotspots)

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
        self.alpha = 0.5  # 时隙比例
        theta = action  # UAV轨迹
        self.t += self.slot

        self.cost_time = 0  # 当前时刻的时间成本
        self.cost_energy = 0  # 当前时刻的能耗成本
        self.cost_penalty = 0
        self.penalty_trajectory = 0
        self.penalty_energy = 0
        self.penalty_task = 0
        self.reward_job_success = 0
        self.reward_task_num = 0
        self.reward_eer = 0
        self.reward_hotspots = 0
        self.reward_deadline_margin = 0.0
        self.reward_coverage = 0.0
        self.uav_pre_pos = []

        # 先计算Reward值，更新UAV状态
        for u in range(self.n_uav):
            # 更新 UAV 剩余能耗
            # 悬停能耗
            e_hov = self.uav[u].hov_energy_coef * self.alpha * self.slot
            # 飞行能耗
            p_fly = self.uav[u].fly_energy_coef * 0.5 * (self.uav[u].vels ** 3)
            direction_factor = 1.0 + 0.5 * (1.0 - math.cos(theta[u]))
            e_fly = p_fly * direction_factor * (1 - self.alpha) * self.slot
            # 计算转向能耗
            d_theta = theta[u] - self.prev_theta[u]
            d_theta = (d_theta + math.pi) % (2 * math.pi) - math.pi
            e_turn = self.uav[u].turn_energy_coef * (1 - math.cos(d_theta))
            # e_turn = 0
            e_total = e_fly + e_hov + e_turn
            self.cost_energy += e_total
            self.uav[u].current_energy -= e_total

            # 更新 UAV 位置
            prev_pos = self.uav[u].pos.copy()
            self.uav_pre_pos.append(prev_pos)
            pos_x = self.uav[u].pos[0] + self.uav[u].vels * (1 - self.alpha) * self.slot * np.cos(theta[u])
            pos_y = self.uav[u].pos[1] + self.uav[u].vels * (1 - self.alpha) * self.slot * np.sin(theta[u])
            pos_z = self.uav[u].pos[2]
            # 边界碰撞检测：是否飞出地图范围
            out_of_bounds = (
                    pos_x < 0 or pos_x > args_env.ranges_x or
                    pos_y < 0 or pos_y > args_env.ranges_y
            )
            if out_of_bounds:
                # 碰撞：增加惩罚，并将位置保持在上一步（不穿过障碍/边界）
                self.penalty_trajectory = 100
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

        # 动态更新热点（未完成任务），让奖励跟随任务分布变化
        unfinished_jobs = [job for job in self.jobs if not job.finish]
        # 更新动态任务负载：只统计未完成任务的分布
        self.grid_load_dynamic = np.zeros(self.map.n_grids)
        for job in unfinished_jobs:
            if job.lab is not None:
                self.grid_load_dynamic[job.lab] += 1
        
        if unfinished_jobs:
            self.hotspots = compute_hotspots_from_jobs(unfinished_jobs, self.n_hotspots)
            _, self.uav_hotspots = assign_hotspots_to_uav(self.uav, self.hotspots)
        else:
            # 如果没有未完成任务，热点设为地图中心（避免除零）
            self.hotspots = np.array([[args_env.ranges_x / 2, args_env.ranges_y / 2]])
            _, self.uav_hotspots = assign_hotspots_to_uav(self.uav, self.hotspots)

        # 热点奖励（提高权重，让agent更关注向任务移动）
        self.reward_hotspots = self.compute_hotspot_reward(w_hot=20.0)  # 从10提高到20

        # 覆盖奖励：鼓励 UAV 分布在任务密集区域
        self.reward_coverage = 0.0
        max_load = float(np.max(self.grid_load_dynamic)) if np.any(self.grid_load_dynamic) else 0.0
        if max_load > 0:
            coverage_sum = 0.0
            for u in range(self.n_uav):
                lab_idx = int(self.uav[u].lab) if self.uav[u].lab is not None else 0
                lab_idx = np.clip(lab_idx, 0, self.grid_load_dynamic.shape[0] - 1)
                coverage_sum += self.grid_load_dynamic[lab_idx] / max_load
            self.reward_coverage = coverage_sum / max(1, self.n_uav)

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
        # self.reward_eer = 10 * (self.reward_job_success + self.reward_task_num) / self.cost_energy
        # self.penalty_energy = self.penalty_energy / (self.n_uav * 10)
        # self.penalty_task = self.penalty_task / self.n_jobs
        # self.cost_penalty = self.penalty_trajectory + self.penalty_energy + self.penalty_task
        # reward_value = 0.5 * self.reward_eer - 0.5 * self.cost_penalty

        # 手工加权 Reward：碰撞/超时惩罚 > 按时完成任务 > 能耗/时间 + 少量角度多样性奖励
        # 说明：
        #   - self.cost_penalty：包含碰撞、超时、欠传等惩罚（越大越差）
        #   - self.reward_task_success：按时完成任务数量（越大越好）
        #   - self.cost_time：执行时间成本
        #   - self.cost_energy：能耗成本
        #   - theta：本时刻各 UAV 的方向角（来自动作），方差越大表示方向越分散
        # self.cost_penalty += self.penalty_trajectory + self.penalty_energy
        self.cost_penalty = self.penalty_trajectory + self.penalty_energy + self.penalty_task
        success_term = self.reward_job_success          # 完成的任务数
        task_term = self.reward_task_num                # 本步分配的任务数
        time_term = self.cost_time                      # 本步任务执行的增量时间
        energy_term = self.cost_energy                  # 本步能耗
        hotspots_term = self.reward_hotspots            # 向热点靠近的距离改变量

        time_norm = self._normalize_time(time_term)
        energy_norm = self._normalize_energy(energy_term)
        hotspot_norm = self._normalize_hotspot(hotspots_term)
        task_norm = task_term / max(1.0, self.n_jobs)
        success_norm = (success_term + self.reward_deadline_margin) / max(1.0, self.n_jobs)
        coverage_norm = self.reward_coverage
        timeout_norm = self.penalty_task / max(1.0, self.n_jobs)

        reward_value = (
            args_env.reward_w_hotspot * hotspot_norm
            + args_env.reward_w_task * task_norm
            + args_env.reward_w_success * success_norm
            + args_env.reward_w_coverage * coverage_norm
            - args_env.reward_w_time * time_norm
            - args_env.reward_w_energy * energy_norm
            - args_env.reward_w_collision * self.penalty_trajectory
            - args_env.reward_w_timeout * timeout_norm
        )

        # reward_set = np.array([-self.cost_time, -self.cost_energy, -self.cost_penalty])
        # # reward_set = self.reward_norm(reward_set)
        # reward_set = np.array(reward_set)
        # reward_value = round(np.sum(reward_set * self.reward_weight), 2)

        return reward_value, -self.cost_time, -self.cost_energy, -self.cost_penalty, self.reward_job_success, next_state, isterminal

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
                    t_total = t_exe * self.slot

                    self.jobs[j].finish_time[i] = self.t + t_total  # 更新任务的完成时间点
                    self.job_decision_time[j] = self.jobs[j].finish_time[i]   # 更新job下一个可决策的时间点
                    self.uav_free_time[u] = self.jobs[j].finish_time[i]       # 更新UAV的空闲时间点，与job下一个可决策时间点一致
                    # 只累加本次任务的增量执行时间，避免使用绝对时间导致 reward 巨大负数
                    self.cost_time += t_total
                    self.reward_task_num += 1

                    # 任务完成奖励值
                    if i == (self.jobs[j].n_task - 1):
                        if self.job_decision_time[j] <= self.jobs[j].deadline:
                            self.reward_job_success += 1
                            margin = (self.jobs[j].deadline - self.job_decision_time[j]) / max(1.0, self.jobs[j].deadline)
                            self.reward_deadline_margin += max(0.0, margin)
                        else:
                            overdue = (self.job_decision_time[j] - self.jobs[j].deadline) / max(1.0, self.jobs[j].deadline)
                            self.penalty_task += max(0.0, overdue)

                    # 软约束--任务传输时长超过Phase A
                    time_a = self.t_tx[j] - self.alpha * self.slot
                    self.cost_penalty += max(0, time_a)  # 欠传时长

                    # UAV 计算能耗+传输能耗
                    e_comp = self.uav[u].comp_energy_coef * self.uav[u].p_cm ** 3 * t_comp
                    e_tx = self.uav[req_u].send_energy_coef * self.uav[req_u].p_tx * t_tx
                    self.uav[u].current_energy -= e_comp
                    self.uav[req_u].current_energy -= e_tx
                    self.cost_energy += e_comp + e_tx

            # UAV 能耗惩罚
            if self.uav[u].current_energy < 0:
                self.penalty_energy += 10

        return self.get_state()

    def compute_hotspot_reward(self, w_hot=10):
        """
        prev_pos: np.ndarray, shape = (N_uav, 2)，step 前的 UAV 位置
        uavs: list[UAV]，当前 step 的 UAV 对象（已经更新了 pos）
        assigned_hotspots: np.ndarray, shape = (N_uav, 2)，每个 UAV 的目标热点
        w_hot: float，热点奖励权重
        """
        # 当前步位置
        curr_pos = np.array([[u.pos[0], u.pos[1]] for u in self.uav], dtype=float)
        pre_pos = np.array([[p[0], p[1]] for p in self.uav_pre_pos], dtype=float)

        # 距离
        d_prev = np.linalg.norm(pre_pos - self.uav_hotspots, axis=1)
        d_now = np.linalg.norm(curr_pos - self.uav_hotspots, axis=1)

        # 每个 UAV 的热点奖励：w_hot * (d_prev - d_now)
        r_u = w_hot * (d_prev - d_now)

        # 返回总热点奖励（也可以选择 np.mean(r_u)）
        return float(np.sum(r_u))

    def _normalize_time(self, value):
        denom = max(1.0, args_env.deadline_max * self.n_jobs)
        return value / denom

    def _normalize_energy(self, value):
        denom = max(1.0, args_env.max_energy_uav * self.n_uav)
        return value / denom

    def _normalize_hotspot(self, value):
        diag = math.hypot(args_env.ranges_x, args_env.ranges_y)
        denom = max(1.0, diag * self.n_uav)
        return value / denom

    def get_state(self):
        for uav_id in range(self.n_uav):
            self.uav_feats[uav_id, 0] = self.uav[uav_id].pos[0]
            self.uav_feats[uav_id, 1] = self.uav[uav_id].pos[1]
            self.uav_feats[uav_id, 2] = self.uav[uav_id].current_energy
            self.uav_feats[uav_id, 3] = np.sum(self.r_u2u[uav_id, :] > 0)
            self.uav_feats[uav_id, 4] = 1 if self.uav[uav_id].free else 0
            self.uav_feats[uav_id, 5] = sum(1 for job in self.jobs if job.uav_request[job.curr_task_id] == uav_id)
            # if self.use_potential_reward:
            #     self.uav_feats[uav_id, 6] = self.uav[uav_id].move_x
            #     self.uav_feats[uav_id, 7] = self.uav[uav_id].move_y
            # 用 UAV 所在网格的动态任务负载（反映剩余任务）
            lab_idx = int(self.uav[uav_id].lab) if self.uav[uav_id].lab is not None else 0
            self.uav_feats[uav_id, 6] = self.grid_load_dynamic[lab_idx]
            
            # 添加热点方向信息（归一化的相对位置）
            if self.uav_hotspots is not None and len(self.uav_hotspots) > uav_id:
                hotspot = self.uav_hotspots[uav_id]
                # 计算到热点的归一化方向向量（相对于地图范围）
                dx = (hotspot[0] - self.uav[uav_id].pos[0]) / max(args_env.ranges_x, 1)
                dy = (hotspot[1] - self.uav[uav_id].pos[1]) / max(args_env.ranges_y, 1)
                # 如果状态维度允许，可以添加这两个特征
                # 这里先不添加，避免改变状态维度
            if self.use_potential_reward:
                self.uav_feats[uav_id, 7] = self.uav[uav_id].move_x
                self.uav_feats[uav_id, 8] = self.uav[uav_id].move_y

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
