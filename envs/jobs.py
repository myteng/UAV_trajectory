import random
from typing import Tuple, List

import numpy as np
from parameter.paramEnv import *
from envs.map import Map


class Job(object):
    def __init__(self, job_id):
        self.id = job_id
        self.pos = np.empty(3, dtype=np.float32)
        self.lab = None

        # ---- 作业级：到达与截止（绝对时间）----
        self.arrival_time = args_env.job_arr_time
        self.deadline = random.randint(args_env.deadline_min, args_env.deadline_max)  # d_j
        self.deadline_max = random.randint(args_env.deadline_min, args_env.deadline_max)  # d_j
        self.status = True  # job的状态，True表示当前时刻可以执行，False表示当前时刻不能执行
        self.finish = False  # Job是否执行完
        self.start = False  # Job是否开始执行
        self.timeout_flag = False

        # ---- 任务数与静态需求 ----
        self.n_task = random.randint(args_env.n_task_min, args_env.n_task_max)  # The Number of Tasks
        self.curr_task_id = None
        self.uav_request = [-1 for _ in range(self.n_task)]   # task的发送UAV
        self.uav_offload = [-1 for _ in range(self.n_task)]   # task的卸载UAV
        self.workload = [random.randint(args_env.workload_min, args_env.workload_max) for _ in range(self.n_task)]
        self.data_size = [random.randint(args_env.data_size_min, args_env.data_size_max) for _ in range(self.n_task)]

        # ---- 任务级：运行状态 ----
        self.assigned_uav = [-1 for _ in range(self.n_task)]  # task是否分配
        self.start_time = [-1 for _ in range(self.n_task)]  # 实际开始执行时间
        self.finish_time = [-1 for _ in range(self.n_task)]  # 实际完成时间

        self.map = Map()

    def reset_jobs(self, gt_id):
        # initial positions of GT
        self.pos, self.lab = self.map.init_jobs_position(gt_id)
        self.deadline = self.deadline_max
        self.status = True
        self.timeout_flag = False

        self.uav_request = [-1 for _ in range(self.n_task)]  # task的发送UAV
        self.uav_offload = [-1 for _ in range(self.n_task)]  # task的卸载UAV
        # self.uav_request[0] = self.lab

        self.curr_task_id = 0
        self.start_time = [-1 for _ in range(self.n_task)]  # 实际开始执行时间
        self.finish_time = [-1 for _ in range(self.n_task)]  # 实际完成时
        self.start_time[0] = self.arrival_time  # 第一个任务实际开始执行时间为0

    def step_gts(self, assigned_uav_id):
        # offloading
        self.assigned_uav = assigned_uav_id

        # Update grid_load


# class Task:
#     def __init__(
#         self,
#         task_id: int,
#         location: Tuple[float, float],
#         workload: float,
#         deadline: float,
#         data_size: float,
#         value: float,
#         candidate_uavs: List[int]
#     ):
#         self.task_id = task_id
#         self.location = location  # (x, y)
#         self.workload = workload  # w_j
#         self.deadline = deadline  # d_j
#         self.data_size = data_size  # s_j
#         self.value = value  # v_j
#         self.candidate_uavs = candidate_uavs  # 可通信UAV编号列表
#         self.assigned_uav: int = -1  # 未分配时为 -1
#         self.start_time: float = None
#         self.end_time: float = None
#
#     def is_assigned(self) -> bool:
#         return self.assigned_uav != -1
#
#     def assign_to(self, uav_id: int):
#         self.assigned_uav = uav_id
