import random
from typing import Tuple, List

from envs.map import Map
from parameter.paramEnv import *


class UAV(object):
    def __init__(self, uav_id):
        self.id = uav_id  # ID
        self.pos = np.empty(3, dtype=np.float32)  # Position (x, y)
        self.lab = None  # UAV所在的grid_id
        self.vels = args_env.vels_uav  # Velocity (V_u)
        self.free = True  # UAV当前时刻是否空闲

        self.p_tx = random.randint(args_env.p_tx_uav_min, args_env.p_tx_uav_max)   # Transmission Power
        self.p_cm = random.randint(args_env.p_cm_uav_min, args_env.p_cm_uav_max)   # Computing Power (C_u)

        self.current_energy = args_env.max_energy_uav  # E_u(t)
        self.fly_energy_coef = args_env.fly_energy_coef  # η_fly
        self.comp_energy_coef = args_env.comp_energy_coef  # η_comp
        self.hov_energy_coef = args_env.hov_energy_coef  # η_recv
        self.send_energy_coef = args_env.send_energy_coef  # η_send
        self.turn_energy_coef = args_env.turn_energy_coef  # η_recv

        self.max_energy = args_env.max_energy_uav

        # 表示UAV上一个时隙的移动轨迹
        self.move_x = None
        self.move_y = None

        # 每个UAV最多执行一个任务，这个特性可以不加
        # self.max_parallel_tasks = args_env.max_parallel_tasks  # K_u
        # self.current_parallel_tasks = 0  # K_u(t)

        self.map = Map()

    def reset_uav(self, uav_id):
        # initial positions of UAVs
        self.pos = self.map.init_uav_position(uav_id)
        self.lab = uav_id
        self.current_energy = args_env.max_energy_uav
        self.max_energy = args_env.max_energy_uav
        self.free = True  # UAV当前时刻是否空闲
        self.move_x = 0
        self.move_y = 0
        # self.current_parallel_tasks = 0

    def step_uav(self, new_position, energy):
        # Step Position
        self.pos = new_position

        # Step Energy
        self.current_energy = self.current_energy - energy


# class UAV:
#     def __init__(
#         self,
#         uav_id: int,
#         position: Tuple[float, float],
#         compute_capacity: float,
#         energy: float,
#         fly_energy_coef: float,
#         comp_energy_coef: float,
#         recv_energy_coef: float,
#         send_energy_coef: float,
#         max_tasks: int,
#     ):
#         self.uav_id = uav_id
#         self.position = position  # (x, y)
#         self.compute_capacity = compute_capacity  # C_u
#         self.energy = energy  # E_u(t)
#         self.fly_energy_coef = fly_energy_coef  # η_fly
#         self.comp_energy_coef = comp_energy_coef  # η_comp
#         self.recv_energy_coef = recv_energy_coef  # η_recv
#         self.send_energy_coef = send_energy_coef  # η_send
#         self.max_tasks = max_tasks  # K_u
#         self.assigned_tasks: List[int] = []  # 任务编号列表
#
#     def update_position(self, new_position: Tuple[float, float]):
#         self.position = new_position
#
#     def can_assign_task(self) -> bool:
#         return len(self.assigned_tasks) < self.max_tasks
#
#     def assign_task(self, task_id: int):
#         if self.can_assign_task():
#             self.assigned_tasks.append(task_id)
#
#     def reset_tasks(self):
#         self.assigned_tasks = []
