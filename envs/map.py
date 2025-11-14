import random

import numpy as np
from itertools import repeat, product
from random import sample

from parameter.paramEnv import *


class Map(object):
    def __init__(self):
        self.ranges_x = args_env.ranges_x
        self.ranges_y = args_env.ranges_y
        self.range_pos = args_env.range_pos
        self.h_uav = args_env.h_uav

        self.num_x = self.ranges_x // self.range_pos
        self.num_y = self.ranges_y // self.range_pos

        self.n_grids = self.num_x * self.num_y

        self.pos_set = np.empty((self.num_x * self.num_y, 2), dtype=int)
        self.get_pos_set()

        self.pos_jobs_set = np.empty((args_env.n_jobs, 3), dtype=np.float32)
        self.lab_jobs = np.zeros(args_env.n_jobs, dtype=int)

        self.r_safe = args_env.r_safe

    def get_pos_set(self):
        i = 0
        for y in range(0, self.num_y):
            range_y_min = y * self.range_pos
            range_y_max = (y + 1) * self.range_pos
            for x in range(0, self.num_x):
                range_x_min = x * self.range_pos
                range_x_max = (x + 1) * self.range_pos
                self.pos_set[i][0] = (range_x_min + range_x_max) / 2
                self.pos_set[i][1] = (range_y_min + range_y_max) / 2
                i = i + 1

    def init_uav_position(self, uav_id):
        return np.array([self.pos_set[uav_id][0], self.pos_set[uav_id][1], self.h_uav])

    def init_jobs_position(self, job_id):
        lab = random.randint(0, self.n_grids - 1)
        self.lab_jobs[job_id] = lab
        center_pos = self.pos_set[int(self.lab_jobs[job_id])]
        pos_x = random.randint(int(center_pos[0] - self.range_pos / 2), int(center_pos[0] + self.range_pos / 2))
        pos_y = random.randint(int(center_pos[1] - self.range_pos / 2), int(center_pos[1] + self.range_pos / 2))
        self.pos_jobs_set[job_id] = [int(pos_x), int(pos_y), 0]

        return self.pos_jobs_set[job_id], self.lab_jobs[job_id]

    # def get_jobs_pos_set(self, n_jobs):
    #     for g in range(n_jobs):
    #         self.lab_jobs[g] = random.randint(0, args_env.n_uav - 1)
    #         center_pos = self.pos_set[int(self.lab_jobs[g])]
    #         pos_x = random.randint(center_pos[0] - self.range_pos / 2, center_pos[0] + self.range_pos / 2)
    #         pos_y = random.randint(center_pos[1] - self.range_pos / 2, center_pos[1] + self.range_pos / 2)
    #         self.pos_jobs_set[g] = [int(pos_x), int(pos_y), 0]

    # def get_uav_position(self, uav_id, uav_vel, pre_pos, time):
    #     tra_id = random.randint(1, 2)
    #     if tra_id == 1:
    #         return self.uav_traject_1(uav_id, uav_vel, pre_pos, time)
    #     else:
    #         return self.uav_traject_2(uav_id, uav_vel, pre_pos, time)
    #
    # def get_uav_random_position(self, uav_id):
    #     y = uav_id // self.num_y
    #     x = uav_id % self.num_x
    #     range_y_min = y * self.range_pos
    #     range_y_max = (y + 1) * self.range_pos
    #     range_x_min = x * self.range_pos
    #     range_x_max = (x + 1) * self.range_pos
    #     uav_x = random.randint(range_x_min, range_x_max)
    #     uav_y = random.randint(range_y_min, range_y_max)
    #     uav_h = random.randint(50, 100)
    #     return np.array([uav_x, uav_y, uav_h])

    # def get_gts_position(self, gt_id, time):
    #     # 先设置GT静止
    #     gt_pos = self.pos_gts_set[gt_id]
    #     # 计算gt属于哪个区域
    #     x = self.pos_gts_set[gt_id][0]
    #     y = self.pos_gts_set[gt_id][1]
    #     x_lab = (x - 1e-9) // self.range_pos
    #     y_lab = (y - 1e-9) // self.range_pos
    #     lab = int(x_lab + y_lab * (self.ranges_x // self.range_pos))
    #     return gt_pos, lab

    # def uav_traject_1(self, uav_id, uav_vel, pre_pos, time):
    #     x = self.pos_set[uav_id][0]
    #     y = self.pos_set[uav_id][1]
    #     h = self.h_uav
    #     if time == 1:
    #         x = pre_pos[0]
    #         y = pre_pos[1] + uav_vel
    #         # return np.array([x, y])
    #     elif time % 8 == 2 or time % 8 == 1:
    #         x = pre_pos[0] + uav_vel
    #         y = pre_pos[1]
    #         # return np.array([x, y])
    #     elif time % 8 == 3 or time % 8 == 4:
    #         x = pre_pos[0]
    #         y = pre_pos[1] - uav_vel
    #         # return np.array([x, y])
    #     elif time % 8 == 5 or time % 8 == 6:
    #         x = pre_pos[0] - uav_vel
    #         y = pre_pos[1]
    #         # return np.array([x, y])
    #     elif time % 8 == 7 or time % 8 == 0:
    #         x = pre_pos[0]
    #         y = pre_pos[1] + uav_vel
    #         # return np.array([x, y])
    #     return np.array([x, y, h])
    #
    # def uav_traject_2(self, uav_id, uav_vel, pre_pos, time):
    #     x = self.pos_set[uav_id][0]
    #     y = self.pos_set[uav_id][1]
    #     h = self.h_uav
    #     if time == 1 or time == 2:
    #         x = pre_pos[0]
    #         y = pre_pos[1] + uav_vel
    #         # return np.array([x, y])
    #     elif time % 16 == 1 or time % 16 == 2 or time % 16 == 3 or time % 16 == 4:
    #         x = pre_pos[0] + uav_vel
    #         y = pre_pos[1]
    #         # return np.array([x, y])
    #     elif time % 16 == 5 or time % 16 == 6 or time % 16 == 7 or time % 16 == 8:
    #         x = pre_pos[0]
    #         y = pre_pos[1] - uav_vel
    #         # return np.array([x, y])
    #     elif time % 16 == 9 or time % 16 == 10 or time % 16 == 11 or time % 16 == 12:
    #         x = pre_pos[0] - uav_vel
    #         y = pre_pos[1]
    #         # return np.array([x, y])
    #     elif time % 16 == 13 or time % 16 == 14 or time % 16 == 15 or time % 16 == 0:
    #         x = pre_pos[0]
    #         y = pre_pos[1] + uav_vel
    #         # return np.array([x, y])
    #     return np.array([x, y, h])




