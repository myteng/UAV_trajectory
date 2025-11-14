import numpy as np

from envs.is_intersect_building_3d import line_segment_intersects_building
from envs.obstacle import ObstacleMap
from parameter.paramEnv import *


class Channel(object):
    # chan_params = {
    #     'suburban': (4.88, 0.43, 0.1, 21),
    #     'urban': (9.61, 0.16, 1, 20),
    #     'dense-urban': (12.08, 0.11, 1.6, 23),
    #     'high-rise-urban': (27.23, 0.08, 2.3, 34)
    # }
    chan_params = {
        'suburban': (0.8, 0.2, 0.1, 21),
        'urban': (0.5, 0.5, 1, 20),
        'dense-urban': (0.4, 0.6, 1.6, 23),
        'high-rise-urban': (0.2, 0.8, 2.3, 34)
    }

    def __init__(self):
        self.bw = args_env.bw
        self.p_tx_gts = args_env.p_tx_gts
        self.no = 1e-3 * np.power(10, args_env.no / 10)  # dbm转换Watt
        self.fc = args_env.fc
        self.scene = args_env.scene
        self.max_dis = args_env.max_distance
        self.dataset = args_env.datasets  # Use the building dataset
        if args_env.scene == 'dense-urban':
            self.path = 'Building_Outline_2015_with_xy_du_2.csv'
        else:
            self.path = 'Building_Outline_2015_with_xy_2.csv'

        self.obs_map = ObstacleMap()

        params = self.chan_params[self.scene]
        self.p_los, self.p_nlos = params[0], params[1]  # Constants for computing p_los
        self.eta_los, self.eta_nlos = params[2], params[3]  # Path loss exponents (LoS/NLoS)

    # Get Gain
    def get_gain(self, pos_1, pos_2):
        """Estimates the channel gain from horizontal distance."""
        # Estimate probability of LoS link emergence.
        # p_los = 0.5
        # Get direct link distance.
        d = np.linalg.norm(pos_1 - pos_2)
        # Compute free space path loss (FSPL).
        fspl = (4 * np.pi * self.fc * d / 3e8) ** 2
        if self.dataset:
            # 遮挡建筑物数量和信息
            # print("--------------------------Use Datasets!---------------------------------")
            intersects_building_num, building_info = line_segment_intersects_building(self.path, np.array(pos_1), np.array(pos_2))
            if intersects_building_num > 0:
                # Path loss is the NLoS cases.
                pl = fspl * 10 ** (self.eta_nlos / 20)
            else:
                # Path loss is the LoS cases.
                pl = fspl * 10 ** (self.eta_los / 20)
        else:
            is_blocked = self.obs_map.is_path_blocked_3d(pos_1, pos_2)
            if is_blocked:
                pl = fspl * 10 ** (self.eta_nlos / 20)
            else:
                pl = fspl * 10 ** (self.eta_nlos / 20)
            # # Path loss is the weighted average of LoS and NLoS cases.
            # pl = self.p_los * fspl * 10 ** (self.eta_los / 20) + self.p_nlos * fspl * 10 ** (self.eta_nlos / 20)

        # Get channel gain
        gain = 1 / pl
        return gain

    # Get U2U Rate
    def get_u2u_rate(self, uav_1, uav_2):
        d = np.linalg.norm(uav_1.pos - uav_2.pos)
        if d >= self.max_dis:
            rate = 0
        else:
            gain = self.get_gain(uav_1.pos, uav_2.pos)

            # Get SNR
            p_tx = 1e-3 * np.power(10, uav_1.p_tx / 10)  # Tx power (Watt)  10dbm转换为Watt
            snr = p_tx * gain / (self.no * self.bw)

            # link rate (Mbps)
            rate = self.bw * np.log2(1 + snr) * 1e-6

        return rate

    def get_j2u_rate(self, job, uav):
        gain = self.get_gain(job.pos, uav.pos)
        # Get SNR
        p_tx = 1e-3 * np.power(10, self.p_tx_gts / 10)  # Tx power (Watt)  10dbm转换为Watt
        snr = p_tx * gain / (self.no * self.bw)
        # link rate (Mbps)
        rate = self.bw * np.log2(1 + snr) * 1e-6
        return rate



