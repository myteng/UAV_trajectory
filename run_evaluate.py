import random

import numpy as np
from matplotlib import pyplot as plt

from algos.DDPG.evaluate_ddpg import run_evaluate_ddpg
from algos.PPO.evaluate_ppo import run_evaluate_ppo
from parameter.paramAgent import args_agent
from parameter.paramEnv import args_env
from utils.common import set_rand_seed
from utils.normal import Normalize, RewardScaling
import os.path as osp

from utils.save_txt import save_result

if __name__ == '__main__':
    # 训练参数
    epoch = 600
    time = 5
    args_env.datasets = True

    args_env.scene = 'suburban'
    # args_env.scene = 'dense-urban'
    log_dir_result = "/Users/tengman/Documents/Python/UAVTrajectory/pythonProject1103/results_data/"
    save = save_result(log_dir_result)

    log_dir_bp_0 = "/Users/tengman/Documents/Python/UAVTrajectory/pythonProject1103/log_bp_data/"
    seed_set = [128]

    ranges_1 = [300, 300]
    ranges_2 = [500, 500]
    uav_set = [9, 25]
    job_set = [10, 20, 30, 40, 50, 60]
    p = 0

    for g in range(3):
        if p == 0:
            args_env.ranges_x = ranges_1[0]
            args_env.ranges_y = ranges_1[1]
        else:
            args_env.ranges_x = ranges_2[0]
            args_env.ranges_y = ranges_2[1]

        args_env.n_uav = uav_set[p]
        args_env.n_jobs = job_set[g]

        print(f"----------------------------------------n_nua={args_env.n_uav}, n_gts={args_env.n_jobs}-----------------------------------------------------")

        rewards_ppo_avg = []
        # rewards_ppo_time = []
        # rewards_ppo_load = []
        # rewards_ppo_success = []
        # rewards_ppo_energy = []
        # rewards_ppo_rate = []
        i = 0
        for i in range(len(seed_set)):
            args_agent.seed = seed_set[i]
            seed = args_agent.seed

            # Evaluate PPO.
            set_rand_seed(seed)
            reward_avg_evaluate_ppo = run_evaluate_ppo(epoch, time, log_dir_bp_0, args_agent.seed)
            # ppo_reward_rate = np.round(rewards_success_ppo / (-rewards_energy_ppo), 2)
            rewards_ppo_avg.append(reward_avg_evaluate_ppo)
            # rewards_ppo_time.append(rewards_time_ppo)
            # rewards_ppo_load.append(rewards_load_ppo)
            # rewards_ppo_success.append(rewards_success_ppo)
            # rewards_ppo_energy.append(rewards_energy_ppo)
            # rewards_ppo_rate.append(ppo_reward_rate)

        rewards_ddpg_avg = []
        # rewards_ddpg_time = []
        # rewards_ddpg_load = []
        # rewards_ddpg_success = []
        # rewards_ddpg_energy = []
        # rewards_ddpg_rate = []
        i = 0
        for i in range(len(seed_set)):
            args_agent.seed = seed_set[i]
            seed = args_agent.seed

            # Evaluate DDPG.
            set_rand_seed(seed)
            reward_avg_evaluate_ddpg = run_evaluate_ddpg(epoch, time, log_dir_bp_0, args_agent.seed)
            # ddpg_reward_rate = np.round(rewards_success_ddpg / (-rewards_energy_ddpg), 2)
            rewards_ddpg_avg.append(reward_avg_evaluate_ddpg)
            # rewards_ddpg_time.append(rewards_time_ddpg)
            # rewards_ddpg_load.append(rewards_load_ddpg)
            # rewards_ddpg_success.append(rewards_success_ddpg)
            # rewards_ddpg_energy.append(rewards_energy_ddpg)
            # rewards_ddpg_rate.append(ddpg_reward_rate)

        rewards_ppo_no_avg = []
        # rewards_ppo_no_time = []
        # rewards_ppo_no_load = []
        # rewards_ppo_no_success = []
        # rewards_ppo_no_energy = []
        # rewards_ppo_no_rate = []
        i = 0
        for i in range(len(seed_set)):
            args_agent.seed = seed_set[i]
            seed = args_agent.seed
            args_env.use_potential_reward = False

            # Evaluate PPO.
            set_rand_seed(seed)
            reward_avg_evaluate_ppo_no = run_evaluate_ppo(epoch, time, log_dir_bp_0, args_agent.seed)
            # ppo_no_reward_rate = np.round(rewards_success_ppo_no / (-rewards_energy_ppo_no), 2)
            rewards_ppo_no_avg.append(reward_avg_evaluate_ppo_no)
            # rewards_ppo_no_time.append(rewards_time_ppo_no)
            # rewards_ppo_no_load.append(rewards_load_ppo_no)
            # rewards_ppo_no_success.append(rewards_success_ppo_no)
            # rewards_ppo_no_energy.append(rewards_energy_ppo_no)
            # rewards_ppo_no_rate.append(ppo_no_reward_rate)

        rewards_ddpg_no_avg = []
        # rewards_ddpg_no_time = []
        # rewards_ddpg_no_load = []
        # rewards_ddpg_no_success = []
        # rewards_ddpg_no_energy = []
        # rewards_ddpg_no_rate = []
        i = 0
        for i in range(len(seed_set)):
            args_agent.seed = seed_set[i]
            seed = args_agent.seed

            # Evaluate DDPG.
            set_rand_seed(seed)
            reward_avg_evaluate_ddpg_no = run_evaluate_ddpg(epoch, time, log_dir_bp_0, args_agent.seed)
            # ddpg_no_reward_rate = np.round(rewards_success_ddpg_no / (-rewards_energy_ddpg_no), 2)
            rewards_ddpg_no_avg.append(reward_avg_evaluate_ddpg_no)
            # rewards_ddpg_no_time.append(rewards_time_ddpg_no)
            # rewards_ddpg_no_load.append(rewards_load_ddpg_no)
            # rewards_ddpg_no_success.append(rewards_success_ddpg_no)
            # rewards_ddpg_no_energy.append(rewards_energy_ddpg_no)
            # rewards_ddpg_no_rate.append(ddpg_no_reward_rate)

        print("++++++++++++++++++ Average Rewards ++++++++++++++++++++++++")
        ppo_avg = np.round(sum(rewards_ppo_avg) / (i + 1), 2)
        # ppo_time = np.round(sum(rewards_ppo_time) / (i + 1), 2)
        # ppo_load = np.round(sum(rewards_ppo_load) / (i + 1), 2)
        # ppo_success = np.round(sum(rewards_ppo_success) / (i + 1), 2)
        # ppo_energy = np.round(sum(rewards_ppo_energy) / (i + 1), 2)
        # ppo_rate = np.round(sum(rewards_ppo_rate) / (i + 1), 2)
        save.save_reward_txt("PPO", args_env.n_uav, args_env.n_jobs, ppo_avg)
        print(f"PPO algorithm: Average Reward, {ppo_avg}")

        ddpg_avg = np.round(sum(rewards_ddpg_avg) / (i + 1), 2)
        save.save_reward_txt("PPO", args_env.n_uav, args_env.n_jobs, ddpg_avg)
        print(f"DDPG algorithm: Average Reward, {ddpg_avg}")

        ppo_no_avg = np.round(sum(rewards_ppo_no_avg) / (i + 1), 2)
        save.save_reward_txt("PPO_no_potential", args_env.n_uav, args_env.n_jobs, ppo_no_avg)
        print(f"PPO_no_potential algorithm: Average Reward, {ppo_no_avg}")

        ddpg_no_avg = np.round(sum(rewards_ddpg_no_avg) / (i + 1), 2)
        save.save_reward_txt("PPO_no_potential", args_env.n_uav, args_env.n_jobs, ddpg_no_avg)
        print(f"DDPG_no_potential algorithm: Average Reward, {ddpg_no_avg}")

