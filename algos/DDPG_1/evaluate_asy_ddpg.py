import datetime

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from matplotlib import pyplot as plt
from matplot.uav_trajectory_plot import plot_uav_and_jobs

from algos.DDPG_1.asy_ddpg_agent import asyDDPGAgent
from envs.env_1 import Environment_1
from parameter.paramAgent import args_agent
from parameter.paramEnv import args_env
from utils.common import set_rand_seed
from utils.normal import Normalize, RewardScaling
from utils.save_txt import TXT_FILE
import os.path as osp


def evaluate_policy_asy_ddpg(env, agent, times, state_norm):
    rewards = 0
    r_time_sum = 0
    r_energy_sum = 0
    r_penalty_sum = 0
    r_task_success_sum = 0
    save_txt = TXT_FILE()
    save_txt.clear_pos()

    for t in range(times):
        episode_reward = []
        episode_r_time = []
        episode_r_energy = []
        episode_r_penalty = []
        r_task_success = 0

        state, reward, r_time, r_energy, r_penalty = env.reset()

        save_txt.save_job_position(env.jobs, env.n_jobs)

        episode_reward.append(reward)
        episode_r_time.append(r_time)
        episode_r_energy.append(r_energy)
        episode_r_penalty.append(r_penalty)

        for step in range(1000):
            save_txt.save_uav_position(t, step, env.uav, env.n_uav)
            # 归一化当前状态
            state = state.astype(np.float32)[None, ...]
            if args_agent.use_state_normal:
                state = state_norm(state)

            # 动作
            action = agent.evaluate(state)
            print(f"action = {action}")

            # 与环境交互
            reward, r_time, r_energy, r_penalty, r_task_success, next_state, is_terminal = env.step_asy(action)
            print(f"step: {step}, reward: {reward}, reward_task_success: {r_task_success}")

            episode_reward.append(reward)
            episode_r_time.append(r_time)
            episode_r_energy.append(r_energy)
            episode_r_penalty.append(r_penalty)

            if is_terminal:
                break
            state = next_state

        rewards_sum = sum(episode_reward)
        rewards_avg = np.round(rewards_sum / len(episode_reward), 2)
        r_time_sum += sum(episode_r_time)
        r_energy_sum += sum(episode_r_energy)
        r_penalty_sum += sum(episode_r_penalty)
        r_task_success_sum += r_task_success

        rewards += rewards_sum

    return rewards / times, r_time_sum / times, r_energy_sum / times, r_penalty_sum / times, r_task_success_sum / times


def run_evaluate_asy_ddpg(epoch, time, log_dir_bp, seed):
    set_rand_seed(seed)
    args_agent.agent_type = 'ddpg'

    # Environment
    env = Environment_1()
    state_size = (env.n_uav, env.n_feature)
    action_size = env.n_uav

    # Agent
    agent_asy_ddpg = asyDDPGAgent(state_size, action_size)

    # Load Model
    if args_env.use_potential_reward:
        path_ddpg = osp.join(log_dir_bp + "ddpg_1/", f'checkpoint_epoch{epoch - 1}.pt')
    else:
        path_ddpg = osp.join(log_dir_bp + "ddpg_no_potential_1/", f'checkpoint_epoch{epoch - 1}.pt')
    agent_asy_ddpg.load_checkpoint(path_ddpg)

    # Normalizer
    state_norm = Normalize(shape=state_size)
    # reward_norm = Normalize(shape=4)

    reward_avg_evaluate_ddpg, reward_avg_time, reward_avg_energy, reward_avg_penalty, reward_task_success = evaluate_policy_asy_ddpg(env, agent_asy_ddpg, time, state_norm)

    reward_avg_evaluate_ddpg = np.round(reward_avg_evaluate_ddpg, 2)
    reward_avg_time = np.round(reward_avg_time, 2)
    reward_avg_energy = np.round(reward_avg_energy, 2)
    reward_avg_penalty = np.round(reward_avg_penalty, 2)
    reward_avg_task_success = np.round(reward_task_success, 2)

    return reward_avg_evaluate_ddpg, reward_avg_time, reward_avg_energy, reward_avg_penalty, reward_avg_task_success


if __name__ == '__main__':
    # 训练参数
    epoch = 200
    time = 1
    args_env.datasets = True

    args_env.scene = 'suburban'
    # args_env.scene = 'dense-urban'
    log_dir_result = "/Users/tengman/Documents/Python/UAVTrajectory/pythonProject1103/results_data/"

    log_dir_bp_0 = "/Users/tengman/Documents/Python/UAVTrajectory/pythonProject1103/log_bp_data/"
    seed_set = [0]

    ranges_1 = [300, 300]
    ranges_2 = [500, 500]
    uav_set = [9, 25]
    job_set = [50]
    p = 0

    for g in range(len(job_set)):
        if p == 0:
            args_env.ranges_x = ranges_1[0]
            args_env.ranges_y = ranges_1[1]
        else:
            args_env.ranges_x = ranges_2[0]
            args_env.ranges_y = ranges_2[1]

        args_env.n_uav = uav_set[p]
        args_env.n_jobs = job_set[g]

        print(f"----------------------------------------n_nua={args_env.n_uav}, n_gts={args_env.n_jobs}-----------------------------------------------------")
        rewards_ddpg_1_no_avg = []
        rewards_ddpg_1_no_time = []
        rewards_ddpg_1_no_energy = []
        rewards_ddpg_1_no_penalty = []

        i = 0
        for i in range(len(seed_set)):
            print(f"*****************seed = {seed_set[i]}*****************")
            args_env.use_potential_reward = False

            # Evaluate DDPG_1.
            args_agent.seed = seed_set[i]
            seed = args_agent.seed
            set_rand_seed(seed)
            reward_avg_evaluate_ddpg_1_no, rewards_time_ddpg_1_no, rewards_energy_ddpg_1_no, rewards_penalty_ddpg_1_no, rewards_task_success_ddpg_1_no = run_evaluate_asy_ddpg(epoch, time, log_dir_bp_0, args_agent.seed)
            # ddpg_1_no_reward_rate = np.round(rewards_success_ddpg_no / (-rewards_energy_ddpg_no), 2)
            rewards_ddpg_1_no_avg.append(reward_avg_evaluate_ddpg_1_no)
            rewards_ddpg_1_no_time.append(rewards_time_ddpg_1_no)
            rewards_ddpg_1_no_energy.append(rewards_energy_ddpg_1_no)
            rewards_ddpg_1_no_penalty.append(rewards_penalty_ddpg_1_no)
            print(f"DDPG_1_no_potential: Reward = {reward_avg_evaluate_ddpg_1_no}, Time = {rewards_time_ddpg_1_no}, Energy = {rewards_energy_ddpg_1_no}, penalty = {rewards_penalty_ddpg_1_no}, task_success = {rewards_task_success_ddpg_1_no} ")

        plot_uav_and_jobs(uav_file="results_data/uav_pos.txt", job_file="results_data/jobs_pos.txt", save_path="results_data/uav_jobs_plot_1.png")