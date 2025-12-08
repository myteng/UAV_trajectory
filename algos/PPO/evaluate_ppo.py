import datetime

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from matplotlib import pyplot as plt

from algos.PPO.ppo_agent import PPOAgent
from envs.env import Environment
from parameter.paramAgent import args_agent
from parameter.paramEnv import args_env
from utils.common import set_rand_seed
from utils.normal import Normalize, RewardScaling
import os.path as osp


def evaluate_policy_ppo(env, agent, times, state_norm):
    rewards = 0
    r_time_sum = 0
    r_energy_sum = 0
    r_penalty_sum = 0
    r_task_success_sum = 0

    for t in range(times):
        state = env.reset()

        episode_reward = []
        episode_r_time = []
        episode_r_energy = []
        episode_r_penalty = []
        r_task_success = 0

        for step in range(1000):
            # 归一化当前状态
            state = state.astype(np.float32)[None, ...]
            if args_agent.use_state_normal:
                state = state_norm(state)

            # 动作
            action = agent.evaluate(state)
            # print(f"action = {action}")

            # 与环境交互
            reward, r_time, r_energy, r_penalty, r_task_success, next_state, is_terminal = env.step_evaluate(action)
            # print(f"reward: {reward}, reward_scal: {reward_scal(reward)}")

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
        # print(f"rewards_sum = {rewards_sum}, r_time_sum = {r_time_sum}, r_energy_sum = {r_energy_sum}, r_penalty_sum = {r_penalty_sum}")

        rewards += rewards_sum

    return rewards / times, r_time_sum / times, r_energy_sum / times, r_penalty_sum / times, r_task_success_sum / times


def run_evaluate_ppo(epoch, time, log_dir_bp, seed):
    set_rand_seed(seed)
    args_agent.agent_type = 'ppo'

    # Environment
    env = Environment()
    state_size = (env.n_uav, env.n_feature)
    action_size = env.n_uav

    # Agent
    agent_ppo = PPOAgent(state_size, action_size)

    # Load Model
    if args_env.use_potential_reward:
        path_ppo = osp.join(log_dir_bp + "ppo/", f'checkpoint_epoch{epoch - 1}.pt')
    else:
        path_ppo = osp.join(log_dir_bp + "ppo_no_potential/", f'checkpoint_epoch{epoch - 1}.pt')
    agent_ppo.load_checkpoint(path_ppo)

    # Normalizer
    state_norm = Normalize(shape=state_size)
    # reward_norm = Normalize(shape=4)

    reward_avg_evaluate_ppo = evaluate_policy_ppo(env, agent_ppo, time, state_norm)

    reward_avg_evaluate_ppo, reward_avg_time, reward_avg_energy, reward_avg_penalty, reward_task_success = np.round(reward_avg_evaluate_ppo, 2)
    reward_avg_time = np.round(reward_avg_time, 2)
    reward_avg_energy = np.round(reward_avg_energy, 2)
    reward_avg_penalty = np.round(reward_avg_penalty, 2)
    reward_avg_task_success = np.round(reward_task_success, 2)

    # if args_env.use_potential_reward:
    #     print(f"PPO algorithm: average reward evaluate, {reward_avg_evaluate_ppo}")
    # else:
    #     print(f"PPO_no_potential algorithm: average reward evaluate, {reward_avg_evaluate_ppo}")

    return reward_avg_evaluate_ppo, reward_avg_time, reward_avg_energy, reward_avg_penalty, reward_avg_task_success


