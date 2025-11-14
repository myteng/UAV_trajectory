import datetime

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from matplotlib import pyplot as plt

from algos.DDPG.ddpg_agent import DDPGAgent
from envs.env import Environment
from parameter.paramAgent import args_agent
from parameter.paramEnv import args_env
from utils.common import set_rand_seed
from utils.normal import Normalize, RewardScaling
import os.path as osp


def evaluate_policy_ddpg(env, agent, times, state_norm):
    rewards = 0

    for t in range(times):
        state = env.reset()

        episode_reward = []

        for step in range(1000):
            # 归一化当前状态
            state = state.astype(np.float32)[None, ...]
            if args_agent.use_state_normal:
                state = state_norm(state)

            # 动作
            action = agent.evaluate(state)
            # print(f"action = {action}")

            # 与环境交互
            reward, next_state, is_terminal = env.step_evaluate(action)
            # print(f"reward: {reward}, reward_scal: {reward_scal(reward)}")

            episode_reward.append(reward)

            if is_terminal:
                break
            state = next_state

        rewards_sum = sum(episode_reward)
        rewards_avg = np.round(rewards_sum / len(episode_reward), 2)
        # print(f"len(episode_reward) = {len(episode_reward)}")

        rewards += rewards_sum

    return rewards / times


def run_evaluate_ddpg(epoch, time, log_dir_bp, seed):
    set_rand_seed(seed)
    args_agent.agent_type = 'ddpg'

    # Environment
    env = Environment()
    state_size = (env.n_uav, env.n_feature)
    action_size = env.n_uav + 1

    # Agent
    agent_ddpg = DDPGAgent(state_size, action_size)

    # Load Model
    if args_env.use_potential_reward:
        path_ddpg = osp.join(log_dir_bp + "ddpg/", f'checkpoint_epoch{epoch - 1}.pt')
    else:
        path_ddpg = osp.join(log_dir_bp + "ddpg_no_potential/", f'checkpoint_epoch{epoch - 1}.pt')
    agent_ddpg.load_checkpoint(path_ddpg)

    # Normalizer
    state_norm = Normalize(shape=state_size)
    # reward_norm = Normalize(shape=4)

    reward_avg_evaluate_ddpg = evaluate_policy_ddpg(env, agent_ddpg, time, state_norm)

    reward_avg_evaluate_ddpg = np.round(reward_avg_evaluate_ddpg, 2)
    # reward_avg_time = np.round(-reward_avg_time, 2)
    # reward_avg_load = np.round(-reward_avg_load, 2)
    # reward_avg_success = np.round(reward_avg_success, 2)
    # reward_avg_energy = np.round(-reward_avg_energy, 2)
    # reward_avg_rate = np.round(reward_avg_success * 100 / (reward_avg_time + reward_avg_energy + reward_avg_energy), 2)

    if args_env.use_potential_reward:
        print(f"DDPG algorithm: average reward evaluate, {reward_avg_evaluate_ddpg}")
    else:
        print(f"DDPG_no_potential algorithm: average reward evaluate, {reward_avg_evaluate_ddpg}")

    # print(
    #     f"DDPG algorithm: average reward evaluate, {reward_avg_evaluate_ppo}, execution time, {reward_avg_time}, load, {reward_avg_load}, success, {reward_avg_success}, energy, {reward_avg_energy}, rate, {reward_avg_rate}")

    return reward_avg_evaluate_ddpg


