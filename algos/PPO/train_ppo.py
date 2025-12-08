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


def training_ppo(epochs, log_dir_log, log_dir_bp):
    # Set random seeds.
    seed = args_agent.seed
    set_rand_seed(seed)
    args_agent.agent_type = 'PPO'

    env = Environment()

    state_size = (env.n_uav, env.n_feature)
    action_size = env.n_uav

    agent = PPOAgent(state_size, action_size)

    # Normalizer
    # reward_scal = RewardScaling(shape=1, gamma=0.9)
    state_norm = Normalize(shape=state_size)

    log_dir_log = log_dir_log + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "-PPO"
    summary_writer = SummaryWriter(log_dir_log)

    rewards_set = []  # 存储每回合的奖励
    epoch_set = []

    # start_epoch = agent.load_checkpoint(log_dir_bp + "checkpoint_epoch299.pt")

    # 循环训练
    for e in range(epochs):
        print("++++++++++++++++++++++++++++++++++++++++++++++++++第", e, "次训练+++++++++++++++++++++++++++++++++++++++++++")
        state = env.reset()
        # reward_scal.reset()

        states = []  # 存储状态
        actions = []  # 存储动作
        rewards = []  # 存储奖励
        is_terminals = []  # 存储是否终止
        reward_sum = []

        for step in range(3000):
            print("-------------------------PPO：第", e, "次训练，第", step, "步测试-----------------------------")
            # 归一化当前状态
            state = state.astype(np.float32)[None, ...]
            state = state_norm(state)

            # 动作
            action = agent.get_action(state)
            print(f"action = {action}")

            # 与环境交互
            reward, next_state, is_terminal = env.step(action)
            print(f"reward = {reward}")
            print(f"is_terminal = {is_terminal}")
            # reward = reward_scal(reward)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            is_terminals.append(is_terminal)

            reward_sum.append(reward)

            if is_terminal:
                actor_loss, actor_grads, critic_loss, critic_grads = agent.update(states, actions, rewards, is_terminals)
                states, actions, rewards, is_terminals = [], [], [], []
                break  # 结束回合

            state = next_state

        rewards_avg = np.round(sum(reward_sum) / len(reward_sum), 2)
        summary_writer.add_scalar("rewards", rewards_avg, e)

        rewards_set.append(rewards_avg)
        epoch_set.append(e)

        # Breakpoint
        if (e + 1) % 200 == 0:
            # Save Breakpoint
            save_path = osp.join(log_dir_bp, f'checkpoint_epoch{e}.pt')
            agent.save_checkpoint(save_path, e)

    summary_writer.close()
    return epoch_set, rewards_set


if __name__ == '__main__':

    ranges_1 = [300, 300]
    args_env.ranges_x = ranges_1[0]
    args_env.ranges_y = ranges_1[1]
    args_env.n_uav = 9
    args_env.n_jobs = 30
    args_env.use_potential_reward = True

    if args_env.use_potential_reward:
        log_dir_log = "/Users/tengman/Documents/Python/UAVTrajectory/pythonProject1103/log_data/ppo/"
        log_dir_bp = "/Users/tengman/Documents/Python/UAVTrajectory/pythonProject1103/log_bp_data/ppo/"
    else:
        log_dir_log = "/Users/tengman/Documents/Python/UAVTrajectory/pythonProject1103/log_data/ppo_no_potential/"
        log_dir_bp = "/Users/tengman/Documents/Python/UAVTrajectory/pythonProject1103/log_bp_data/ppo_no_potential/"

    epoch_set, rewards_set = training_ppo(1000, log_dir_log, log_dir_bp)
    plt.figure(figsize=(8, 6))
    plt.plot(epoch_set, rewards_set, label='PPO', color='red')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    # 保存图表
    plt.savefig('/Users/tengman/Documents/Python/UAVTrajectory/pythonProject1103/results_fig/train_ppo_results/PPO_Reward_Contrast.png')
