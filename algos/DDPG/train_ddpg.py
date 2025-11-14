import datetime
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
import os.path as osp

from algos.DDPG.ddpg_agent import DDPGAgent   # ← 替换为你的DDPGAgent路径
from envs.env import Environment
from parameter.paramAgent import args_agent
from parameter.paramEnv import args_env
from utils.common import set_rand_seed
from utils.normal import Normalize, RewardScaling


def training_ddpg(epochs, log_dir_log, log_dir_bp):
    # Set random seeds.
    seed = args_agent.seed
    set_rand_seed(seed)
    args_agent.agent_type = 'DDPG'

    env = Environment()

    state_size = (env.n_uav, env.n_feature)
    action_size = env.n_uav + 1

    agent = DDPGAgent(state_size, action_size)

    # Normalizer / Scaler（保持与PPO一致的接口）
    # reward_scal = RewardScaling(shape=1, gamma=0.9)
    state_norm = Normalize(shape=state_size)

    log_dir_log = log_dir_log + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "-DDPG"
    summary_writer = SummaryWriter(log_dir_log)

    rewards_set = []
    epoch_set = []

    for e in range(epochs):
        print("++++++++++++++++++++++++++++++++++++++++++++++++++第", e, "次训练（DDPG）+++++++++++++++++++++++++++++++++++++++++++")
        state = env.reset()
        # reward_scal.reset()

        reward_sum = []
        actor_loss_val, critic_loss_val = None, None

        for step in range(3000):
            print("-------------------------DDPG：第", e, "次训练，第", step, "步-----------------------------")

            # 归一化当前状态
            s = state.astype(np.float32)[None, ...]
            s = state_norm(s)
            # print(f"state = {s}")

            # 动作（训练时带噪声探索）
            action = agent.get_action(s, add_noise=True)
            print(f"action = {action}")

            # 与环境交互
            reward, next_state, is_terminal = env.step(action)
            print(f"reward = {reward}")
            print(f"is_terminal = {is_terminal}")
            # reward = reward_scal(reward)

            # 归一化下一个状态（与当前状态保持一致处理）
            s_next = next_state.astype(np.float32)[None, ...]
            s_next = state_norm(s_next)

            # 写入经验池并训练一步
            agent.remember(s, action, reward, s_next, float(is_terminal))
            a_loss, _, c_loss, _ = agent.update()  # 当经验不足时会返回 (None, None, None, None)

            reward_sum.append(reward)

            if is_terminal:
                break  # 结束回合

            state = next_state

        # 记录回合平均奖励
        rewards_avg = np.round(sum(reward_sum) / max(1, len(reward_sum)), 2)
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
    args_env.use_potential_reward = False

    if args_env.use_potential_reward:
        log_dir_log = "/Users/tengman/Documents/Python/UAVTrajectory/pythonProject1103/log_data/ddpg/"
        log_dir_bp = "/Users/tengman/Documents/Python/UAVTrajectory/pythonProject1103/log_bp_data/ddpg/"
    else:
        log_dir_log = "/Users/tengman/Documents/Python/UAVTrajectory/pythonProject1103/log_data/ddpg_no_potential/"
        log_dir_bp = "/Users/tengman/Documents/Python/UAVTrajectory/pythonProject1103/log_bp_data/ddpg_no_potential/"

    epoch_set, rewards_set = training_ddpg(1000, log_dir_log, log_dir_bp)
    plt.figure(figsize=(8, 6))
    plt.plot(epoch_set, rewards_set, label='DDPG', color='blue')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.savefig('/Users/tengman/Documents/Python/UAVTrajectory/pythonProject1103/results_fig/train_ddpg_results/DDPG_Reward.png')
