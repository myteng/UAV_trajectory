import datetime
import math

import numpy as np
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Rectangle
import os.path as osp

from ddpg_agent import DDPGAgent  # ← 替换为你的DDPGAgent路径
from envs.env import Environment
from parameter.paramAgent import args_agent
from parameter.paramEnv import args_env
from utils.common import set_rand_seed
from utils.normal import Normalize, RewardScaling


def visualize_uav_trajectories(env, uav_trajectories, epoch):
    """
    可视化无人机路径，包括边界和障碍物

    Args:
        env: Environment对象
        uav_trajectories: 每个UAV的路径列表，格式为 [[(x1,y1,z1), (x2,y2,z2), ...], ...]
        epoch: 当前训练轮次
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    # 绘制边界
    width = args_env.ranges_x
    height = args_env.ranges_y
    boundary = Rectangle((0, 0), width, height, linewidth=2,
                         edgecolor='black', facecolor='none', linestyle='--')
    ax.add_patch(boundary)

    # 绘制障碍物
    for obs in env.obstacle_map.obstacles:
        x, y = obs.center_xy
        if obs.shape == 'circle':
            circle = Circle((x, y), obs.size[0], fill=True,
                            color='gray', alpha=0.6, edgecolor='darkgray', linewidth=1.5)
            ax.add_patch(circle)
        elif obs.shape == 'rectangle':
            w, h = obs.size
            rect = Rectangle((x - w / 2, y - h / 2), w, h, fill=True,
                             color='gray', alpha=0.6, edgecolor='darkgray', linewidth=1.5)
            ax.add_patch(rect)

    # 绘制任务点分布
    job_x = [job.pos[0] for job in env.jobs]
    job_y = [job.pos[1] for job in env.jobs]
    ax.scatter(job_x, job_y, marker='*', s=180, color='#f6c344',
               edgecolors='black', linewidths=1.5, alpha=0.9, label='Jobs', zorder=4)

    # 定义颜色列表（为每个UAV分配不同颜色）
    colors = plt.cm.tab10(np.linspace(0, 1, env.n_uav))

    # 绘制每个UAV的路径
    for uav_id, trajectory in enumerate(uav_trajectories):
        if len(trajectory) == 0:
            continue

        # 提取x, y坐标（忽略z坐标）
        x_coords = [pos[0] for pos in trajectory]
        y_coords = [pos[1] for pos in trajectory]

        # 绘制路径线
        ax.plot(x_coords, y_coords, color=colors[uav_id], linewidth=2.5,
                alpha=0.8, label=f'UAV {uav_id}', zorder=3)

        # 标记起始点（只在第一个UAV时添加图例）
        if len(trajectory) > 0:
            label_start = 'Start' if uav_id == 0 else ""
            ax.scatter(x_coords[0], y_coords[0], color=colors[uav_id],
                       s=150, marker='o', edgecolors='black', linewidths=2.5,
                       zorder=5, label=label_start)

        # 标记终点（只在第一个UAV时添加图例）
        if len(trajectory) > 1:
            label_end = 'End' if uav_id == 0 else ""
            ax.scatter(x_coords[-1], y_coords[-1], color=colors[uav_id],
                       s=150, marker='s', edgecolors='black', linewidths=2.5,
                       zorder=5, label=label_end)

    # 设置图形属性
    ax.set_xlim(-10, width + 10)
    ax.set_ylim(-10, height + 10)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('X (m)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Y (m)', fontsize=13, fontweight='bold')
    ax.set_title(f'UAV Trajectories - Epoch {epoch}', fontsize=15, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9,
              framealpha=0.9, fancybox=True, shadow=True)

    plt.tight_layout()
    return fig


def training_ddpg(epochs, log_dir_log, log_dir_bp):
    # Set random seeds.
    seed = args_agent.seed
    set_rand_seed(seed)
    args_agent.agent_type = 'DDPG'

    env = Environment()

    state_size = (env.n_uav, env.n_feature)
    action_size = env.n_uav

    agent = DDPGAgent(state_size, action_size)

    # Normalizer / Scaler（保持与PPO一致的接口）
    # reward_scal = RewardScaling(shape=1, gamma=0.9)
    state_norm = Normalize(shape=state_size)

    log_dir_log = log_dir_log + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "-DDPG"
    summary_writer = SummaryWriter(log_dir_log)

    rewards_set = []
    epoch_set = []

    for e in range(epochs):
        print("++++++++++++++++++++++++++++++++++++++++++++++++++第", e,
              "次训练（DDPG）+++++++++++++++++++++++++++++++++++++++++++")
        state = env.reset()
        # reward_scal.reset()

        reward_sum = []
        actor_loss_val, critic_loss_val = None, None

        # 记录每个UAV的路径（用于可视化）
        uav_trajectories = [[] for _ in range(env.n_uav)]
        # 记录初始位置
        for uav_id in range(env.n_uav):
            uav_trajectories[uav_id].append(tuple(env.uav[uav_id].pos.copy()))

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

            # 记录每个UAV的当前位置
            for uav_id in range(env.n_uav):
                uav_trajectories[uav_id].append(tuple(env.uav[uav_id].pos.copy()))

            if is_terminal:
                break  # 结束回合

            state = next_state

        # 记录回合平均奖励
        rewards_avg = np.round(sum(reward_sum) / max(1, len(reward_sum)), 2)
        summary_writer.add_scalar("rewards", rewards_avg, e)

        rewards_set.append(rewards_avg)
        epoch_set.append(e)

        # 每100轮可视化路径
        if (e + 1) % 100 == 0:
            fig = visualize_uav_trajectories(env, uav_trajectories, e)
            summary_writer.add_figure('UAV_Trajectories', fig, e)
            plt.close(fig)  # 关闭图形以释放内存

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
    args_env.n_jobs = 50
    args_env.use_potential_reward = True

    if args_env.use_potential_reward:
        log_dir_log = "../../log_data/ddpg/"
        log_dir_bp = "../../log_data/ddpg/"
    else:
        log_dir_log = "../../log_data/ddpg_no_potential/"
        log_dir_bp = "../../log_bp_data/ddpg_no_potential/"

    epoch_set, rewards_set = training_ddpg(800, log_dir_log, log_dir_bp)
    plt.figure(figsize=(8, 6))
    plt.plot(epoch_set, rewards_set, label='DDPG', color='blue')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.savefig('../../results_fig/train_ddpg_results/DDPG_Reward.png')