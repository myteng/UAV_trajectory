import math
import tensorflow as tf
import numpy as np
import torch as th
from collections import deque
import random


class ReplayBuffer:
    def __init__(self, capacity=int(1e6)):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((np.array(state, copy=False),
                            np.array(action, copy=False),
                            float(reward),
                            np.array(next_state, copy=False),
                            float(done)))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards.reshape(-1, 1), next_states, dones.reshape(-1, 1)

    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """
    优先经验回放（Prioritized Experience Replay, PER）

    简化实现：
    - 使用一个 list 存储 transition，一个 list 存储对应的 priority。
    - 采样时按 p_i^alpha 归一化为概率分布进行随机采样。
    - 提供重要性采样权重 IS weights 以减小偏差。
    """

    def __init__(self, capacity=int(1e6), alpha: float = 0.6):
        self.capacity = capacity
        self.buffer = []
        self.priorities = []
        self.pos = 0
        self.alpha = alpha  # 控制优先程度，0 表示退化为均匀采样

    def push(self, state, action, reward, next_state, done):
        max_prio = max(self.priorities) if self.priorities else 1.0

        transition = (
            np.array(state, copy=False),
            np.array(action, copy=False),
            float(reward),
            np.array(next_state, copy=False),
            float(done),
        )

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(max_prio)
        else:
            self.buffer[self.pos] = transition
            self.priorities[self.pos] = max_prio
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta: float = 0.4):
        """返回带优先级的 batch 和 IS 权重。

        返回：
            states, actions, rewards, next_states, dones, indices, is_weights
        """
        if len(self.buffer) == 0:
            raise ValueError("PrioritizedReplayBuffer is empty")

        prios = np.array(self.priorities, dtype=np.float32)
        # 计算采样概率分布
        probs = prios ** self.alpha
        probs_sum = probs.sum()
        if probs_sum <= 0:
            # 极端情况保护：退化为均匀采样
            probs = np.full_like(probs, 1.0 / len(probs))
        else:
            probs /= probs_sum

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

        # 重要性采样权重
        N = len(self.buffer)
        weights = (N * probs[indices]) ** (-beta)
        weights /= weights.max()  # 归一化到 [0,1]

        return (
            states,
            actions,
            rewards.reshape(-1, 1),
            next_states,
            dones.reshape(-1, 1),
            indices,
            weights.reshape(-1, 1),
        )

    def update_priorities(self, indices, td_errors, eps: float = 1e-6):
        """根据 TD error 更新对应 transition 的 priority。"""
        td_errors = np.abs(np.asarray(td_errors)) + eps
        for idx, err in zip(indices, td_errors):
            self.priorities[int(idx)] = float(err)

    def __len__(self):
        return len(self.buffer)


class DDPGAgent:
    def __init__(self, state_dim, action_dim):
        # ---- 参数命名 ----
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.update_epochs = 1              # DDPG 每一步更新一次（也可按需多步），默认 1
        self.learning_rate = 1e-4
        self.gamma = 0.99
        # self.lam = 0.95                      # DDPG 不使用，但保留字段以“参数一致”
        self.horizon = 28                    # 用作默认的 batch_size
        self.hidden_sizes = (64, 64)

        # DDPG 相关新增超参
        self.tau = 0.005                     # 软更新系数
        self.noise_std = 0.3                 # 探索噪声（训练时用）

        # ---- 经验回放：使用优先经验回放（PER） ----
        # alpha 控制优先采样程度（0 表示退化为均匀采样）
        self.per_alpha = 0.6
        # beta 控制重要性采样权重，训练过程中通常从较小值逐渐增大到 1.0
        self.per_beta = 0.4
        self.per_beta_increment = 1e-4

        self.replay = PrioritizedReplayBuffer(capacity=int(1e6), alpha=self.per_alpha)
        self.min_replay_for_update = max(1000, 10 * self.horizon)  # 开始训练前需的最小经验量

        # 动作缩放（α∈(0,1)，θ∈[0,2π)）
        self.scale = tf.constant([0.5] + [math.pi] * (action_dim - 1), dtype=tf.float32)
        self.shift = tf.constant([0.5] + [math.pi] * (action_dim - 1), dtype=tf.float32)

        # ---- 构建网络：Actor / Critic 及其 Target ----
        self.actor = self.build_actor(self.hidden_sizes)
        self.critic = self.build_critic(self.hidden_sizes)
        self.target_actor = self.build_actor(self.hidden_sizes)
        self.target_critic = self.build_critic(self.hidden_sizes)

        # 初始化 target = online
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

        # 优化器（与 PPO 相同学习率）
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    # -------- 网络结构（与 PPO 风格保持：Sequential + Flatten）--------
    def build_actor(self, hidden_sizes):
        # 输出为 (-1,1) 的 u，经线性缩放映射到目标动作区间
        model = tf.keras.Sequential()
        # model.add(tf.keras.layers.InputLayer(shape=self.state_dim))
        model.add(tf.keras.layers.InputLayer(input_shape=self.state_dim))
        model.add(tf.keras.layers.Flatten())
        for h in hidden_sizes:
            model.add(tf.keras.layers.Dense(h, activation='relu'))
        model.add(tf.keras.layers.Dense(self.action_dim, activation='tanh'))  # u in (-1,1)
        return model

    def build_critic(self, hidden_sizes):
        # Q(s,a) 网络：输入 [state, action] 拼接后回归标量
        state_in = tf.keras.Input(shape=self.state_dim)
        x = tf.keras.layers.Flatten()(state_in)
        for h in hidden_sizes:
            x = tf.keras.layers.Dense(h, activation='relu')(x)

        action_in = tf.keras.Input(shape=(self.action_dim,))
        y = action_in
        for _ in range(0):  # 保留结构位，若想给 action 单独 MLP，可将 0 改为正整数
            y = tf.keras.layers.Dense(64, activation='relu')(y)

        # 拼接状态与动作特征
        z = tf.keras.layers.Concatenate()([x, y])
        for h in hidden_sizes:
            z = tf.keras.layers.Dense(h, activation='relu')(z)

        q = tf.keras.layers.Dense(1, activation=None)(z)
        model = tf.keras.Model(inputs=[state_in, action_in], outputs=q)
        return model

    # -------- 动作相关：缩放映射（将 Actor 输出的 u∈(-1,1) 映射为物理动作区间）--------
    def _squash_and_scale(self, u):
        # u in (-1,1)  ->  action = scale * (u + 1.0)
        return self.scale * (u + 1.0)

    def _clip_to_valid(self, action):
        # 将动作裁剪到合法范围：因为映射是单调的，(-1,1)->(0,1)×[0,2π)，这里保险起见再做下限/上限裁剪
        low = tf.zeros_like(self.scale)                     # (0, 0, ..., 0)
        high = 2.0 * self.scale                             # (1, 2π, ..., 2π)
        return tf.clip_by_value(action, low, high)

    # -------- 选择动作 --------
    def get_action(self, state, add_noise=True):
        """
        训练时推荐 add_noise=True（DDPG 需要探索），评估时可 False。
        返回 numpy 一维动作。
        """
        u = self.actor(state)                                   # (-1,1)
        if add_noise:
            noise = tf.random.normal(shape=tf.shape(u), stddev=self.noise_std)
            u = tf.clip_by_value(u + noise, -1.0, 1.0)
        action = self._squash_and_scale(u)
        action = self._clip_to_valid(action)
        return action.numpy()[0]

    def evaluate(self, state):
        # 评估用确定性策略
        u = self.actor(state)
        action = self._squash_and_scale(u)
        action = self._clip_to_valid(action)
        return action.numpy()[0]

    # -------- 训练更新（从 ReplayBuffer 采样）--------
    @tf.function
    def _train_step(self, states, actions, rewards, next_states, dones, is_weights):
        # 目标动作与 Q 目标
        next_u = self.target_actor(next_states)                    # (-1,1)
        next_a = self._squash_and_scale(next_u)
        next_a = self._clip_to_valid(next_a)
        target_q = self.target_critic([next_states, next_a])
        y = rewards + (1.0 - dones) * self.gamma * target_q

        # 更新 Critic：最小化加权 (Q - y)^2，这里的 is_weights 是 PER 的重要性采样权重
        with tf.GradientTape() as tape:
            q = self.critic([states, actions])
            td_errors = q - tf.stop_gradient(y)
            critic_loss = tf.reduce_mean(is_weights * tf.square(td_errors))
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        # 更新 Actor：最大化 Q(s, π(s))，等价于最小化 -Q
        with tf.GradientTape() as tape:
            u = self.actor(states)
            a = self._squash_and_scale(u)
            a = self._clip_to_valid(a)
            q_pi = self.critic([states, a])
            actor_loss = -tf.reduce_mean(q_pi)
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        # 这里额外返回 td_errors，供 PER 更新 priority 使用
        return actor_loss, actor_grads, critic_loss, critic_grads, td_errors

    def update(self):
        """
        与 PPOAgent.update 返回风格对齐：
        返回 (actor_loss, actor_grads, critic_loss, critic_grads)
        说明：
          - DDPG 不需要外部传回合轨迹，只要 ReplayBuffer 足够就可训练
          - 你可以每步交互后调用 push，再按固定频率调用 update()
        """
        if len(self.replay) < self.min_replay_for_update:
            return None, None, None, None

        # PER 中 beta 随训练逐渐增大，减小采样偏差
        self.per_beta = min(1.0, self.per_beta + self.per_beta_increment)

        # 这里用 horizon 作为 batch_size，保持“参数名一致”的风格
        batch_size = self.horizon
        (
            states,
            actions,
            rewards,
            next_states,
            dones,
            indices,
            is_weights,
        ) = self.replay.sample(batch_size, beta=self.per_beta)

        # 转为 tf tensor
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        is_weights = tf.convert_to_tensor(is_weights, dtype=tf.float32)
        # 如果多了一层 1（B,1,H,W），挤掉
        if states.shape.rank == 4 and states.shape[1] == 1:
            states = tf.squeeze(states, axis=1)
        if next_states.shape.rank == 4 and next_states.shape[1] == 1:
            next_states = tf.squeeze(next_states, axis=1)

        # 训练一步
        actor_loss, actor_grads, critic_loss, critic_grads, td_errors = self._train_step(
            states, actions, rewards, next_states, dones, is_weights
        )

        # 使用 TD error 更新 PER 中的 priority
        self.replay.update_priorities(indices, td_errors.numpy())

        # 软更新
        self.soft_update(self.target_actor, self.actor)
        self.soft_update(self.target_critic, self.critic)

        return actor_loss, actor_grads, critic_loss, critic_grads

    def soft_update(self, target_net, source_net):
        target_vars = target_net.trainable_variables
        source_vars = source_net.trainable_variables
        for t, s in zip(target_vars, source_vars):
            t.assign(self.tau * s + (1.0 - self.tau) * t)

    # -------- checkpoint 接口 --------
    def save_checkpoint(self, path, epoch):
        checkpoint = {
            'actor_weights': self.actor.get_weights(),
            'critic_weights': self.critic.get_weights(),
            'target_actor_weights': self.target_actor.get_weights(),
            'target_critic_weights': self.target_critic.get_weights(),
            'epoch': epoch
        }
        th.save(checkpoint, path)

    def load_checkpoint(self, path):
        # checkpoint = th.load(path, map_location='cpu')
        try:
            # 旧版本 PyTorch (<2.6) 或文件兼容：正常读取
            checkpoint = th.load(path, map_location='cpu', weights_only=False)
        except Exception as e:
            # 新版本 PyTorch 2.6+ 会报安全限制错误，添加 numpy 反序列化白名单
            import torch.serialization
            torch.serialization.add_safe_globals([np._core.multiarray._reconstruct])
            checkpoint = th.load(path, map_location='cpu', weights_only=True)
            # print(f" Safe-load fallback triggered due to {e}")

        epoch = checkpoint.get('epoch', 0)
        self.actor.set_weights(checkpoint['actor_weights'])
        self.critic.set_weights(checkpoint['critic_weights'])
        # 若包含 target 权重则一并恢复
        if 'target_actor_weights' in checkpoint:
            self.target_actor.set_weights(checkpoint['target_actor_weights'])
        else:
            self.target_actor.set_weights(self.actor.get_weights())

        if 'target_critic_weights' in checkpoint:
            self.target_critic.set_weights(checkpoint['target_critic_weights'])
        else:
            self.target_critic.set_weights(self.critic.get_weights())

        # print(f"Checkpoint loaded from {path} at step {epoch}")

    # -------- 与环境交互的便捷封装（可选）--------
    def remember(self, state, action, reward, next_state, done):
        self.replay.push(state, action, reward, next_state, done)
