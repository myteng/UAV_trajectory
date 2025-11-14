import math

import tensorflow as tf
import numpy as np
import torch as th


class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim  # 状态的维度
        self.action_dim = action_dim  # 动作的维度
        self.update_epochs = 5  # 更新循环次数
        self.learning_rate = 1e-4  # 学习率
        self.clip_ratio = 0.2  # 策略更新的裁剪范围
        self.gamma = 0.99  # 折扣因子
        self.lam = 0.95  # GAE的λ参数
        self.horizon = 28  # 收集数据大小，即批次大小
        self.hidden_sizes = (64, 64)

        # 各动作分量的缩放参数
        # α: scale=0.5→(0,1)， θ: scale=π→[0,2π)
        self.scale = tf.constant([0.5] + [math.pi] * (action_dim - 1), dtype=tf.float32)
        self.shift = tf.constant([0.5] + [math.pi] * (action_dim - 1), dtype=tf.float32)

        # 构建策略网络（Actor）和价值网络（Critic）
        self.actor = self.build_actor(self.hidden_sizes)
        self.critic = self.build_critic(self.hidden_sizes)

        # 可训练log_std
        self.log_std = tf.Variable(tf.zeros(self.action_dim), trainable=True)

        # 定义优化器
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def build_actor(self, hidden_sizes):
        model = tf.keras.Sequential()  # 初始化一个顺序模型
        model.add(tf.keras.layers.InputLayer(shape=self.state_dim))  # 二维输入
        model.add(tf.keras.layers.Flatten())  # 在模型内展平
        for size in hidden_sizes:  # 添加隐藏层
            model.add(tf.keras.layers.Dense(size, activation='relu'))
        model.add(tf.keras.layers.Dense(self.action_dim, activation=None))  # 输出层，使用softmax激活
        return model

    def build_critic(self, hidden_sizes):
        model = tf.keras.Sequential()  # 初始化一个顺序模型
        model.add(tf.keras.layers.InputLayer(shape=self.state_dim))
        model.add(tf.keras.layers.Flatten())
        for size in hidden_sizes:  # 添加隐藏层
            model.add(tf.keras.layers.Dense(size, activation='relu'))
        model.add(tf.keras.layers.Dense(1, activation=None))  # 输出层，输出状态价值
        return model

    def get_action(self, state):
        mu = self.actor(state)
        std = tf.exp(tf.clip_by_value(self.log_std, -5.0, 2.0))
        eps = tf.random.normal(shape=tf.shape(mu))
        z = mu + eps * std  # 高斯采样
        u = tf.tanh(z)  # 压缩到 (-1,1)
        action = self.scale * (u + 1.0)  # 映射到目标区间
        # 检查action是否为NaN
        if tf.reduce_any(tf.math.is_nan(action)):
            tf.print("NaN detected in action!")
            tf.debugging.assert_all_finite(action, "Action contains NaN/Inf")
        return action.numpy()[0]

    def evaluate(self, state):
        mu = self.actor(state)
        std = tf.exp(tf.clip_by_value(self.log_std, -5.0, 2.0))
        eps = tf.random.normal(shape=tf.shape(mu))
        z = mu + eps * std  # 高斯采样
        u = tf.tanh(z)  # 压缩到 (-1,1)
        action = self.scale * (u + 1.0)  # 映射到目标区间
        return action.numpy()[0]

    def update(self, states, actions, rewards, is_terminals):
        """
        参数:
          states:    list/array of states  -> 会 vstack 成 [B, ...]
          actions:   list/array of actions -> 形状 [B, D]（就是 get_action 的返回，不裁剪）
          rewards:   [B,]
          is_terminals: [B,] 布尔或 0/1
        """
        # 批量更新策略网络和价值网络
        rewards_discount = self.compute_rewards_discount(rewards, is_terminals)

        states = np.vstack(states)
        actions = np.vstack(actions).astype(np.float32)

        # 计算旧策略下的概率值
        old_log_probs = self.compute_log_prob(states, actions)
        tf.debugging.assert_all_finite(old_log_probs, "old_log_probs contains NaN/Inf")

        # 循环多次更新
        actor_grads = None
        critic_grads = None
        for _ in range(self.update_epochs):
            # 更新策略网络Actor
            with tf.GradientTape() as tape:
                new_log_probs = self.compute_log_prob(states, actions)
                tf.debugging.assert_all_finite(new_log_probs, "new_log_probs contains NaN/Inf")
                delta = tf.clip_by_value(new_log_probs - old_log_probs, -20.0, 20.0)
                ratio = tf.exp(delta)  # [B]
                tf.debugging.assert_all_finite(ratio, "ratio contains NaN/Inf")
                actor_loss = self.compute_actor_loss(states, ratio, rewards_discount)
            actor_params = self.actor.trainable_variables + [self.log_std]
            actor_grads = tape.gradient(actor_loss, actor_params)
            self.actor_optimizer.apply_gradients(zip(actor_grads, actor_params))

            # 更新策略网络Critic
            with tf.GradientTape() as tape:
                critic_loss = self.compute_critic_loss(states, rewards_discount)
            critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        return actor_loss, actor_grads, critic_loss, critic_grads

    def compute_rewards_discount(self, rewards, is_terminals):
        rewards_discount = np.zeros_like(rewards)
        R = 0
        for t in reversed(range(len(rewards))):
            if is_terminals[t]:
                R = 0
            R = rewards[t] + self.gamma * R
            rewards_discount[t] = R
        return rewards_discount

    def compute_log_prob(self, states, actions):
        """连续动作下的计算概率比率 log_prob，用于PPO的损失函数"""
        mu = self.actor(states)
        std = tf.exp(tf.clip_by_value(self.log_std, -5.0, 2.0))
        # 反算采样前的 z
        u = actions / self.scale - 1.0
        u = tf.clip_by_value(u, -1.0 + 1e-6, 1.0 - 1e-6)
        z = tf.atanh(u)

        # Normal log_prob(z)
        normal_logp = -0.5 * (((z - mu) / std) ** 2 + 2.0 * tf.math.log(std) + tf.math.log(2.0 * math.pi))
        logp_z = tf.reduce_sum(normal_logp, axis=-1)

        # tanh 雅可比修正 + 线性缩放
        log_det_tanh = tf.reduce_sum(tf.math.log(1.0 - tf.tanh(z) ** 2 + 1e-6), axis=-1)
        log_det_scale = tf.reduce_sum(tf.math.log(self.scale), axis=-1)
        log_det_scale = tf.broadcast_to(log_det_scale, tf.shape(logp_z))  # 避免 log_det_scale 成为标量
        logp = logp_z - (-log_det_tanh) - log_det_scale
        return logp

    def compute_actor_loss(self, states, ratio, rewards_discount):
        # 计算策略（Actor）损失
        advantages = rewards_discount - self.critic(states)
        adv_mean = tf.reduce_mean(advantages)
        adv_std = tf.math.reduce_std(advantages) + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        sur1 = ratio * advantages
        sur2 = tf.clip_by_value(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
        return -tf.reduce_mean(tf.minimum(sur1, sur2))

    def compute_critic_loss(self, states, rewards_discount):
        # 计算价值（Critic）损失
        return tf.reduce_mean((rewards_discount - self.critic(states)) ** 2)  # 最小化回报和预测的价值之间的平方误差

    # ----------------- checkpoint -----------------
    def save_checkpoint(self, path, epoch):
        """Saves checkpoint for inference or resuming training."""
        checkpoint = {
            'actor_weights': self.actor.get_weights(),
            'critic_weights': self.critic.get_weights(),
            'epoch': epoch
        }
        th.save(checkpoint, path)
        # print(f"Save checkpoint to {path}.")

    def load_checkpoint(self, path):
        """Loads checkpoint from given path."""
        import numpy as np
        import torch as th
        try:
            # 旧版本 PyTorch (<2.6) 或文件兼容：正常读取
            checkpoint = th.load(path, map_location='cpu', weights_only=False)
        except Exception as e:
            # 新版本 PyTorch 2.6+ 会报安全限制错误，添加 numpy 反序列化白名单
            import torch.serialization
            torch.serialization.add_safe_globals([np._core.multiarray._reconstruct])
            checkpoint = th.load(path, map_location='cpu', weights_only=True)
            print(f" Safe-load fallback triggered due to {e}")

        epoch = checkpoint.get('epoch', 0)
        self.actor.set_weights(checkpoint['actor_weights'])
        self.critic.set_weights(checkpoint['critic_weights'])

        # print(f"Checkpoint loaded from {path} at step {epoch}")
        return epoch





