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
    """优先经验回放（Prioritized Experience Replay, PER）"""

    def __init__(self, capacity=int(1e6), alpha: float = 0.6):
        self.capacity = capacity
        self.buffer = []
        self.priorities = []
        self.pos = 0
        self.alpha = alpha

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
        if len(self.buffer) == 0:
            raise ValueError("PrioritizedReplayBuffer is empty")
        prios = np.array(self.priorities, dtype=np.float32)
        probs = prios ** self.alpha
        probs_sum = probs.sum()
        if probs_sum <= 0:
            probs = np.full_like(probs, 1.0 / len(probs))
        else:
            probs /= probs_sum

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

        N = len(self.buffer)
        weights = (N * probs[indices]) ** (-beta)
        weights /= weights.max()

        return (
            states, actions, rewards.reshape(-1, 1), next_states, dones.reshape(-1, 1),
            indices, weights.reshape(-1, 1),
        )

    def update_priorities(self, indices, td_errors, eps: float = 1e-6):
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
        self.update_epochs = 1
        
        # 学习率：Critic 稍高
        self.actor_lr = 1e-4
        self.critic_lr = 2e-4
        self.gamma = 0.99
        
        # batch_size 适度增大（原28太小）
        self.batch_size = 64
        self.hidden_sizes = (128, 128)  # 适度增大网络

        # DDPG 超参
        self.tau = 0.005
        self.noise_std = 0.2  # 探索噪声（纯高斯，不用OU）

        # PER 参数
        self.per_alpha = 0.6
        self.per_beta = 0.4
        self.per_beta_increment = 1e-4
        self.replay = PrioritizedReplayBuffer(capacity=int(1e6), alpha=self.per_alpha)
        self.min_replay_for_update = 1000  # 恢复原值

        # 动作缩放：θ ∈ [0, 2π)
        self.scale = tf.constant([math.pi] * action_dim, dtype=tf.float32)
        self.shift = self.scale

        # 构建网络
        self.actor = self.build_actor(self.hidden_sizes)
        self.critic = self.build_critic(self.hidden_sizes)
        self.target_actor = self.build_actor(self.hidden_sizes)
        self.target_critic = self.build_critic(self.hidden_sizes)

        # 初始化 target = online
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

        # 优化器
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=self.actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=self.critic_lr)

    def build_actor(self, hidden_sizes):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=self.state_dim))
        model.add(tf.keras.layers.Flatten())
        for h in hidden_sizes:
            model.add(tf.keras.layers.Dense(h, activation='relu'))
        model.add(tf.keras.layers.Dense(self.action_dim, activation='tanh'))
        return model

    def build_critic(self, hidden_sizes):
        state_in = tf.keras.Input(shape=self.state_dim)
        x = tf.keras.layers.Flatten()(state_in)
        for h in hidden_sizes:
            x = tf.keras.layers.Dense(h, activation='relu')(x)

        action_in = tf.keras.Input(shape=(self.action_dim,))
        z = tf.keras.layers.Concatenate()([x, action_in])
        for h in hidden_sizes:
            z = tf.keras.layers.Dense(h, activation='relu')(z)
        q = tf.keras.layers.Dense(1, activation=None)(z)
        return tf.keras.Model(inputs=[state_in, action_in], outputs=q)

    def _squash_and_scale(self, u):
        return self.scale * (u + 1.0)

    def _clip_to_valid(self, action):
        low = tf.zeros_like(self.scale)
        high = 2.0 * self.scale
        return tf.clip_by_value(action, low, high)

    def get_action(self, state, add_noise=True):
        """训练时带噪声探索"""
        u = self.actor(state)
        if add_noise:
            noise = tf.random.normal(shape=tf.shape(u), stddev=self.noise_std)
            u = tf.clip_by_value(u + noise, -1.0, 1.0)
        action = self._squash_and_scale(u)
        action = self._clip_to_valid(action)
        return action.numpy()[0]

    def evaluate(self, state):
        u = self.actor(state)
        action = self._squash_and_scale(u)
        action = self._clip_to_valid(action)
        return action.numpy()[0]

    @tf.function
    def _train_step(self, states, actions, rewards, next_states, dones, is_weights):
        next_u = self.target_actor(next_states)
        next_a = self._squash_and_scale(next_u)
        next_a = self._clip_to_valid(next_a)
        target_q = self.target_critic([next_states, next_a])
        y = rewards + (1.0 - dones) * self.gamma * target_q

        with tf.GradientTape() as tape:
            q = self.critic([states, actions])
            td_errors = q - tf.stop_gradient(y)
            critic_loss = tf.reduce_mean(is_weights * tf.square(td_errors))
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            u = self.actor(states)
            a = self._squash_and_scale(u)
            a = self._clip_to_valid(a)
            q_pi = self.critic([states, a])
            actor_loss = -tf.reduce_mean(q_pi)
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        return actor_loss, actor_grads, critic_loss, critic_grads, td_errors

    def update(self):
        if len(self.replay) < self.min_replay_for_update:
            return None, None, None, None

        self.per_beta = min(1.0, self.per_beta + self.per_beta_increment)

        (states, actions, rewards, next_states, dones, indices, is_weights
        ) = self.replay.sample(self.batch_size, beta=self.per_beta)

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        is_weights = tf.convert_to_tensor(is_weights, dtype=tf.float32)

        if states.shape.rank == 4 and states.shape[1] == 1:
            states = tf.squeeze(states, axis=1)
        if next_states.shape.rank == 4 and next_states.shape[1] == 1:
            next_states = tf.squeeze(next_states, axis=1)

        actor_loss, actor_grads, critic_loss, critic_grads, td_errors = self._train_step(
            states, actions, rewards, next_states, dones, is_weights
        )

        self.replay.update_priorities(indices, td_errors.numpy())
        self.soft_update(self.target_actor, self.actor)
        self.soft_update(self.target_critic, self.critic)

        return actor_loss, actor_grads, critic_loss, critic_grads

    def soft_update(self, target_net, source_net):
        for t, s in zip(target_net.trainable_variables, source_net.trainable_variables):
            t.assign(self.tau * s + (1.0 - self.tau) * t)

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
        try:
            checkpoint = th.load(path, map_location='cpu', weights_only=False)
        except Exception:
            import torch.serialization
            torch.serialization.add_safe_globals([np._core.multiarray._reconstruct])
            checkpoint = th.load(path, map_location='cpu', weights_only=True)

        self.actor.set_weights(checkpoint['actor_weights'])
        self.critic.set_weights(checkpoint['critic_weights'])
        if 'target_actor_weights' in checkpoint:
            self.target_actor.set_weights(checkpoint['target_actor_weights'])
        else:
            self.target_actor.set_weights(self.actor.get_weights())
        if 'target_critic_weights' in checkpoint:
            self.target_critic.set_weights(checkpoint['target_critic_weights'])
        else:
            self.target_critic.set_weights(self.critic.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.replay.push(state, action, reward, next_state, done)
