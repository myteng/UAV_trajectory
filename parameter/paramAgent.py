import argparse

import numpy as np

parser = argparse.ArgumentParser(description='AGENT')

# -- Agent --
# - Device -
parser.add_argument('--device', type=str, default='cpu',
                    help='The device (default: cpu)')
# - Seed -
parser.add_argument('--seed', type=int, default=0,
                    help='The random seed (default: 42)')
# - Model Type -
parser.add_argument('--agent_type', type=str, default='PPO',
                    help='The agent type (default: rgnn)')
# - Normal -
parser.add_argument('--use_state_normal', type=bool, default=True,
                    help='Normal State (default: True)')

# - Model parameters -
parser.add_argument('--epoch', type=int, default=5000,
                    help='The Number of epochs to run (default: 5000)')
parser.add_argument('--episode', type=int, default=1000,
                    help='The Number of episodes to run (default: 1000)')


parser.add_argument('--n_layers', type=int, default=2,
                    help='The number of layers (default: 2)')
parser.add_argument('--hidden_size', type=int, default=3,
                    help='Hidden size of rnn networks (default: 3)')

parser.add_argument('--gat_hidden_size', type=int, default=64,
                    help='Hidden size of GAT networks (default: 64)')
parser.add_argument('--gat_out_dim', type=int, default=3,
                    help='Out dim of GAT networks (default: 12)')
parser.add_argument('--n_heads', type=int, default=1,
                    help='Number of attention heads in graph observation encoder (default: 2)')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout of HAN (default: 0.5)')

# - Basic training hyperparameters -
parser.add_argument('--gamma', type=float, default=0.9,
                    help='The Discount factor (default: 0.9)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='The Learning rate (default: 1e-4)')
parser.add_argument('--polyak', type=float, default=0.995,
                    help='The Interpolation factor in polyak averaging for target network (default: 0.995)')

# - Buffer -
parser.add_argument('--batch_size', type=int, default=64,
                    help='The Minibatch size for SGD (default: 5)')
parser.add_argument('--replay_size', type=int, default=int(5e4),
                    help='The Capacity of replay buffer (default: 5e4)')
parser.add_argument('--max_seq_len', type=int, default=1e4,
                    help='The Maximum length of episode (default: 1e4)')

# - Optimization techniques -
parser.add_argument('--anneal_lr', type=bool, default=True,
                    help='Whether lr annealing is used (default: True)')


args_agent = parser.parse_args()