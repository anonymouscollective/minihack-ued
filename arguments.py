import argparse

import torch

from util import str2bool

parser = argparse.ArgumentParser(description='RL')

# PPO Arguments. 
parser.add_argument(
    '--algo',
    type=str,
    default='ppo',
    choices=['ppo', 'a2c', 'acktr', 'ucb', 'mixreg'],
    help='Which RL algorithm to use.')
parser.add_argument(
    '--lr', 
    type=float, 
    default=1e-4, 
    help='learning rate')
parser.add_argument(
    '--eps',
    type=float,
    default=1e-5,
    help='RMSprop optimizer epsilon')
parser.add_argument(
    '--alpha',
    type=float,
    default=0.99,
    help='RMSprop optimizer apha')
parser.add_argument(
    '--gamma',
    type=float,
    default=0.995,
    help='discount factor for rewards')
parser.add_argument(
    '--use_gae',
    type=str2bool, nargs='?', const=True, default=True,
    help='Use generalized advantage estimator.')
parser.add_argument(
    '--gae_lambda',
    type=float,
    default=0.95,
    help='gae lambda parameter')
parser.add_argument(
    '--entropy_coef',
    type=float,
    default=0.0,
    help='entropy term coefficient')
parser.add_argument(
    '--adv_entropy_coef',
    type=float,
    default=0.005,
    help='entropy term coefficient')
parser.add_argument(
    '--value_loss_coef',
    type=float,
    default=0.5,
    help='value loss coefficient (default: 0.5)')
parser.add_argument(
    '--max_grad_norm',
    type=float,
    default=0.5,
    help='max norm of gradients)')
parser.add_argument(
    '--normalize_returns',
    type=str2bool, nargs='?', const=True, default=False,
    help='Whether to use unnormalized returns')
parser.add_argument(
    '--seed', 
    type=int, 
    default=1, 
    help='random seed')
parser.add_argument(
    '--num_processes',
    type=int,
    default=4,
    help='how many training CPU processes to use')
parser.add_argument(
    '--num_steps',
    type=int,
    default=256,
    help='number of forward steps in A2C')
parser.add_argument(
    '--ppo_epoch',
    type=int,
    default=5,
    help='number of ppo epochs')
parser.add_argument(
    '--num_mini_batch',
    type=int,
    default=1,
    help='number of batches for ppo')
parser.add_argument(
    '--clip_param',
    type=float,
    default=0.2,
    help='ppo clip parameter')
parser.add_argument(
    '--num_env_steps',
    type=int,
    default=500000,
    help='number of environment steps to train')

# Architecture arguments.
parser.add_argument(
    '--recurrent_arch',
    type=str,
    default='lstm',
    choices=['gru', 'lstm'],
    help='RNN architecture')
parser.add_argument(
    '--recurrent_agent',
    type=str2bool, nargs='?', const=True, default=True,
    help='disables CUDA training')
parser.add_argument(
    '--recurrent_adversary_env',
    type=str2bool, nargs='?', const=True, default=False,
    help='disables CUDA training')
parser.add_argument(
    '--recurrent_hidden_size',
    type=int,
    default=256,
    help='Recurrent hidden state size')

# Environment arguments.
parser.add_argument(
    '--env_name',
    type=str,
    default='MiniHack-GoalLastAdversarial-v0',
    help='Environment to train on')
parser.add_argument(
    '--singleton_env',
    type=str2bool, nargs='?', const=True, default=False,
    help="When using a fixed env, whether the same environment should also be reused across workers.")

# PAIRED arguments.
parser.add_argument(
    '--ued_algo',
    type=str,
    default='paired',
    choices=['paired', 'flexible_paired', 'domain_randomization', 'minimax'],
    help='agent architecture')

# Hardware arguments.
parser.add_argument(
    '--no_cuda',
    type=str2bool, nargs='?', const=True, default=False,
    help='disables CUDA training')

# Logging arguments.
parser.add_argument(
    "--verbose", 
    type=str2bool, nargs='?', const=True, default=True,
    help="Whether to print logs")
parser.add_argument(
    '--xpid',
    default='latest',
    help='name for the run - prefix to log files')
parser.add_argument(
    '--log_dir',
    default='~/logs/paired/',
    help='directory to save agent logs')
parser.add_argument(
    '--log_interval',
    type=int,
    default=1,
    help='log interval, one log per n updates')
parser.add_argument(
    "--save_interval", 
    type=int, 
    default=20,
    help="Save model every this many minutes.")
parser.add_argument(
    '--archive_interval',
    type=int,
    default=100,
    help='archive interval, one archive per n updates')
parser.add_argument(
    "--weight_log_interval", 
    type=int, 
    default=0,
    help="Save level weights every this many updates")
parser.add_argument(
    "--screenshot_interval", 
    type=int, 
    default=1,
    help="Save screenshot of environment every this many updates.")
parser.add_argument(
    "--checkpoint", 
    type=str2bool, nargs='?', const=True, default=False,
    help="Begin training from checkpoint.")
parser.add_argument(
    "--disable_checkpoint", 
    type=str2bool, nargs='?', const=True, default=False,
    help="Disable saving checkpoint.")
parser.add_argument(
    '--log_grad_norm',
    type=str2bool, nargs='?', const=True, default=False,
    help="Log the gradient norm of the actor critic")
parser.add_argument(
    '--test_num_episodes',
    type=int,
    default=10,
    help='Number of test episodes per environment')
parser.add_argument(
    '--test_num_processes',
    type=int,
    default=4,
    help='Number of test processes per environment')
parser.add_argument(
    '--test_env_names',
    type=str,
    default='MiniHack-Room-15x15-v0,MiniHack-MazeWalk-9x9-v0,MiniHack-MazeWalk-15x15-v0',
    help='Environment to evaluate on')
parser.add_argument(
    '--test_interval',
    type=int,
    default=10,
    help='Evaluate on test envs every this many updates.')
