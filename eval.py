import sys
import os
import json
import argparse

import numpy as np
import torch
import gym
from baselines.common.vec_env import DummyVecEnv
from baselines.logger import HumanOutputFormat
from tqdm import tqdm

import nle.minihack

import os
import matplotlib as mpl
# mpl.use("macOSX")
import matplotlib.pyplot as plt

import nle.minihack

from envs.wrappers import VecMonitor, VecPreprocessImageWrapper, ParallelAdversarialVecEnv
from util import DotDict, str2bool, make_agent, create_parallel_env
from arguments import parser


def parse_args():
    parser = argparse.ArgumentParser(description='RL')

    parser.add_argument('--base_path', type=str, default='~/logs/paired', help='Base path to experiment results directories.')
    parser.add_argument('--xpid', type=str, default='latest', help='xpid for evaluation')
    parser.add_argument('--env_names', type=str, default='MiniHack-GoalLastAdversarial-v0', help='csv of env names')
    parser.add_argument('--singleton_env', type=str2bool, nargs='?', const=True, default=False, help="When using a fixed env, whether the same environment should also be reused across workers.")
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--num_processes', type=int, default=32, help='Number of CPU processes to use.')
    parser.add_argument('--num_episodes', type=int, default=100, help='Number of evaluation episodes.')
    parser.add_argument('--model_name', type=str, default='agent', choices=['agent', 'adversary_agent'], help='Which agent to evaluate if more than one option.')
    parser.add_argument('--deterministic', type=str2bool, nargs='?', const=True, default=False, help="Show logging messages in stdout")
    parser.add_argument('--verbose', type=str2bool, nargs='?', const=True, default=False, help="Show logging messages in stdout")

    return parser.parse_args()


class Evaluator(object):
    def __init__(self, env_names, num_processes, num_episodes=10, device='cpu'):
        self._init_parallel_envs(env_names, num_processes, device=device)
        self.num_episodes = num_episodes

    def _init_parallel_envs(self, env_names, num_processes, device=None):
        self.env_names = env_names
        self.num_processes = num_processes
        self.device = device
        self.venv = {env_name: None for env_name in env_names}

        make_fn = []
        for env_name in env_names:
            make_fn = [lambda: gym.make(env_name)] * self.num_processes
            venv = ParallelAdversarialVecEnv(make_fn)
            venv = VecMonitor(venv=venv, filename=None, keep_buf=100)

            transpose_order = [2, 0, 1]
            scale = 10.0
            obs_key = 'image'
            if 'MiniHack' in env_names[0]:
                transpose_order = None
                scale = None
                obs_key = 'chars_crop'

            venv = VecPreprocessImageWrapper(venv=venv, obs_key=obs_key, transpose_order=transpose_order, scale=scale, device=device)
            self.venv[env_name] = venv

    def close(self):
        for _, venv in self.venv.items():
            venv.close()

    def evaluate(self, agent, deterministic=False, show_progress=False):
        # Evaluate agent for N episodes
        venv = self.venv
        env_returns = {}
        env_solved_episodes = {}

        for env_name, venv in self.venv.items():
            returns = []
            solved_episodes = 0

            obs = venv.reset()
            recurrent_hidden_states = torch.zeros(self.num_processes, agent.algo.actor_critic.recurrent_hidden_state_size, device=self.device)
            if agent.algo.actor_critic.rnn is not None and agent.algo.actor_critic.rnn.arch == 'lstm':
                recurrent_hidden_states = (recurrent_hidden_states, torch.zeros_like(recurrent_hidden_states))
            masks = torch.ones(self.num_processes, 1, device=self.device)

            pbar = None
            if show_progress:
                print(f'Evaluating on {env_name}')
                pbar = tqdm(total=self.num_episodes)

            while len(returns) < self.num_episodes:
                # Sample actions
                with torch.no_grad():
                    _, action, _, recurrent_hidden_states = agent.act(obs, recurrent_hidden_states, masks, deterministic=deterministic)

                # Observe reward and next obs
                obs, reward, done, infos = venv.step(action.cpu())

                masks = torch.tensor([[0.0] if done_ else [1.0] for done_ in done], dtype=torch.float32, device=self.device)

                for i, info in enumerate(infos):
                    if 'episode' in info.keys():
                        returns.append(info['episode']['r'])
                        if returns[-1] > 0:
                            solved_episodes += 1
                        if pbar:
                            pbar.update(1)

                        if len(returns) >= self.num_episodes:
                            break
            if pbar:
                pbar.close()

            env_returns[env_name] = returns
            env_solved_episodes[env_name] = solved_episodes

        stats = {}
        for env_name in self.env_names:
            stats[f"solved_rate:{env_name}"] = env_solved_episodes[env_name] / self.num_episodes
            stats[f"test_returns:{env_name}"] = np.mean(env_returns[env_name])

        return stats


if __name__ == '__main__':
    os.environ["OMP_NUM_THREADS"] = "1"

    args = DotDict(vars(parse_args()))
    args.num_processes = min(args.num_processes, args.num_episodes)

    # === Determine device ====
    device = 'cpu'

    # === Load checkpoint ===
    # Load meta.json into flags object
    base_path = os.path.expandvars(os.path.expanduser(args.base_path))
    xpid_dir = os.path.join(base_path, args.xpid)
    meta_json_path = os.path.join(xpid_dir, 'meta.json')
    checkpoint_path = os.path.join(xpid_dir, 'model.tar')
    if os.path.exists(checkpoint_path):
        meta_json_file = open(meta_json_path)
        xpid_flags = DotDict(json.load(meta_json_file)['args'])

        # Get envs
        env_names = args.env_names.split(',')

        # Evaluate the model
        xpid_flags.update(args)
        make_fn = [lambda: gym.make(env_names[0])]
        dummy_venv = ParallelAdversarialVecEnv(make_fn)
        dummy_venv = VecPreprocessImageWrapper(venv=dummy_venv, obs_key='image', transpose_order=[2, 0, 1], scale=10.0)
        agent = make_agent(name='agent', env=dummy_venv, args=xpid_flags, device=device)

        dummy_venv.close()

        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        agent.algo.actor_critic.load_state_dict(checkpoint['runner_state_dict']['agent_state_dict']['agent'])

        evaluator = Evaluator(env_names, num_processes=args.num_processes, num_episodes=args.num_episodes)
        stats = evaluator.evaluate(agent, deterministic=args.deterministic, show_progress=args.verbose)

        HumanOutputFormat(sys.stdout).writekvs(stats)

        evaluator.close()
    else:
        raise ValueError(f'No model path {checkpoint_path}')