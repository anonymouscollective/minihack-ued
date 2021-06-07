from collections import deque, defaultdict

import numpy as np
import torch


class AdversarialRunner(object):
    """
    Performs rollouts of an adversarial environment, given 
    protagonist (agent), antogonist (adversary_agent), and
    environment adversary (advesary_env)
    """
    def __init__(
        self,
        args,
        venv,
        agent,
        adversary_agent=None,
        adversary_env=None,
        flexible_protagonist=False,
        train=False,
        device='cpu'):
        """
        venv: Vectorized, base adversarial gym env.
        agent: Protagonist trainer.
        adversary_agent: Antogonist trainer.
        adversary_env: Environment adversary trainer.

        flexible_protagonist: Which agent plays the role of protagonist in
        calculating the regret depends on which has the lowest score.
        """
        self.args = args

        self.venv = venv

        self.agents = {
            'agent': agent,
            'adversary_agent': adversary_agent,
            'adversary_env': adversary_env
        }

        self.agent_rollout_steps = args.num_steps
        self.adversary_env_rollout_steps = self.venv.adversary_observation_space['time_step'].high[0]
        self.is_training_env = args.ued_algo in ['paired', 'flexible_paired', 'minimax']
        self.is_paired = args.ued_algo in ['paired', 'flexible_paired']

        self.device = device

        self.reset()

        if train:
            self.train()
        else:
            self.eval()

    def reset(self):
        self.num_updates = 0
        self.total_episodes_collected = 0
        self.agent_returns = deque(maxlen=10)
        self.adversary_agent_returns = deque(maxlen=10)

    def train(self):
        self.is_training = True
        [agent.train() for _,agent in self.agents.items()]

    def eval(self):
        self.is_training = False
        [agent.eval() for _,agent in self.agents.items()]

    def state_dict(self):
        agent_state_dict = {}
        optimizer_state_dict = {}
        for k, agent in self.agents.items():
            if agent:
                agent_state_dict[k] = agent.algo.actor_critic.state_dict()
                optimizer_state_dict[k] = agent.algo.optimizer.state_dict()

        return {
            'agent_state_dict': agent_state_dict,
            'optimizer_state_dict': optimizer_state_dict,
            'agent_returns': self.agent_returns,
            'adversary_agent_returns': self.adversary_agent_returns,
            'num_updates': self.num_updates,
            'total_episodes_collected': self.total_episodes_collected
        }

    def load_state_dict(self, state_dict):
        agent_state_dict = state_dict['agent_state_dict']
        for k,state in agent_state_dict.items():
            self.agents[k].algo.actor_critic.load_state_dict(state)

        optimizer_state_dict = state_dict['optimizer_state_dict']
        for k,state in optimizer_state_dict.items():
            self.agents[k].algo.optimizer.load_state_dict(state)

        self.agent_returns = state_dict['agent_returns']
        self.adversary_agent_returns = state_dict['adversary_agent_returns']
        self.num_updates = state_dict['num_updates']
        self.total_episodes_collected = state_dict['total_episodes_collected']

    def _get_rollout_return_stats(self, rollout_returns):
        mean_return = torch.zeros(self.args.num_processes, 1)
        max_return = torch.zeros(self.args.num_processes, 1)
        for b, returns in enumerate(rollout_returns):
            if len(returns) > 0:
                mean_return[b] = float(np.mean(returns))
                max_return[b] = float(np.max(returns))

        stats = {
            'mean_return': mean_return,
            'max_return': max_return,
            'returns': rollout_returns 
        }

        return stats

    def _get_env_stats(self, agent_info, adversary_agent_info):
        num_blocks = np.mean(self.venv.get_num_blocks())
        passable_ratio = np.mean(self.venv.get_passable())
        shortest_path_lengths = self.venv.get_shortest_path_length()
        shortest_path_length = np.mean(shortest_path_lengths)

        if 'max_returns' in adversary_agent_info:
            solved_idx = \
                (torch.max(agent_info['max_return'], \
                    adversary_agent_info['max_return']) > 0).numpy().squeeze()
        else:
            solved_idx = (agent_info['max_return'] > 0).numpy().squeeze()

        solved_path_lengths = np.array(shortest_path_lengths)[solved_idx]
        solved_path_length = np.mean(solved_path_lengths) if len(solved_path_lengths) > 0 else 0

        stats = {
            'num_blocks': num_blocks,
            'passable_ratio': passable_ratio,
            'shortest_path_length': shortest_path_length,
            'solved_path_length': solved_path_length
        }       

        return stats

    def agent_rollout(self, agent, num_steps, update=False, is_env=False):
        if is_env:
            obs = self.venv.reset()
        else:
            obs = self.venv.reset_agent()
        
        # Initialize first observation
        try:
            agent.storage.copy_obs_to_index(obs,0)
        except:
            import pdb; pdb.set_trace()

        mean_return = 0

        rollout_returns = [[] for _ in range(self.args.num_processes)]
        for step in range(num_steps):
            # Sample actions
            with torch.no_grad():
                obs_id = agent.storage.get_obs(step)
                value, action, action_log_dist, recurrent_hidden_states = agent.act(
                    obs_id, agent.storage.get_recurrent_hidden_state(step), agent.storage.masks[step])
                action_log_prob = action_log_dist.gather(-1, action)

            # Observe reward and next obs
            if is_env:
                obs, reward, done, infos = self.venv.step_adversary(action.cpu())
            else:
                obs, reward, done, infos = self.venv.step(action.cpu())

            if not is_env and step >= num_steps - 1:
                done = np.ones_like(done, dtype=np.float)

            for i, info in enumerate(infos):
                if 'episode' in info.keys():
                    rollout_returns[i].append(info['episode']['r'])
                    self.total_episodes_collected += 1

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])

            agent.insert(
                obs, recurrent_hidden_states, 
                action, action_log_prob, action_log_dist, 
                value, reward, masks, bad_masks)

        rollout_info = self._get_rollout_return_stats(rollout_returns)

        if update:
            with torch.no_grad():
                obs_id = agent.storage.get_obs(-1)
                next_value = agent.get_value(
                    obs_id, agent.storage.get_recurrent_hidden_state(-1),
                    agent.storage.masks[-1]).detach()

            args = self.args
            agent.storage.compute_returns(next_value, args.use_gae, args.gamma, args.gae_lambda)

            # @todo: PLR level sampler update goes here
            value_loss, action_loss, dist_entropy, info = agent.update()
            rollout_info.update({
                'value_loss': value_loss,
                'action_loss': action_loss,
                'dist_entropy': dist_entropy,
                'update_info': info
            })

        return rollout_info

    def _compute_env_return(self, agent_info, adversary_agent_info):
        args = self.args
        if args.ued_algo == 'paired':
            env_return = torch.max(adversary_agent_info['max_return'] - agent_info['mean_return'], \
                torch.zeros_like(agent_info['mean_return']))

        elif args.ued_algo == 'flexible_paired':
            env_return = torch.zeros_like(agent_info['max_return'], dtype=torch.float)
            adversary_agent_max_idx = adversary_agent_info['max_return'] > agent_info['max_return']
            agent_max_idx = ~adversary_agent_max_idx

            env_return[adversary_agent_max_idx] = \
                adversary_agent_info['max_return'][adversary_agent_max_idx]
            env_return[agent_max_idx] = agent_info['max_return'][agent_max_idx]
            
            env_mean_return = torch.zeros_like(env_return, dtype=torch.float)
            env_mean_return[adversary_agent_max_idx] = \
                agent_info['mean_return'][adversary_agent_max_idx]
            env_mean_return[agent_max_idx] = \
                adversary_agent_info['mean_return'][agent_max_idx]

            env_return = torch.max(env_return - env_mean_return, torch.zeros_like(env_return))

        elif args.ued_algo == 'minimax':
            env_return = -agent_info['max_return']

        else:
            env_return = torch.zeros_like(agent_info['mean_return'])

        return env_return

    def run(self):
        args = self.args

        adversary_env = self.agents['adversary_env']
        agent = self.agents['agent']
        adversary_agent = self.agents['adversary_agent']

        # print(f'Peforming update {self.num_updates}')

        # Generate a batch of adversarial environments
        if self.is_training_env:
            self.agent_rollout(
                agent=adversary_env, 
                num_steps=self.adversary_env_rollout_steps, 
                update=False,
                is_env=True)
        elif args.ued_algo == 'domain_randomization':
            self.venv.reset_random()
        else:
            raise NotImplementedError

        # Run agent episodes
        agent_info = self.agent_rollout(
            agent=agent, 
            num_steps=self.agent_rollout_steps,
            update=self.is_training)

        adversary_agent_info = defaultdict(float)
        if self.is_paired:
            # Run adversary agent episodes
            adversary_agent_info = self.agent_rollout(
                agent=adversary_agent, 
                num_steps=self.agent_rollout_steps, 
                update=self.is_training)

        # Update adversary agent final return
        env_return = self._compute_env_return(agent_info, adversary_agent_info)


        adversary_env_info = defaultdict(float)
        if self.is_training and self.is_training_env:
            # print('UPDATING ENVIRONMENT')
            with torch.no_grad():
                obs_id = adversary_env.storage.get_obs(-1)
                next_value = adversary_env.get_value(
                    obs_id, adversary_env.storage.get_recurrent_hidden_state(-1),
                    adversary_env.storage.masks[-1]).detach()
            adversary_env.storage.replace_final_return(env_return)
            adversary_env.storage.compute_returns(next_value, args.use_gae, args.gamma, args.gae_lambda)
            env_value_loss, env_action_loss, env_dist_entropy, info = adversary_env.update()
            adversary_env_info.update({
                'action_loss': env_action_loss,
                'value_loss': env_value_loss,
                'dist_entropy': env_dist_entropy
            })

        self.num_updates += 1

        # For logging
        stats = self._get_env_stats(agent_info, adversary_agent_info)
        [self.agent_returns.append(r) for b in agent_info['returns'] for r in reversed(b)]
        mean_agent_return = 0
        if len(self.agent_returns) > 0:
            mean_agent_return = np.mean(self.agent_returns)

        mean_adversary_agent_return = 0
        if self.is_paired:
            [self.adversary_agent_returns.append(r) for b in adversary_agent_info['returns'] for r in reversed(b)]
            if len(self.adversary_agent_returns) > 0:
                mean_adversary_agent_return = np.mean(self.adversary_agent_returns)

        stats.update({
            'steps': (self.num_updates) * args.num_processes * args.num_steps,
            'total_episodes': self.total_episodes_collected,
            'mean_agent_return': mean_agent_return,
            'mean_adversary_agent_return': mean_adversary_agent_return,
            'mean_env_return': env_return.mean().item(),
            'agent_value_loss': agent_info['value_loss'],
            'agent_pg_loss': agent_info['action_loss'],
            'agent_dist_entropy': agent_info['dist_entropy'],
            'adversary_value_loss': adversary_agent_info['value_loss'],
            'adversary_pg_loss': adversary_agent_info['action_loss'],
            'adversary_dist_entropy': adversary_agent_info['dist_entropy'],
            'adversary_env_pg_loss': adversary_env_info['action_loss'],
            'adversary_env_value_loss': adversary_env_info['value_loss'],
            'adversary_env_dist_entropy': adversary_env_info['dist_entropy'],
        }) 

        return stats
