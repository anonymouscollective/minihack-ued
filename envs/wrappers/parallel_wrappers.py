import multiprocessing as mp

import numpy as np
from baselines.common.vec_env import VecEnv, CloudpickleWrapper
from baselines.common.vec_env.vec_env import clear_mpi_env_vars


def worker(remote, parent_remote, env_fn_wrappers):
    def step_env(env, action):
        ob, reward, done, info = env.step(action)
        if done:
            try:
                ob = env.reset_agent()
            except AttributeError:
                ob = env.reset() # for eval, the envs don't have a reset_agent attribute.
        return ob, reward, done, info

    def step_adversary(env, action):
        ob, reward, done, info = env.step_adversary(action)
        return ob, reward, done, info

    parent_remote.close()
    envs = [env_fn_wrapper() for env_fn_wrapper in env_fn_wrappers.x]
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                remote.send([step_env(env, action) for env, action in zip(envs, data)])
            elif cmd == 'observation_space':
                remote.send(envs[0].observation_space)
            elif cmd == 'adversary_observation_space':
                remote.send(envs[0].adversary_observation_space)
            elif cmd == 'adversary_action_space':
                remote.send(envs[0].adversary_action_space)
            elif cmd == 'render':
                if data is not None:
                    remote.send([env.render(mode=data) for env in envs])
                else:
                    remote.send([env.render() for env in envs])
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces_spec':
                remote.send(CloudpickleWrapper((envs[0].observation_space, envs[0].action_space, envs[0].spec)))
            elif hasattr(envs[0], cmd):
                attrs = [getattr(env, cmd) for env in envs]
                is_callable = hasattr(attrs[0], '__call__')
                if is_callable:
                    if not hasattr(data, '__len__'):
                        data = [data]*len(attrs)
                    remote.send([attr(d) if d is not None else attr() for attr, d in zip(attrs, data)])
                else:
                    remote.send([attr for attr in attrs])
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
    finally:
        for env in envs:
            env.close()


class SubprocVecEnv(VecEnv):
    """
    VecEnv that runs multiple environments in parallel in subproceses and communicates with them via pipes.
    Recommended to use when num_envs > 1 and step() can be a bottleneck.
    """
    def __init__(self, env_fns, spaces=None, context='spawn', in_series=1):
        """
        Arguments:

        env_fns: iterable of callables -  functions that create environments to run in subprocesses. Need to be cloud-pickleable
        in_series: number of environments to run in series in a single process
        (e.g. when len(env_fns) == 12 and in_series == 3, it will run 4 processes, each running 3 envs in series)
        """
        self.waiting = False
        self.closed = False
        self.in_series = in_series
        nenvs = len(env_fns)
        assert nenvs % in_series == 0, "Number of envs must be divisible by number of envs to run in series"
        self.nremotes = nenvs // in_series
        env_fns = np.array_split(env_fns, self.nremotes)
        ctx = mp.get_context(context)
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.nremotes)])
        self.ps = [ctx.Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            with clear_mpi_env_vars():
                p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces_spec', None))
        observation_space, action_space, self.spec = self.remotes[0].recv().x
        self.viewer = None
        VecEnv.__init__(self, nenvs, observation_space, action_space)

    def step_async(self, actions):
        self._assert_not_closed()
        actions = np.array_split(actions, self.nremotes)
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        self._assert_not_closed()
        results = [remote.recv() for remote in self.remotes]
        results = _flatten_list(results)
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return _flatten_obs(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('reset', None))
        obs = [remote.recv() for remote in self.remotes]
        obs = _flatten_list(obs)
        return _flatten_obs(obs)

    def close_extras(self):
        self.closed = True
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    def render(self, mode):
        self._assert_not_closed()
        for pipe in self.remotes:
            pipe.send(('render', mode))
        imgs = [pipe.recv() for pipe in self.remotes]
        imgs = _flatten_list(imgs)
        return imgs

    def _assert_not_closed(self):
        assert not self.closed, "Trying to operate on a SubprocVecEnv after calling close()"

    def __del__(self):
        if not self.closed:
            self.close()

def _flatten_obs(obs):
    assert isinstance(obs, (list, tuple))
    assert len(obs) > 0

    if isinstance(obs[0], dict):
        keys = obs[0].keys()
        return {k: np.stack([o[k] for o in obs]) for k in keys if k != 'colors_crop'}
    else:
        return np.stack(obs)

def _flatten_list(l):
    assert isinstance(l, (list, tuple))
    assert len(l) > 0
    assert all([len(l_) > 0 for l_ in l])

    return [l__ for l_ in l for l__ in l_]


class ParallelAdversarialVecEnv(SubprocVecEnv):
    def __init__(self, env_fns):
        super().__init__(env_fns, )

    def seed_async(self, seed, index):
        self._assert_not_closed()
        self.remotes[index].send(('seed', seed))
        self.waiting = True

    def seed_wait(self, index):
        self._assert_not_closed()
        obs = self.remotes[index].recv()
        self.waiting = False
        return obs

    def seed(self, seed, index):
        self.seed_async(seed, index)
        return self.seed_wait(index)

    def level_seed_async(self, index):
        self._assert_not_closed()
        self.remotes[index].send(('level_seed', None))
        self.waiting = True

    def level_seed_wait(self, index):
        self._assert_not_closed()
        level_seed = self.remotes[index].recv()
        self.waiting = False
        return level_seed

    def level_seed(self, index):
        self.level_seed_async(index)
        return self.level_seed_wait(index)

    # step_adversary
    def step_adversary(self, action):
        self.step_adversary_async(action)
        return self.step_adversary_wait()

    def step_adversary_async(self, action):
        self._assert_not_closed()
        [remote.send(('step_adversary', a)) for remote, a in zip(self.remotes, action)]
        self.waiting = True

    def step_adversary_wait(self):
        self._assert_not_closed()
        results = [remote.recv() for remote in self.remotes]
        results = _flatten_list(results)
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return _flatten_obs(obs), np.stack(rews), np.stack(dones), infos

    # reset_agent
    def reset_agent(self):
        self._assert_not_closed()
        [remote.send(('reset_agent', None)) for remote in self.remotes]
        obs = [remote.recv() for remote in self.remotes]
        obs = _flatten_list(obs)
        return _flatten_obs(obs)

    # reset_random
    def reset_random(self):
        self._assert_not_closed()
        [remote.send(('reset_random', None)) for remote in self.remotes]
        obs = [remote.recv() for remote in self.remotes]
        obs = _flatten_list(obs)
        return _flatten_obs(obs)

    # observation_space
    def get_observation_space(self):
        self._assert_not_closed()
        self.remotes[0].send(('observation_space', None))
        obs_space = self.remotes[0].recv()
        if hasattr(obs_space, 'spaces'):
            obs_space = obs_space.spaces
        return obs_space

    # adversary_observation_space
    def get_adversary_observation_space(self):
        self._assert_not_closed()
        self.remotes[0].send(('adversary_observation_space', None))
        obs_space = self.remotes[0].recv()
        if hasattr(obs_space, 'spaces'):
            obs_space = obs_space.spaces
        return obs_space

    def get_adversary_action_space(self):
        self._assert_not_closed()
        self.remotes[0].send(('adversary_action_space', None))
        action_dim = self.remotes[0].recv()
        return action_dim

    def get_grid_str(self):
        self._assert_not_closed()
        self.remotes[0].send(('get_grid_str', None))
        grid_obs = self.remotes[0].recv()
        return grid_obs[0]

    def remote_attr(self, name, data=None, flatten=False):
        self._assert_not_closed()
        if hasattr(data, '__len__'):
            assert len(data) == len(self.remotes)
            [remote.send((name, d)) for remote, d in zip(self.remotes, data)]
        else:
            [remote.send((name, data)) for remote in self.remotes]
        result = [remote.recv() for remote in self.remotes]
        return _flatten_list(result) if flatten else result

    def get_num_blocks(self):
        return self.remote_attr('n_clutter_placed', flatten=True)

    def get_distance_to_goal(self):
        return self.remote_attr('distance_to_goal', flatten=True)

    def get_passable(self):
        return self.remote_attr('passable', flatten=True)

    def get_shortest_path_length(self):
        return self.remote_attr('shortest_path_length', flatten=True)

    def get_seed(self):
        return self.remote_attr('seed_value', flatten=True)

    def set_seed(self, seeds):
        self.remote_attr('seed', data=seeds, flatten=True)

    def __getattr__(self, name):
        if name == 'observation_space':
            return self.get_observation_space()
        elif name == 'adversary_observation_space':
            return self.get_adversary_observation_space()
        elif name == 'adversary_action_space':
            return self.get_adversary_action_space()
        else:
            return self.__getattribute__(name)



