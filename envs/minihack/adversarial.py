
"""An environment which is built by a learning adversary.

Has additional functions, step_adversary, and reset_agent. How to use:
1. Call reset() to reset to an empty environment
2. Call step_adversary() to place the goal, agent, and obstacles. Repeat until
   a done is received.
3. Normal RL loop. Use learning agent to generate actions and use them to call
   step() until a done is received.
4. If required, call reset_agent() to reset the environment the way the
   adversary designed it. A new agent can now play it using the step() function.
"""

import random

import gym
import networkx as nx
from networkx import grid_graph
import numpy as np

from nle.minihack.navigation import MiniHackNavigation
#from nle.minihack.tasks import MiniHackMaze
from . import register
from . import minihackgrid


class MiniHackAdversarialEnv(MiniHackNavigation):
  """Grid world where an adversary build the environment the agent plays.

  The adversary places the goal, agent, and up to n_clutter blocks in sequence.
  The action dimension is the number of squares in the grid, and each action
  chooses where the next item should be placed.

  The difference here is we are now in MiniHack!
  """
  def __init__(self, n_clutter=50, size=15, agent_view_size=5, max_steps=250,
               goal_noise=0., random_z_dim=50, choose_goal_last=False, seed=0, fixed_environment=False):
    """Initializes environment in which adversary places goal, agent, obstacles.

    Args:
      n_clutter: The maximum number of obstacles the adversary can place.
      size: The number of tiles across one side of the grid; i.e. make a
        size x size grid.
      agent_view_size: The number of tiles in one side of the agent's partially
        observed view of the grid.
      max_steps: The maximum number of steps that can be taken before the
        episode terminates.
      goal_noise: The probability with which the goal will move to a different
        location than the one chosen by the adversary.
      random_z_dim: The environment generates a random vector z to condition the
        adversary. This gives the dimension of that vector.
      choose_goal_last: If True, will place the goal and agent as the last
        actions, rather than the first actions.
    """
    self.agent_start_pos = None
    self.goal_pos = None
    self.n_clutter = n_clutter
    self.goal_noise = goal_noise
    self.random_z_dim = random_z_dim
    self.choose_goal_last = choose_goal_last
    self.n_agents = 1

    # Add two actions for placing the agent and goal.
    self.adversary_max_steps = self.n_clutter + 2

    # generate empty grid
    self.width=size
    self.height=size
    self._gen_grid(self.width, self.height)

    super().__init__(des_file=self.grid.level.get_des())

    # Metrics
    self.reset_metrics()

    # Create spaces for adversary agent's specs.
    self.adversary_action_dim = (size - 2)**2
    self.adversary_action_space = gym.spaces.Discrete(self.adversary_action_dim)
    self.adversary_ts_obs_space = gym.spaces.Box(
        low=0, high=self.adversary_max_steps, shape=(1,), dtype='uint8')
    self.adversary_randomz_obs_space = gym.spaces.Box(
        low=0, high=1.0, shape=(random_z_dim,), dtype=np.float32)

    self.image_obs_space = gym.spaces.Box(
        low=0,
        high=255,
        shape=(self.width-2, self.height-2),
        dtype='uint8')

    # Adversary observations are dictionaries containing an encoding of the
    # grid, the current time step, and a randomly generated vector used to
    # condition generation (as in a GAN).
    self.adversary_observation_space = gym.spaces.Dict(
        {'image': self.image_obs_space,
         'time_step': self.adversary_ts_obs_space,
         'random_z': self.adversary_randomz_obs_space})

    self.observation_space = gym.spaces.Dict(
      {'image': self.image_obs_space,
       'chars_crop': self.observation_space['chars_crop']})

    # NetworkX graph used for computing shortest path
    self.graph = grid_graph(dim=[size-2, size-2])
    self.wall_locs = []

  def _gen_grid(self, width, height):
    """Grid is initially empty, because adversary will create it."""
    # Create an empty grid
    self.grid = minihackgrid.Grid(width=width, height=height)

  def get_goal_x(self):
    if self.goal_pos is None:
      return -1
    return self.goal_pos[0]

  def get_goal_y(self):
    if self.goal_pos is None:
      return -1
    return self.goal_pos[1]

  def reset_metrics(self):
    self.distance_to_goal = -1
    self.n_clutter_placed = 0
    self.deliberate_agent_placement = -1
    self.passable = -1
    self.shortest_path_length = (self.width - 2) * (self.height - 2) + 1

  def place_obj(self, obj):
    """
    randomly place an object in an empty loc
    """
    while True:
      x_rand = np.random.randint(0,self.width-2) + 1
      y_rand = np.random.randint(0,self.height-2) + 1
      if self.grid.get(x_rand, y_rand) == '.':
        self.grid.set(x_rand, y_rand, obj)
        return x_rand, y_rand

  def _get_image_obs(self):
    image = self.grid.encode()
    return image

  def reset(self):
    """Fully resets the environment to an empty grid with no agent or goal."""
    """The actual env (MiniHackMaze) won't change until you compile env."""

    self.graph = grid_graph(dim=[self.width-2, self.height-2])
    self.wall_locs = []

    self.step_count = 0
    self.adversary_step_count = 0

    self.agent_start_dir = np.random.randint(0, 4)

    # Current position and direction of the agent
    self.reset_agent_status()

    self.agent_start_pos = None
    self.goal_pos = None

    # Extra metrics
    self.reset_metrics()

    # Generate the grid. Will be random by default, or same environment if
    # 'fixed_environment' is True.
    self._gen_grid(self.width, self.height)

    cropped = super().reset()

    image = self._get_image_obs()
    obs = {
        'chars_crop': cropped['chars_crop'],
        'image': image, ## 2d array with string inputs
        'time_step': [self.adversary_step_count],
        'random_z': self.generate_random_z()
    }

    return obs

  def reset_agent_status(self):
    """Reset the agent's position, direction, done, and carrying status."""
    self.agent_pos = [None] * self.n_agents
    self.agent_dir = [self.agent_start_dir] * self.n_agents
    self.done = [False] * self.n_agents
    self.carrying = [None] * self.n_agents

  def reset_agent(self):
    """Resets the agent's start position, but leaves goal and walls."""
    # Remove the previous agents from the world
    for a in range(self.n_agents):
      if self.agent_pos[a] is not None:
          self.grid.set(int(self.agent_pos[a][0]), int(self.agent_pos[a][1]), '.')

    # Current position and direction of the agent
    self.reset_agent_status()

    self.grid.finalized = False

    if self.agent_start_pos is None:
      raise ValueError('Trying to place agent at empty start position.')
    else:
      #place agent at pos
      self.grid.set(int(self.agent_start_pos[0]), int(self.agent_start_pos[1]), '<')

    for a in range(self.n_agents):
      self.agent_pos[a] = self.agent_start_pos
      assert self.agent_pos[a] is not None
      assert self.agent_dir[a] is not None

      # Check that the agent doesn't overlap with an object
      start_cell = self.grid.get(*self.agent_pos[a])
      if not (start_cell == '<' or start_cell is None):
        raise ValueError('Wrong object in agent start position.')

    self.grid.finalized = True

    # Step count since episode start
    self.step_count = 0

    # Return first observation
    image = self._get_image_obs()
    obs = {
        'image': image,
        'chars_crop': super().reset()['chars_crop']
    }

    return obs

  def remove_wall(self, x, y):
    if (x-1, y-1) in self.wall_locs:
      self.wall_locs.remove((x-1, y-1))
    obj = self.grid.get(x, y)
    if obj == '-':
      self.grid.set(x, y, '.')

  def compute_shortest_path(self):
    if self.agent_start_pos is None or self.goal_pos is None:
      return 0

    self.distance_to_goal = abs(
        self.goal_pos[0] - self.agent_start_pos[0]) + abs(
            self.goal_pos[1] - self.agent_start_pos[1])

    # Check if there is a path between agent start position and goal. Remember
    # to subtract 1 due to outside walls existing in the Grid, but not in the
    # networkx graph.
    self.passable = nx.has_path(
        self.graph,
        source=(self.agent_start_pos[0] - 1, self.agent_start_pos[1] - 1),
        target=(self.goal_pos[0]-1, self.goal_pos[1]-1))
    if self.passable:
      # Compute shortest path
      self.shortest_path_length = nx.shortest_path_length(
          self.graph,
          source=(self.agent_start_pos[0]-1, self.agent_start_pos[1]-1),
          target=(self.goal_pos[0]-1, self.goal_pos[1]-1))
    else:
      self.shortest_path_length = 0

  def generate_random_z(self):
    return np.random.uniform(size=(self.random_z_dim,)).astype(np.float32)

  def step_adversary(self, loc):
    """The adversary gets n_clutter + 2 moves to place the goal, agent, blocks.

    The action space is the number of possible squares in the grid. The squares
    are numbered from left to right, top to bottom.

    Args:
      loc: An integer specifying the location to place the next object which
        must be decoded into x, y coordinates.

    Returns:
      Standard RL observation, reward (always 0), done, and info
    """
    if loc >= self.adversary_action_dim:
      raise ValueError('Position passed to step_adversary is outside the grid.')

    # Add offset of 1 for outside walls
    x = int(loc % (self.width - 2)) + 1
    y = int(loc // (self.width - 2)) + 1
    done = False

    if self.choose_goal_last:
      should_choose_goal = self.adversary_step_count == self.adversary_max_steps - 2
      should_choose_agent = self.adversary_step_count == self.adversary_max_steps - 1
    else:
      should_choose_goal = self.adversary_step_count == 0
      should_choose_agent = self.adversary_step_count == 1

    # Place goal
    if should_choose_goal:
      # If there is goal noise, sometimes randomly place the goal
      if random.random() < self.goal_noise:
        self.goal_pos = self.place_obj('>')
      else:
        self.remove_wall(x, y)  # Remove any walls that might be in this loc
        self.grid.set(x, y, '>')
        self.goal_pos = (x, y)

    # Place the agent
    elif should_choose_agent:
      self.remove_wall(x, y)  # Remove any walls that might be in this loc

      # Goal has already been placed here
      if self.grid.get(x, y) == '>':
        # Place agent randomly
        pos = self.place_obj('<')
        self.agent_start_pos = (pos[0], pos[1])
        self.grid.set(self.agent_start_pos[0], self.agent_start_pos[1], '<')
        self.deliberate_agent_placement = 0
      else:
        self.agent_start_pos = np.array([x, y])
        self.grid.set(self.agent_start_pos[0], self.agent_start_pos[1], '<')
        self.deliberate_agent_placement = 1

    # Place wall
    elif self.adversary_step_count < self.adversary_max_steps:
      # If there is already an object there, action does nothing
      if self.grid.get(x, y) == '.':
        self.grid.set(x, y, '-')
        self.n_clutter_placed += 1
        if (x-1, y-1) not in self.wall_locs:
          self.wall_locs.append((x-1, y-1))

    self.adversary_step_count += 1

    # End of episode
    if self.adversary_step_count >= self.adversary_max_steps:
      done = True
      # Build graph after we are certain agent and goal are placed
      for w in self.wall_locs:
        # check the wall loc is not the agent start or end pos
        if w == (self.agent_start_pos[0]-1, self.agent_start_pos[1]-1):
          pass
        elif w == (self.goal_pos[0]-1, self.goal_pos[1]-1):
          pass
        else:
          self.graph.remove_node(w)

      try:
        self.compute_shortest_path()
      except:
        print(self.grid.map, flush=True)
        print(self.grid.agent_start_pos, flush=True)
        print(self.grid.goal_pos, flush=True)
        self.shortest_path_length = 0


      # finalizing...
      self.grid.finalize_agent_goal()  # appends the locations to des file
      self.compile_env() # makes the environment

    image = self._get_image_obs()
    obs = {
        'image': image,
        'time_step': [self.adversary_step_count],
        'random_z': self.generate_random_z()
    }
    return obs, 0, done, {}

  def step(self, action):
    obs, reward, done, info = super().step(action)

    image = self._get_image_obs()
    obs['image'] = image ## 2d array with string inputs

    return obs, reward, done, info

  def compile_env(self):
      # updates the des file
      super().update(self.grid.level.get_des())

  def reset_random(self):
    if self.fixed_environment:
      self.seed(self.seed_value)

    """Use domain randomization to create the environment."""
    self.graph = grid_graph(dim=[self.width-2, self.height-2])

    self.step_count = 0
    self.adversary_step_count = 0

    # Current position and direction of the agent
    self.reset_agent_status()

    self.agent_start_pos = None
    self.goal_pos = None

    # Extra metrics
    self.reset_metrics()

    # Create empty grid
    self._gen_grid(self.width, self.height)

    # Randomly place goal
    self.goal_pos = self.place_obj('>')

    # Randomly place agent
    #self.agent_start_dir = self._rand_int(0, 4)
    self.agent_start_pos = self.place_obj('<')

    # Randomly place walls
    for _ in range(int(self.n_clutter / 2)):
      wall_loc = self.place_obj('-')
      if (wall_loc[0] - 1, wall_loc[1] - 1) not in self.wall_locs:
        self.wall_locs.append((wall_loc[0] - 1, wall_loc[1] - 1))

    self.compute_shortest_path()
    self.n_clutter_placed = int(self.n_clutter / 2)

    return self.reset_agent()

  def get_grid_str(self):
    return self.grid.get_grid_str()


class MiniHackGoalLastAdversarialEnv(MiniHackAdversarialEnv):
  def __init__(self):
    super().__init__(choose_goal_last=True)


if hasattr(__loader__, 'name'):
  module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
  module_path = __loader__.fullname


register.register(
    env_id='MiniHack-GoalLastAdversarial-v0',
    entry_point=module_path + ':MiniHackGoalLastAdversarialEnv'
)
