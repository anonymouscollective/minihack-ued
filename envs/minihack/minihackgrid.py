
import numpy as np
import torch

from nle.minihack import LevelGenerator

class Grid(object): # it may be worth subclassing LevelGenerator
    """
    Simple class to wrap the MiniHack level generator.
    This reduces the amount of string manipulation we need to do in adversarial.py
    """
    def __init__(self, width, height):

        self.level = LevelGenerator(map=None, w=width-2, h=height-2, lit=False)

        # separate the map
        self.get_grid_obs()
        self.agent_start_pos = None
        self.goal_pos = None
        self.finalized = False
        self.x = width-2
        self.y = height-2

    def get_grid_obs(self):
        """
        Converts the des file to a list for the agent
        """
        self.map = self.level.get_map_array()
        self.grid = [''.join(x) for x in self.map]

        return self.grid

    def get_grid_str(self):
        return self.level.get_map_array().__str__()

    def set(self, x, y, character='.'):
        x -= 1
        y -= 1

        if x > len(self.grid):
            raise ValueError(f'x value {x} is larger than the grid.')
        elif y > len(self.grid):
            raise ValueError(f'y value {y} is larger than the grid.')

        if character in ['.', '-']:
            self.level.add_terrain(coord=(x,y),flag = character)
        elif character in ['>']:
            self.goal_pos = (x, y)
        elif character in ['<']:
            self.agent_start_pos = (int(x), int(y))
        else:
            raise ValueError(f'character not supported.')

        # refresh map
        self.map = self.level.get_map_array()
        if self.goal_pos and not self.finalized:
            # add these to the map for the adversary
            self.map[self.goal_pos[0]][self.goal_pos[1]] = '>'
        if self.agent_start_pos and not self.finalized:
            self.map[self.agent_start_pos[0]][self.agent_start_pos[1]] = '<'

    def finalize_agent_goal(self):
        self.level.add_stair_down(self.goal_pos)
        self.map[self.goal_pos[0]][self.goal_pos[1]] = '.'
        self.level.add_stair_up(self.agent_start_pos)
        self.map[self.agent_start_pos[0]][self.agent_start_pos[1]] = '.'
        self.finalized = True

    def get(self, x, y):
        self.map = self.level.get_map_array()
        return self.map[x-1][y-1]

    def encode(self):
        obs = self.level.get_map_array()

        if self.goal_pos and not self.finalized:
            # add these to the map for the adversary
            obs[self.goal_pos[0]][self.goal_pos[1]] = '>'
        if self.agent_start_pos and not self.finalized:
            obs[self.agent_start_pos[0]][self.agent_start_pos[1]] = '<'

        obs = np.array([ord(x) for row in obs for x in row]).reshape([self.x, self.y])
        return obs
