import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random


class DynamicMazeEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, render_mode=None):
        super(DynamicMazeEnv, self).__init__()

        #  Maze settings 
        self.grid_size = (10, 10)  # 10x10 maze
        self.cell_size = 60

        self.start_pos = (0, 0)
        self.goal_pos = (9, 9)

        # Number of elements
        self.num_bombs = 8
        self.num_pits = 2
        self.num_rewards = 5

        #  Action space 
        self.action_space = spaces.Discrete(4)  # Up, Down, Left, Right

        #  Observation space 
        self.observation_space = spaces.Box(
            low=0, high=max(self.grid_size) - 1, shape=(2,), dtype=np.int32
        )

        self.state = np.array(self.start_pos, dtype=np.int32)

        # For pygame rendering
        self.window = None
        self.clock = None

        # Dynamic elements
        self.bombs = []
        self.pits = []
        self.rewards = []

        # Max steps to prevent infinite loops
        self.max_steps = 200
        self.current_steps = 0

    def reset(self, seed=None, options=None):
        self.state = np.array(self.start_pos, dtype=np.int32)
        self.current_steps = 0

        # Generate random positions
        all_positions = [
            (x, y)
            for x in range(self.grid_size[0])
            for y in range(self.grid_size[1])
            if (x, y) != self.start_pos and (x, y) != self.goal_pos
        ]

        random.shuffle(all_positions)

        self.bombs = all_positions[:self.num_bombs]
        self.pits = all_positions[self.num_bombs : self.num_bombs + self.num_pits]
        self.rewards = all_positions[
            self.num_bombs + self.num_pits : self.num_bombs + self.num_pits + self.num_rewards
        ]

        return self.state, {}

    def step(self, action):
        self.current_steps += 1
        x, y = self.state

        # Movement
        if action == 0 and x > 0:
            x -= 1  # Up
        elif action == 1 and x < self.grid_size[0] - 1:
            x += 1  # Down
        elif action == 2 and y > 0:
            y -= 1  # Left
        elif action == 3 and y < self.grid_size[1] - 1:
            y += 1  # Right

        self.state = np.array([x, y], dtype=np.int32)
        pos = (x, y)

        #  Reward shaping 
        goal_distance = abs(self.goal_pos[0] - x) + abs(self.goal_pos[1] - y)
        reward = -0.2 * goal_distance  # Encourages moving toward the goal

        terminated = False

        if pos in self.bombs:
            reward = -20
            terminated = True
        elif pos in self.pits:
            reward = -50
            terminated = True
        elif pos in self.rewards:
            reward += 10
        elif pos == self.goal_pos:
            reward = 200
            terminated = True
        else:
            reward += -1  # Small step penalty

        if self.current_steps >= self.max_steps:
            terminated = True

        return self.state, reward, terminated, False, {}

    def render(self):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode(
                (self.grid_size[1] * self.cell_size, self.grid_size[0] * self.cell_size)
            )
            pygame.display.set_caption("Dynamic Maze Environment")
            self.clock = pygame.time.Clock()

        # Colors
        white = (255, 255, 255)
        black = (0, 0, 0)
        red = (200, 50, 50)
        orange = (255, 165, 0)
        yellow = (230, 230, 50)
        green = (50, 200, 50)
        blue = (50, 50, 230)

        self.window.fill(white)

        # Draw grid
        for row in range(self.grid_size[0]):
            for col in range(self.grid_size[1]):
                rect = pygame.Rect(
                    col * self.cell_size, row * self.cell_size,
                    self.cell_size, self.cell_size
                )
                pygame.draw.rect(self.window, black, rect, width=2)

        def draw_cell(position, color):
            rect = pygame.Rect(
                position[1] * self.cell_size + 5,
                position[0] * self.cell_size + 5,
                self.cell_size - 10,
                self.cell_size - 10,
            )
            pygame.draw.rect(self.window, color, rect)

        for bomb in self.bombs:
            draw_cell(bomb, red)

        for pit in self.pits:
            draw_cell(pit, orange)

        for reward_tile in self.rewards:
            draw_cell(reward_tile, yellow)

        draw_cell(self.goal_pos, green)   # Goal
        draw_cell(tuple(self.state), blue)  # Agent

        pygame.display.flip()
        self.clock.tick(4)

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None


# Register the environment
from gymnasium.envs.registration import register

register(
    id="DynamicMaze-v0",
    entry_point="dynamic_maze:DynamicMazeEnv",
)
