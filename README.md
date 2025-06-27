# Dynamic-Maze-Reinforcement-Learning
A custom **Dynamic Maze Environment** built with **Gymnasium** and **Pygame**, where the layout changes every episode. The goal is to train a reinforcement learning agent using **Proximal Policy Optimization (PPO)** to navigate the maze, collect rewards, avoid traps, and reach the goal.











# Features
# Dynamic Maze: Every episode generates a brand-new maze with randomly placed bombs, death pits, and rewards, keeping the challenge fresh and unpredictable.
# Obstacles & Hazards
    -Bombs: Instant game over with a -20 penalty.
    -Death Pits: Fall in and it’s a -50 penalty, plus immediate game over.

# Rewards
Find and collect +10 reward tiles scattered around the maze to boost your score.

# Goal
Reach the goal tile to earn a massive +200 reward and successfully complete the episode.

# Intelligent Agent
Trained using Proximal Policy Optimization (PPO) to master navigation in this ever-changing environment.

# Visual Rendering
Built with Pygame, offering an interactive and intuitive visual experience.

# Seamless Integration
Fully compatible with Stable-Baselines3 and Gymnasium, making it easy to train, test, and expand.

# Install dependencies
    -pip install gymnasium stable-baselines3 pygame numpy


