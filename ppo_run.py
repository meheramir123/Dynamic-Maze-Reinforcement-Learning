from stable_baselines3 import PPO
import gymnasium as gym
import dynamic_maze  

#  Load environment 
env = gym.make("DynamicMaze-v0", render_mode="human")

#  Load trained model 
model = PPO.load("ppo_dynamic_maze", env=env)

#  Run multiple episodes 
for episode in range(5):
    obs, _ = env.reset()
    done = False
    total_reward = 0

    print(f"\n Episode {episode + 1}")

    while not done:
        env.render()
        action, _ = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        print(f"Position: {obs}, Reward: {reward}")

    print(f" Episode {episode + 1} Total Reward: {total_reward}")

env.close()
