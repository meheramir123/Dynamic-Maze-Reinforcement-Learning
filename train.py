from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import gymnasium as gym
import dynamic_maze  

#  Load environment 
env = gym.make("DynamicMaze-v0")

check_env(env)

#  Create PPO model 
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    ent_coef=0.01, 
    tensorboard_log="./logs/"
)

# Train 
print(" Starting Training...")
model.learn(total_timesteps=200000)  # Longer training for dynamic maze
print("Training Complete!")

# Save model 
model.save("ppo_dynamic_maze")
print(" Model saved as 'ppo_dynamic_maze'")

env.close()
