from stable_baselines3 import PPO, SAC
from dogfight_wrapper import DogfightSelfPlayEnv
from opponent_agents import ALL_OPPONENTS

# --- CHANGE THESE PATHS TO YOUR ACTUAL CHECKPOINTS ---
my_model_path = "../models/dogfight/dogfight-selfplay-phase3.zip" 
# -----------------------------------------------------

print("Loading model...")
model = PPO.load(my_model_path) # Change to SAC.load if testing the SAC model

for opp_name in ["straight", "evasive", "nose_on"]:
    print(f"\nEvaluating vs {opp_name.upper()} (10 matches)...")
    opponent = ALL_OPPONENTS[opp_name]
    
    # Initialize env with the heuristic opponent
    env = DogfightSelfPlayEnv(opponent_policy=opponent, flatten_observation=True)
    
    wins, losses, draws = 0, 0, 0
    for i in range(10):
        obs, _ = env.reset(seed=i)
        total_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            
        if total_reward > 50:
            wins += 1
        elif total_reward < -50:
            losses += 1
        else:
            draws += 1
            
    print(f"Table 8 Entry -> [W]: {wins} / [L]: {losses} / [D]: {draws}")

env.close()