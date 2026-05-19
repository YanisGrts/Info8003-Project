import numpy as np
import os
from stable_baselines3 import PPO

def load_model(path=None):
    """Load and return a trained model for the dogfight tournament."""
    if path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(current_dir, "models/dogfight/dogfight-selfplay-phase1.zip")
    
    # Load the PPO model
    model = PPO.load(path)
    return model
