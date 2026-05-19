import gymnasium
import numpy as np
from gymnasium import spaces


class FlattenWaypointEnv(gymnasium.ObservationWrapper):
    """Flattens the Dict observation of PyFlyt Waypoints envs into a single Box.

    The Waypoints env returns:
      - 'attitude': Box(21,) - drone state
      - 'target_deltas': (N, 3) waypoint deltas — N can decrease as waypoints are reached

    This wrapper pads/truncates target_deltas to a fixed number of waypoints
    and concatenates everything into a single flat vector.
    """

    def __init__(self, env, max_waypoints=4):
        super().__init__(env)
        self.max_waypoints = max_waypoints

        # Determine attitude dim from the observation space
        self.attitude_dim = env.observation_space["attitude"].shape[0]
        total_dim = self.attitude_dim + self.max_waypoints * 3

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_dim,), dtype=np.float64
        )

    def observation(self, obs):
        attitude = obs["attitude"]
        targets = obs["target_deltas"]  # shape (N, 3), N may vary

        # Pad or truncate to max_waypoints
        padded = np.zeros((self.max_waypoints, 3), dtype=np.float64)
        n = min(len(targets), self.max_waypoints)
        padded[:n] = targets[:n]

        return np.concatenate([attitude, padded.flatten()])
        

class ActionRepeat(gymnasium.Wrapper):
    """Repeat each action for n consecutive simulation steps, accumulating reward.

    The agent makes a decision every n steps instead of every step. This gives
    the PID controller time to actually execute the velocity command before the
    agent observes the outcome, reducing the effective control lag seen during
    learning.
    """

    def __init__(self, env, n: int = 4):
        super().__init__(env)
        self.n = n

    def step(self, action):
        total_reward = 0.0
        for _ in range(self.n):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info
    
class SimplifiedObsWrapper(gymnasium.ObservationWrapper):
    """
    Reduces the 33-dim observation to the essential components for
    mode 6 velocity-command navigation:
    - angular_velocity (3): still useful for stability awareness
    - linear_velocity (3): essential — the policy commands velocity, 
                           so knowing current velocity is critical
    - target_deltas[0] (3): direction and distance to next waypoint ONLY
    
    Total: 9 dimensions instead of 33
    """
    def __init__(self, env):
        super().__init__(env)
        # New observation space: 9-dimensional
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32
        )

    def observation(self, obs):
        # obs is already flattened by FlattenWaypointEnv (33-dim)
        angular_velocity = obs[0:3]    # indices 0-2
        linear_velocity = obs[7:10]   # indices 8-10
        next_target = obs[21:24]  # first target_delta (xyz to next waypoint)
        
        return np.concatenate([
            angular_velocity,
            linear_velocity,
            next_target,
        ]).astype(np.float32)
    
class SimpleObsWrapperTotal(gymnasium.ObservationWrapper):
    """
    Extracts the 13 essential components for stable flight and navigation.
    """
    def __init__(self, env):
        super().__init__(env)
        
        # 3 (ang) + 4 (quat) + 3 (lin) + 3 (target) = 13 dimensions
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32
        )

    def observation(self, obs):
        attitude = obs["attitude"]
        targets = obs["target_deltas"]
        
        # 1. Angular Velocity
        angular_velocity = attitude[0:3]
        
        # 2. Orientation (Quaternion) - CRUCIAL FOR SURVIVAL
        quaternion = attitude[3:7] 
        
        # 3. Linear Velocity
        linear_velocity = attitude[7:10]
            
        # 4. Next Target
        if len(targets) > 0:
            next_target = np.array(targets)[0][:3]
        else:
            next_target = np.zeros(3, dtype=np.float32)
            
        return np.concatenate([
            angular_velocity,
            quaternion,
            linear_velocity,
            next_target
        ]).astype(np.float32)

class WaypointCounterWrapper(gymnasium.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.waypoints_reached = 0

    def reset(self, **kwargs):
        self.waypoints_reached = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Check if a target was reached in this specific step
        if info.get("target_reached", False):
            self.waypoints_reached += 1
        
        # Inject the cumulative count into the info dict
        info["wps_reached_count"] = self.waypoints_reached
        
        # Also inject 'is_success' so SB3 shows you a success rate automatically
        info["is_success"] = info.get("env_complete", False)
        
        return obs, reward, terminated, truncated, info
    
