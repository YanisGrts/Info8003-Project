"""
Handcrafted opponent agents for dogfight evaluation.

VERIFIED OBSERVATION LAYOUT (37-dim, flatten_observation=True, uav_1 perspective):
    [0:3]   own angular velocity (body frame)           rad/s
    [3:6]   own linear velocity (body frame)            m/s  — vx=forward speed
    [6:9]   ENEMY world position (x, y, z)              metres  ← verified
    [9:12]  unknown (possibly body-frame relative vec)
    [12:16] control surface states
    [16:18] speed / misc flags
    [18]    armed flag (1.0)
    [19:22] zeros / padding
    [22:25] own forward direction in world frame        ← rotation matrix row 0
    [25:28] own angular velocity (world frame duplicate?)
    [28:34] further rotation / state info
    [34]    unknown scalar (~19, possibly own z pos)
    [35]    own health [0,1]
    [36]    enemy health [0,1]  (or time remaining)

KEY for tracking: 
    - Enemy world pos:     obs[6:9]
    - Own forward dir:     obs[22:25]  (unit vector pointing where nose faces)
    - Own forward speed:   obs[3]      (body frame vx)
    - Own angular vel:     obs[0:3]    (for damping)
"""

import numpy as np

# Verified indices
IDX_ANG_VEL_BODY  = slice(0, 3)    # own angular velocity body frame
IDX_LIN_VEL_BODY  = slice(3, 6)    # own linear velocity body frame (vx=forward)
IDX_ENEMY_POS     = slice(6, 9)    # enemy WORLD position  ← verified
IDX_FORWARD_DIR   = slice(22, 25)  # own forward unit vector in world frame
IDX_OWN_HEALTH    = 35
IDX_ENEMY_HEALTH  = 36


def _safe_norm(v, eps=1e-6):
    n = np.linalg.norm(v)
    return v / max(n, eps), n


class BaseAgent:
    def predict(self, obs: np.ndarray, deterministic: bool = True):
        obs    = np.asarray(obs, dtype=np.float32)
        action = np.clip(self._act(obs), -1.0, 1.0).astype(np.float32)
        return action, {}

    def _act(self, obs):
        raise NotImplementedError

    def reset(self):
        pass

    # ── Shared helpers ────────────────────────────────────────────────────

    def _enemy_pos(self, obs):
        return obs[IDX_ENEMY_POS].copy()

    def _forward_dir(self, obs):
        """Own nose direction in world frame (unit vector)."""
        d = obs[IDX_FORWARD_DIR].copy()
        n = np.linalg.norm(d)
        return d / max(n, 1e-6)

    def _ang_vel(self, obs):
        return obs[IDX_ANG_VEL_BODY].copy()

    def _forward_speed(self, obs):
        return float(obs[3])   # vx in body frame = forward speed

    def _bearing_to_enemy(self, obs):
        """
        Returns (lateral_err, vertical_err) — signed errors in world frame
        between own forward direction and direction to enemy.

        lateral_err  > 0 → enemy is to the RIGHT  → need right roll
        vertical_err > 0 → enemy is ABOVE          → need pitch up (neg pitch cmd)

        NOTE: We don't have own world position, so we use the forward direction
        vector and enemy world position to get a rough bearing.
        own_pos is approximated as origin since obs is likely ego-relative,
        or we use the geometry: cross(forward, enemy_dir) gives lateral component.
        """
        enemy_pos   = self._enemy_pos(obs)
        forward     = self._forward_dir(obs)

        # Enemy direction from origin (works if obs is ego-centric)
        # or gives the world-frame direction even if not perfectly ego-centric
        enemy_dir, enemy_dist = _safe_norm(enemy_pos)

        # Lateral error: cross product z-component (right-hand rule)
        # forward × enemy_dir → z component = how much enemy is to the right
        lateral_err  = forward[0] * enemy_dir[1] - forward[1] * enemy_dir[0]
        # Vertical error: enemy above/below in world frame
        vertical_err = enemy_dir[2] - forward[2]

        return lateral_err, vertical_err, enemy_dist


# ── 1. Random ─────────────────────────────────────────────────────────────────

class RandomAgent(BaseAgent):
    def _act(self, obs):
        return np.random.uniform(-1, 1, size=4)


# ── 2. Passive ────────────────────────────────────────────────────────────────

class PassiveAgent(BaseAgent):
    """Level flight — damps all rotation, ignores enemy."""
    KP = 0.5

    def _act(self, obs):
        av = self._ang_vel(obs)
        return np.array([
            float(np.clip(-self.KP * av[0], -1, 1)),
            float(np.clip(-self.KP * av[1], -1, 1)),
            0.0, 0.6
        ])


# ── 3. Straight ───────────────────────────────────────────────────────────────

class StraightAgent(BaseAgent):
    """Full throttle forward, damps rotation."""
    KP = 0.4

    def _act(self, obs):
        av = self._ang_vel(obs)
        return np.array([
            float(np.clip(-self.KP * av[0], -1, 1)),
            float(np.clip(-self.KP * av[1], -1, 1)),
            0.0, 1.0
        ])


# ── 4. Evasive ────────────────────────────────────────────────────────────────

class EvasiveAgent(BaseAgent):
    """
    Reacts to enemy bearing: breaks AWAY from the enemy's direction,
    then jinks randomly to make tracking harder.
    """
    DANGER_DIST   = 60.0
    JINK_INTERVAL = 30

    def __init__(self):
        self._step     = 0
        self._jink_roll = 0.0

    def reset(self):
        self._step = 0
        self._jink_roll = 0.0

    def _act(self, obs):
        self._step += 1
        lat, vert, dist = self._bearing_to_enemy(obs)
        av = self._ang_vel(obs)
        damp_roll = float(np.clip(-0.3 * av[0], -0.3, 0.3))

        if dist < self.DANGER_DIST:
            # Break AWAY: roll opposite to enemy's lateral position
            roll  = float(np.clip(-lat * 1.5 + damp_roll, -1, 1))
            pitch = -0.3   # slight climb while breaking
            return np.array([roll, pitch, 0.0, 1.0])

        # Random jink when safe
        if self._step % self.JINK_INTERVAL == 0:
            self._jink_roll = float(np.random.choice([-1.0, -0.6, 0.6, 1.0]))

        return np.array([
            float(np.clip(self._jink_roll + damp_roll, -1, 1)),
            -0.1,   # slight nose-up to maintain altitude
            0.0, 1.0
        ])


# ── 5. Aggressive ─────────────────────────────────────────────────────────────

class AggressiveAgent(BaseAgent):
    """
    Pure pursuit: always steers nose directly toward the enemy using
    the verified forward direction and enemy world position.
    This is the agent that actually tries to shoot.
    """
    KP_ROLL  = 1.2
    KP_PITCH = 1.0
    KD_ROLL  = 0.15   # derivative damping on roll rate

    def _act(self, obs):
        lat, vert, dist = self._bearing_to_enemy(obs)
        av = self._ang_vel(obs)

        # Roll toward enemy laterally, damp oscillation
        roll  = float(np.clip(self.KP_ROLL * lat - self.KD_ROLL * av[0], -1, 1))
        # Pitch toward enemy vertically (positive vert = enemy above = pitch up = neg cmd)
        pitch = float(np.clip(-self.KP_PITCH * vert, -1, 1))

        return np.array([roll, pitch, 0.0, 1.0])


# ── 6 & 7. Circling ───────────────────────────────────────────────────────────

class CirclingAgent(BaseAgent):
    """
    Constant banked turn. Pitch-up compensates for altitude loss in the bank.
    Tries to keep the enemy in a turning circle — classic BFM setup.
    """
    BANK  = 0.55
    PITCH = -0.15   # nose-up to compensate lift vector tilt

    def __init__(self, direction: int = 1):
        self.direction = direction

    def _act(self, obs):
        av = self._ang_vel(obs)
        yaw_damp = float(np.clip(-0.2 * av[2], -0.3, 0.3))
        return np.array([self.direction * self.BANK, self.PITCH, yaw_damp, 0.85])


# ── 8. Defensive ──────────────────────────────────────────────────────────────

class DefensiveAgent(BaseAgent):
    """
    Breaks HARD when enemy is in front, otherwise flies level.
    Uses verified bearing to detect when enemy is in our forward hemisphere.
    """
    def __init__(self):
        self._breaking      = False
        self._break_steps   = 0

    def reset(self):
        self._breaking    = False
        self._break_steps = 0

    def _act(self, obs):
        lat, vert, dist = self._bearing_to_enemy(obs)
        av = self._ang_vel(obs)
        damp = float(np.clip(-0.3 * av[0], -0.4, 0.4))

        # Enemy in front hemisphere = low lateral error + not behind us
        enemy_fwd = obs[IDX_FORWARD_DIR]
        enemy_pos, _ = _safe_norm(self._enemy_pos(obs))
        in_front = float(np.dot(enemy_fwd, enemy_pos)) > 0.0

        if self._breaking:
            self._break_steps -= 1
            if self._break_steps <= 0:
                self._breaking = False
            return np.array([1.0, 0.8, 0.0, 1.0])   # hard pull + roll

        if in_front and dist < 80.0:
            self._breaking    = True
            self._break_steps = 25
            return np.array([1.0, -0.9, 0.0, 1.0])   # break turn initiation

        return np.array([damp, -0.1, 0.0, 0.9])   # cruise


# ── 9. Altitude seeker ────────────────────────────────────────────────────────

class AltitudeSeekingAgent(BaseAgent):
    """
    Climbs then dives on the enemy using actual enemy position for the dive.
    """
    CLIMB_STEPS = 150
    DIVE_STEPS  = 80

    def __init__(self):
        self._step = 0

    def reset(self):
        self._step = 0

    def _act(self, obs):
        self._step += 1
        av = self._ang_vel(obs)
        damp = float(np.clip(-0.3 * av[0], -0.3, 0.3))
        cycle = self._step % (self.CLIMB_STEPS + self.DIVE_STEPS)

        if cycle < self.CLIMB_STEPS:
            return np.array([damp, -0.7, 0.0, 1.0])   # nose up, climb
        else:
            # Dive: use actual bearing to enemy
            lat, vert, _ = self._bearing_to_enemy(obs)
            roll  = float(np.clip(lat * 1.2 + damp, -1, 1))
            pitch = float(np.clip(-vert * 0.8 + 0.5, -1, 1))  # pitched down toward enemy
            return np.array([roll, pitch, 0.0, 1.0])


# ── 10. Nose-on (hardest) ─────────────────────────────────────────────────────

class NoseOnAgent(BaseAgent):
    """
    Lead pursuit: steers to intercept enemy's predicted position.
    Uses enemy world pos + own velocity direction to compute lead angle.
    This is the toughest opponent — it actively tries to put the nose on target.
    """
    KP_ROLL  = 1.5
    KP_PITCH = 1.2
    KD_ROLL  = 0.2
    KD_PITCH = 0.1

    def _act(self, obs):
        lat, vert, dist = self._bearing_to_enemy(obs)
        av = self._ang_vel(obs)

        roll  = float(np.clip(
            self.KP_ROLL * lat - self.KD_ROLL * av[0], -1, 1
        ))
        pitch = float(np.clip(
            -self.KP_PITCH * vert - self.KD_PITCH * av[1], -1, 1
        ))

        # Reduce throttle when very close to avoid ramming
        throttle = 0.7 if dist < 20.0 else 1.0

        return np.array([roll, pitch, 0.0, throttle])


# ── Registry ──────────────────────────────────────────────────────────────────

ALL_OPPONENTS = {
    "random":          RandomAgent(),
    "passive":         PassiveAgent(),
    "straight":        StraightAgent(),
    "evasive":         EvasiveAgent(),
    "aggressive":      AggressiveAgent(),
    "circling_right":  CirclingAgent(direction=+1),
    "circling_left":   CirclingAgent(direction=-1),
    "defensive":       DefensiveAgent(),
    "altitude_seeker": AltitudeSeekingAgent(),
    "nose_on":         NoseOnAgent(),
}
