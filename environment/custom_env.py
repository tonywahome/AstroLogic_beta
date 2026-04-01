"""AstroLogic Custom Gymnasium Environment.

Consolidates: celestial body data, orbital physics, reward calculator,
action-space wrappers, and the main AstroExplorationEnv class.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque

# ============================================================
# Celestial body data
# All distances in AU, masses in solar masses, periods in days.
# ============================================================

CELESTIAL_BODIES = {
    "Sun": {
        "mass": 1.0,
        "radius": 0.00465,
        "orbit_radius": 0.0,
        "orbit_period": 0.0,
        "initial_angle": 0.0,
        "color": (255, 223, 0),
        "parent": None,
        "biosignatures": [],
        "detection_zone_radius": 0.0,
    },
    "Earth": {
        "mass": 3.003e-6,
        "radius": 0.0000426,
        "orbit_radius": 1.0,
        "orbit_period": 365.25,
        "initial_angle": 0.0,
        "color": (70, 130, 200),
        "parent": "Sun",
        "biosignatures": [],
        "detection_zone_radius": 0.0,
    },
    "Mars": {
        "mass": 3.213e-7,
        "radius": 0.0000227,
        "orbit_radius": 1.524,
        "orbit_period": 687.0,
        "initial_angle": np.pi / 4,
        "color": (193, 68, 14),
        "parent": "Sun",
        "biosignatures": ["ice", "organic_compounds"],
        "detection_zone_radius": 0.05,
    },
    "Jupiter": {
        "mass": 9.543e-4,
        "radius": 0.000467,
        "orbit_radius": 5.203,
        "orbit_period": 4332.59,
        "initial_angle": np.pi / 3,
        "color": (201, 176, 131),
        "parent": "Sun",
        "biosignatures": [],
        "detection_zone_radius": 0.0,
    },
    "Europa": {
        "mass": 2.528e-8,
        "radius": 0.0000104,
        "orbit_radius": 0.0045,
        "orbit_period": 3.55,
        "initial_angle": 0.0,
        "color": (200, 220, 240),
        "parent": "Jupiter",
        "biosignatures": ["liquid_water", "organic_compounds"],
        "detection_zone_radius": 0.03,
    },
    "Saturn": {
        "mass": 2.857e-4,
        "radius": 0.000389,
        "orbit_radius": 9.537,
        "orbit_period": 10759.22,
        "initial_angle": 2 * np.pi / 3,
        "color": (210, 180, 100),
        "parent": "Sun",
        "biosignatures": [],
        "detection_zone_radius": 0.0,
    },
    "Enceladus": {
        "mass": 1.08e-10,
        "radius": 0.00000168,
        "orbit_radius": 0.00159,
        "orbit_period": 1.37,
        "initial_angle": np.pi / 2,
        "color": (180, 200, 220),
        "parent": "Saturn",
        "biosignatures": ["liquid_water", "ice", "signs_of_intelligence"],
        "detection_zone_radius": 0.02,
    },
}

BIOSIGNATURE_REWARDS = {
    "liquid_water": 500.0,
    "ice": 300.0,
    "organic_compounds": 750.0,
    "signs_of_intelligence": 5000.0,
}

TARGET_BODIES = ["Mars", "Europa", "Enceladus"]

INSTRUMENTS = {
    0: {"name": "None", "detects": []},
    1: {"name": "Spectrometer", "detects": ["liquid_water", "organic_compounds"]},
    2: {"name": "ThermalImager", "detects": ["ice", "liquid_water"]},
    3: {"name": "Drill", "detects": ["organic_compounds", "ice", "signs_of_intelligence"]},
}

# ============================================================
# Orbital physics helpers
# Normalized units: AU, days, solar masses.
# ============================================================

G_NORMALIZED = 4.0 * np.pi**2 / 365.25**2


def gravitational_acceleration(pos, body_pos, body_mass):
    direction = body_pos - pos
    distance = np.linalg.norm(direction)
    if distance < 1e-8:
        return np.zeros(3)
    return G_NORMALIZED * body_mass / (distance**2) * (direction / distance)


def orientation_to_direction(pitch, yaw, roll):
    """Convert Euler angles (radians) to a unit forward direction vector.
    Roll does not affect the forward direction.
    """
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    cos_pitch, sin_pitch = np.cos(pitch), np.sin(pitch)
    forward = np.array([cos_pitch * cos_yaw, cos_pitch * sin_yaw, sin_pitch])
    return forward / (np.linalg.norm(forward) + 1e-8)


def compute_orbital_position(orbit_radius, orbit_period, time,
                             initial_angle, parent_pos=None):
    if orbit_period <= 0 or orbit_radius <= 0:
        return parent_pos.copy() if parent_pos is not None else np.zeros(3)
    angle = initial_angle + 2.0 * np.pi * time / orbit_period
    pos = np.array([orbit_radius * np.cos(angle),
                    orbit_radius * np.sin(angle), 0.0])
    if parent_pos is not None:
        pos += parent_pos
    return pos


def normalize_distance(distance, max_distance=50.0):
    return max(0.0, 1.0 - distance / max_distance)


def compute_heading(from_pos, to_pos):
    direction = to_pos - from_pos
    dist = np.linalg.norm(direction)
    if dist < 1e-8:
        return np.zeros(3)
    return direction / dist


# ============================================================
# Reward calculator
# ============================================================

class RewardCalculator:
    def __init__(self, step_fuel_penalty=0.01, step_time_penalty=0.001,
                 collision_penalty=-1000.0, orbital_insertion_bonus=100.0,
                 transmission_bonus=50.0, proximity_scale=0.1):
        self.step_fuel_penalty = step_fuel_penalty
        self.step_time_penalty = step_time_penalty
        self.collision_penalty = collision_penalty
        self.orbital_insertion_bonus = orbital_insertion_bonus
        self.transmission_bonus = transmission_bonus
        self.proximity_scale = proximity_scale

    def compute(self, state):
        info = {}
        total = 0.0

        fuel_penalty = -(self.step_fuel_penalty * state.get("fuel_used", 0.0))
        time_penalty = -self.step_time_penalty
        info["reward_step_fuel"] = fuel_penalty
        info["reward_step_time"] = time_penalty
        total += fuel_penalty + time_penalty

        for biosig in state.get("new_biosignatures", []):
            reward = BIOSIGNATURE_REWARDS.get(biosig, 0.0)
            info[f"reward_detect_{biosig}"] = reward
            total += reward

        for biosig in state.get("new_transmissions", []):
            info[f"reward_transmit_{biosig}"] = self.transmission_bonus
            total += self.transmission_bonus

        if state.get("orbital_insertion", False):
            info["reward_orbital_insertion"] = self.orbital_insertion_bonus
            total += self.orbital_insertion_bonus

        if state.get("collision", False) or state.get("out_of_bounds", False):
            info["reward_collision"] = self.collision_penalty
            total += self.collision_penalty

        min_dist = state.get("min_target_distance", 50.0)
        if min_dist < 5.0:
            proximity_reward = self.proximity_scale * (1.0 / (min_dist + 0.1) - 1.0 / 5.1)
            proximity_reward = max(0.0, proximity_reward)
            info["reward_proximity"] = proximity_reward
            total += proximity_reward

        info["reward_total"] = total
        return total, info


# ============================================================
# Action-space wrappers
# ============================================================

class FlattenMultiDiscreteToDiscrete(gym.ActionWrapper):
    """Converts MultiDiscrete([5,3,3,4,2]) -> Discrete(360) for DQN."""

    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.action_space, spaces.MultiDiscrete)
        self.nvec = env.action_space.nvec
        self.total_actions = int(np.prod(self.nvec))
        self.action_space = spaces.Discrete(self.total_actions)

    def action(self, flat_action):
        multi_action = np.zeros(len(self.nvec), dtype=np.int64)
        remaining = flat_action
        for i in range(len(self.nvec) - 1, -1, -1):
            multi_action[i] = remaining % self.nvec[i]
            remaining //= self.nvec[i]
        return multi_action

    def reverse_action(self, multi_action):
        flat = 0
        multiplier = 1
        for i in range(len(self.nvec) - 1, -1, -1):
            flat += int(multi_action[i]) * multiplier
            multiplier *= self.nvec[i]
        return flat


# Alias: roll was removed from the base env, so full flatten is equivalent.
ReducedDiscreteWrapper = FlattenMultiDiscreteToDiscrete


# ============================================================
# AstroExploration Gymnasium Environment
# ============================================================

class AstroExplorationEnv(gym.Env):
    """Spacecraft navigates a simplified solar system to detect and transmit
    biosignatures from Mars, Europa, and Enceladus.

    Observation Space (26-dim continuous):
        [0:3]   Spacecraft position (x, y, z) in AU
        [3:6]   Spacecraft velocity (vx, vy, vz)
        [6]     Normalized distance to Mars
        [7:10]  Heading to Mars (unit vector)
        [10]    Normalized distance to Europa
        [11:14] Heading to Europa (unit vector)
        [14]    Normalized distance to Enceladus
        [15:18] Heading to Enceladus (unit vector)
        [18]    Fuel level [0, 1]
        [19]    Battery level [0, 1]
        [20]    SNR biosignature signal [0, 1]
        [21]    Biosignatures found / 3
        [22]    Biosignatures transmitted / 3
        [23]    Agent yaw / pi  [-1, 1]
        [24]    Normalized distance to Sun [0, 1]
        [25]    Active instrument / 3  [0, 1]

    Action Space (MultiDiscrete [5, 3, 3, 4, 2]):
        [0] Thrust:      {0, 0.25, 0.5, 0.75, 1.0}
        [1] Pitch:       {-5, 0, +5} degrees
        [2] Yaw:         {-5, 0, +5} degrees
        [3] Instrument:  {None, Spectrometer, ThermalImager, Drill}
        [4] Communication: {Off, Transmit}
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    THRUST_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]
    ROTATION_DELTAS = [-5.0, 0.0, 5.0]

    MAX_THRUST = 0.001
    FUEL_CONSUMPTION_RATE = 0.0005
    BATTERY_DRAIN_RATE = 0.00005
    SOLAR_RECHARGE_RANGE = 1.5
    SOLAR_RECHARGE_RATE = 0.0001
    DT = 1.0

    SNR_DETECTION_THRESHOLD = 0.5
    BIOSIG_SUCCESS_COUNT = 3
    MAX_DISTANCE = 50.0

    def __init__(self, render_mode=None, max_episode_steps=100_000,
                 reward_kwargs=None):
        super().__init__()
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps

        obs_low = np.array(
            [-50, -50, -50, -10, -10, -10,
             0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1,
             0, 0, 0, 0, 0, -1, 0, 0], dtype=np.float32)
        obs_high = np.array(
            [50, 50, 50, 10, 10, 10,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([5, 3, 3, 4, 2])

        self.reward_calculator = RewardCalculator(**(reward_kwargs or {}))

        self._renderer = None
        self.position = None
        self.velocity = None
        self.orientation = None
        self.fuel = None
        self.battery = None
        self.current_step = None
        self.sim_time = None
        self.biosignatures_found = None
        self.biosignatures_transmitted = None
        self.active_instrument = None
        self.body_positions = None
        self.cumulative_reward = None
        self.trajectory = None
        self._last_thrust_level = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        earth_data = CELESTIAL_BODIES["Earth"]
        self.position = np.array([earth_data["orbit_radius"], 0.0, 0.0],
                                 dtype=np.float64)
        v_earth = 2.0 * np.pi * earth_data["orbit_radius"] / earth_data["orbit_period"]
        self.velocity = np.array([0.0, v_earth, 0.0], dtype=np.float64)

        self.orientation = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.fuel = 1.0
        self.battery = 1.0
        self.current_step = 0
        self.sim_time = 0.0
        self.biosignatures_found = set()
        self.biosignatures_transmitted = set()
        self.active_instrument = 0
        self.cumulative_reward = 0.0
        self.trajectory = deque([self.position.copy()], maxlen=500)

        if self.np_random is not None:
            for name, body in CELESTIAL_BODIES.items():
                if name != "Sun":
                    body["initial_angle"] = (
                        body["initial_angle"] + self.np_random.uniform(-0.1, 0.1))

        self._update_body_positions()
        return self._get_obs(), self._get_info()

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        thrust_idx, pitch_idx, yaw_idx, instrument_idx, comm_idx = action
        thrust_level = self.THRUST_LEVELS[thrust_idx]
        self._last_thrust_level = thrust_level
        pitch_delta = np.radians(self.ROTATION_DELTAS[pitch_idx])
        yaw_delta = np.radians(self.ROTATION_DELTAS[yaw_idx])
        self.active_instrument = instrument_idx

        self.orientation[0] += pitch_delta
        self.orientation[1] += yaw_delta

        fuel_used = 0.0
        if thrust_level > 0 and self.fuel > 0:
            forward = orientation_to_direction(*self.orientation)
            thrust_accel = forward * thrust_level * self.MAX_THRUST
            self.velocity += thrust_accel * self.DT
            fuel_used = thrust_level * self.FUEL_CONSUMPTION_RATE
            self.fuel = max(0.0, self.fuel - fuel_used)

        total_grav_accel = np.zeros(3)
        for name, body in CELESTIAL_BODIES.items():
            body_pos = (np.zeros(3) if name == "Sun"
                        else self.body_positions[name])
            total_grav_accel += gravitational_acceleration(
                self.position, body_pos, body["mass"])

        self.velocity += total_grav_accel * self.DT
        self.position += self.velocity * self.DT

        self.sim_time += self.DT
        self._update_body_positions()

        self.battery -= self.BATTERY_DRAIN_RATE
        sun_dist = np.linalg.norm(self.position)
        if sun_dist < self.SOLAR_RECHARGE_RANGE:
            self.battery += self.SOLAR_RECHARGE_RATE
        self.battery = np.clip(self.battery, 0.0, 1.0)

        snr = self._compute_snr()

        new_biosignatures = []
        if instrument_idx > 0 and snr >= self.SNR_DETECTION_THRESHOLD:
            new_biosignatures = self._attempt_detection(instrument_idx)

        new_transmissions = []
        if comm_idx == 1:
            for biosig in list(self.biosignatures_found):
                if biosig not in self.biosignatures_transmitted:
                    self.biosignatures_transmitted.add(biosig)
                    new_transmissions.append(biosig)

        orbital_insertion = self._check_orbital_insertion()

        collision = self._check_collision()
        out_of_bounds = np.linalg.norm(self.position) > self.MAX_DISTANCE
        resource_depleted = self.fuel <= 0 or self.battery <= 0
        success = len(self.biosignatures_transmitted) >= self.BIOSIG_SUCCESS_COUNT

        self.current_step += 1
        truncated = self.current_step >= self.max_episode_steps
        terminated = collision or out_of_bounds or resource_depleted or success

        min_target_dist = self._min_target_distance()

        reward_state = {
            "fuel_used": fuel_used,
            "collision": collision,
            "out_of_bounds": out_of_bounds,
            "new_biosignatures": new_biosignatures,
            "new_transmissions": new_transmissions,
            "orbital_insertion": orbital_insertion,
            "min_target_distance": min_target_dist,
        }
        reward, reward_info = self.reward_calculator.compute(reward_state)
        self.cumulative_reward += reward

        self.trajectory.append(self.position.copy())

        obs = self._get_obs()
        info = self._get_info()
        info.update(reward_info)
        info["success"] = success
        info["collision"] = collision
        info["out_of_bounds"] = out_of_bounds
        info["resource_depleted"] = resource_depleted

        return obs, reward, terminated, truncated, info

    # ------ internal helpers ------

    def _update_body_positions(self):
        self.body_positions = {}
        for name, body in CELESTIAL_BODIES.items():
            if name == "Sun":
                self.body_positions[name] = np.zeros(3)
            elif body["parent"] == "Sun":
                self.body_positions[name] = compute_orbital_position(
                    body["orbit_radius"], body["orbit_period"],
                    self.sim_time, body["initial_angle"])
        for name, body in CELESTIAL_BODIES.items():
            if body["parent"] and body["parent"] != "Sun":
                parent_pos = self.body_positions.get(body["parent"], np.zeros(3))
                self.body_positions[name] = compute_orbital_position(
                    body["orbit_radius"], body["orbit_period"],
                    self.sim_time, body["initial_angle"],
                    parent_pos=parent_pos)

    def _get_obs(self):
        obs = np.zeros(26, dtype=np.float32)
        obs[0:3] = self.position.astype(np.float32)
        obs[3:6] = np.clip(self.velocity, -10, 10).astype(np.float32)

        for i, target_name in enumerate(TARGET_BODIES):
            target_pos = self.body_positions.get(target_name, np.zeros(3))
            dist = np.linalg.norm(target_pos - self.position)
            heading = compute_heading(self.position, target_pos)
            base_idx = 6 + i * 4
            obs[base_idx] = normalize_distance(dist)
            obs[base_idx + 1: base_idx + 4] = heading.astype(np.float32)

        obs[18] = self.fuel
        obs[19] = self.battery
        obs[20] = self._compute_snr()
        obs[21] = len(self.biosignatures_found) / self.BIOSIG_SUCCESS_COUNT
        obs[22] = len(self.biosignatures_transmitted) / self.BIOSIG_SUCCESS_COUNT
        obs[23] = float(np.clip(self.orientation[1] / np.pi, -1.0, 1.0))
        sun_dist = np.linalg.norm(self.position)
        obs[24] = normalize_distance(sun_dist)
        obs[25] = self.active_instrument / 3.0
        return obs

    def _get_info(self):
        return {
            "step": self.current_step,
            "fuel": self.fuel,
            "battery": self.battery,
            "position": self.position.copy(),
            "velocity": self.velocity.copy(),
            "biosig_found": list(self.biosignatures_found),
            "biosig_transmitted": list(self.biosignatures_transmitted),
            "cumulative_reward": self.cumulative_reward,
            "active_instrument": INSTRUMENTS[self.active_instrument]["name"],
        }

    def _compute_snr(self):
        max_snr = 0.0
        for target_name in TARGET_BODIES:
            body = CELESTIAL_BODIES[target_name]
            target_pos = self.body_positions.get(target_name, np.zeros(3))
            dist = np.linalg.norm(target_pos - self.position)
            zone_radius = body["detection_zone_radius"]
            if zone_radius > 0 and dist < zone_radius:
                snr = 1.0 - (dist / zone_radius)
                max_snr = max(max_snr, snr)
        return float(max_snr)

    def _attempt_detection(self, instrument_idx):
        detected = []
        instrument = INSTRUMENTS[instrument_idx]
        for target_name in TARGET_BODIES:
            body = CELESTIAL_BODIES[target_name]
            target_pos = self.body_positions.get(target_name, np.zeros(3))
            dist = np.linalg.norm(target_pos - self.position)
            if dist < body["detection_zone_radius"]:
                for biosig in body["biosignatures"]:
                    if (biosig in instrument["detects"]
                            and biosig not in self.biosignatures_found):
                        self.biosignatures_found.add(biosig)
                        detected.append(biosig)
        return detected

    def _check_collision(self):
        for name, body in CELESTIAL_BODIES.items():
            body_pos = self.body_positions.get(name, np.zeros(3))
            dist = np.linalg.norm(body_pos - self.position)
            collision_radius = max(body["radius"] * 10, 0.001)
            if dist < collision_radius:
                return True
        return False

    def _check_orbital_insertion(self):
        for target_name in TARGET_BODIES:
            target_pos = self.body_positions.get(target_name, np.zeros(3))
            dist = np.linalg.norm(target_pos - self.position)
            rel_vel = np.linalg.norm(self.velocity)
            if dist < CELESTIAL_BODIES[target_name]["detection_zone_radius"] * 0.5:
                if rel_vel < 0.01:
                    return True
        return False

    def _min_target_distance(self):
        min_dist = float("inf")
        for target_name in TARGET_BODIES:
            target_pos = self.body_positions.get(target_name, np.zeros(3))
            dist = np.linalg.norm(target_pos - self.position)
            min_dist = min(min_dist, dist)
        return min_dist

    def render(self):
        if self.render_mode is None:
            return None
        if self._renderer is None:
            from environment.rendering import PygameRenderer
            self._renderer = PygameRenderer(self)
        return self._renderer.render_frame(self._get_render_state())

    def _get_render_state(self):
        return {
            "position": self.position.copy(),
            "velocity": self.velocity.copy(),
            "orientation": self.orientation.copy(),
            "fuel": self.fuel,
            "battery": self.battery,
            "snr": self._compute_snr(),
            "active_instrument": INSTRUMENTS[self.active_instrument]["name"],
            "biosig_found": list(self.biosignatures_found),
            "biosig_transmitted": list(self.biosignatures_transmitted),
            "body_positions": {k: v.copy() for k, v in self.body_positions.items()},
            "trajectory": list(self.trajectory),
            "current_step": self.current_step,
            "max_steps": self.max_episode_steps,
            "cumulative_reward": self.cumulative_reward,
            "thrust_level": self._last_thrust_level,
        }

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
