"""AstroExploration Gymnasium Environment.

A spacecraft navigates a simplified solar system to detect biosignatures
on Mars, Europa, and Enceladus using various scientific instruments.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from astro_env.celestial_bodies import CELESTIAL_BODIES, TARGET_BODIES, INSTRUMENTS
from astro_env.physics import (
    gravitational_acceleration,
    orientation_to_direction,
    compute_orbital_position,
    normalize_distance,
    compute_heading,
)
from astro_env.rewards import RewardCalculator


class AstroExplorationEnv(gym.Env):
    """Custom Gymnasium environment for astrobiological exploration.

    The agent controls a spacecraft starting from Earth, navigating through
    the solar system to detect and transmit biosignatures from target bodies.

    Observation Space (23-dim continuous):
        [0:3]   - Spacecraft position (x, y, z) in AU
        [3:6]   - Spacecraft velocity (vx, vy, vz)
        [6]     - Normalized distance to Mars
        [7:10]  - Heading to Mars (unit vector)
        [10]    - Normalized distance to Europa
        [11:14] - Heading to Europa (unit vector)
        [14]    - Normalized distance to Enceladus
        [15:18] - Heading to Enceladus (unit vector)
        [18]    - Fuel level [0, 1]
        [19]    - Battery level [0, 1]
        [20]    - SNR biosignature signal [0, 1]
        [21]    - Biosignatures found / 3
        [22]    - Biosignatures transmitted / 3

    Action Space (MultiDiscrete [5, 3, 3, 3, 4, 2]):
        [0] Thrust: {0, 0.25, 0.5, 0.75, 1.0}
        [1] Pitch:  {-5, 0, +5} degrees
        [2] Yaw:    {-5, 0, +5} degrees
        [3] Roll:   {-5, 0, +5} degrees
        [4] Instrument: {None, Spectrometer, ThermalImager, Drill}
        [5] Communication: {Off, Transmit}
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    # Action encoding
    THRUST_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]
    ROTATION_DELTAS = [-5.0, 0.0, 5.0]  # degrees

    # Physics constants
    MAX_THRUST = 0.001          # AU/day^2 max acceleration from thrust
    FUEL_CONSUMPTION_RATE = 0.0005  # fuel consumed per unit thrust per step
    BATTERY_DRAIN_RATE = 0.00005    # battery drain per step
    SOLAR_RECHARGE_RANGE = 1.5      # AU - recharge battery within this distance of Sun
    SOLAR_RECHARGE_RATE = 0.0001    # battery recharged per step when near Sun
    DT = 1.0                        # time step in days

    # Detection parameters
    SNR_DETECTION_THRESHOLD = 0.5   # minimum SNR to detect biosignatures
    BIOSIG_SUCCESS_COUNT = 3        # number of distinct biosignatures to transmit for success

    # Boundary
    MAX_DISTANCE = 50.0  # AU from origin

    def __init__(self, render_mode=None, max_episode_steps=100_000):
        super().__init__()
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps

        # Observation space: 23-dimensional continuous
        obs_low = np.array(
            [-50, -50, -50,      # position
             -10, -10, -10,      # velocity
             0,                  # mars dist
             -1, -1, -1,         # mars heading
             0,                  # europa dist
             -1, -1, -1,         # europa heading
             0,                  # enceladus dist
             -1, -1, -1,         # enceladus heading
             0, 0,               # fuel, battery
             0,                  # snr
             0, 0],              # biosig found, transmitted
            dtype=np.float32,
        )
        obs_high = np.array(
            [50, 50, 50,
             10, 10, 10,
             1,
             1, 1, 1,
             1,
             1, 1, 1,
             1,
             1, 1, 1,
             1, 1,
             1,
             1, 1],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        # Action space: MultiDiscrete [thrust, pitch, yaw, roll, instrument, comm]
        self.action_space = spaces.MultiDiscrete([5, 3, 3, 3, 4, 2])

        # Reward calculator
        self.reward_calculator = RewardCalculator()

        # Renderer (lazy init)
        self._renderer = None

        # State variables (initialized in reset)
        self.position = None
        self.velocity = None
        self.orientation = None  # [pitch, yaw, roll] in radians
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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Spacecraft starts at Earth's position
        earth_data = CELESTIAL_BODIES["Earth"]
        self.position = np.array([
            earth_data["orbit_radius"], 0.0, 0.0
        ], dtype=np.float64)

        # Earth's orbital velocity (tangential, circular orbit)
        # v = 2*pi*r / T in AU/day
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
        self.trajectory = [self.position.copy()]

        # Randomize initial orbital angles slightly
        if self.np_random is not None:
            for name, body in CELESTIAL_BODIES.items():
                if name != "Sun":
                    body["initial_angle"] = (
                        body["initial_angle"] + self.np_random.uniform(-0.1, 0.1)
                    )

        # Compute initial body positions
        self._update_body_positions()

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        # Decode sub-actions
        thrust_idx, pitch_idx, yaw_idx, roll_idx, instrument_idx, comm_idx = action
        thrust_level = self.THRUST_LEVELS[thrust_idx]
        pitch_delta = np.radians(self.ROTATION_DELTAS[pitch_idx])
        yaw_delta = np.radians(self.ROTATION_DELTAS[yaw_idx])
        roll_delta = np.radians(self.ROTATION_DELTAS[roll_idx])
        self.active_instrument = instrument_idx

        # Update orientation
        self.orientation[0] += pitch_delta
        self.orientation[1] += yaw_delta
        self.orientation[2] += roll_delta

        # Apply thrust along forward direction
        fuel_used = 0.0
        if thrust_level > 0 and self.fuel > 0:
            forward = orientation_to_direction(*self.orientation)
            thrust_accel = forward * thrust_level * self.MAX_THRUST
            self.velocity += thrust_accel * self.DT
            fuel_used = thrust_level * self.FUEL_CONSUMPTION_RATE
            self.fuel = max(0.0, self.fuel - fuel_used)

        # Compute gravitational acceleration from all bodies
        total_grav_accel = np.zeros(3)
        for name, body in CELESTIAL_BODIES.items():
            if name == "Sun":
                body_pos = np.zeros(3)
            else:
                body_pos = self.body_positions[name]
            total_grav_accel += gravitational_acceleration(
                self.position, body_pos, body["mass"]
            )

        # Euler integration
        self.velocity += total_grav_accel * self.DT
        self.position += self.velocity * self.DT

        # Update simulation time and celestial body positions
        self.sim_time += self.DT
        self._update_body_positions()

        # Battery management
        self.battery -= self.BATTERY_DRAIN_RATE
        sun_dist = np.linalg.norm(self.position)
        if sun_dist < self.SOLAR_RECHARGE_RANGE:
            self.battery += self.SOLAR_RECHARGE_RATE
        self.battery = np.clip(self.battery, 0.0, 1.0)

        # Compute SNR based on proximity to target bodies
        snr = self._compute_snr()

        # Biosignature detection
        new_biosignatures = []
        if instrument_idx > 0 and snr >= self.SNR_DETECTION_THRESHOLD:
            new_biosignatures = self._attempt_detection(instrument_idx)

        # Communication (transmit found biosignatures)
        new_transmissions = []
        if comm_idx == 1:
            for biosig in list(self.biosignatures_found):
                if biosig not in self.biosignatures_transmitted:
                    self.biosignatures_transmitted.add(biosig)
                    new_transmissions.append(biosig)

        # Check orbital insertion (simplified: close and low relative velocity)
        orbital_insertion = self._check_orbital_insertion()

        # Check terminal conditions
        collision = self._check_collision()
        out_of_bounds = np.linalg.norm(self.position) > self.MAX_DISTANCE
        resource_depleted = self.fuel <= 0 or self.battery <= 0
        success = len(self.biosignatures_transmitted) >= self.BIOSIG_SUCCESS_COUNT

        self.current_step += 1
        truncated = self.current_step >= self.max_episode_steps
        terminated = collision or out_of_bounds or resource_depleted or success

        # Compute distances to targets
        min_target_dist = self._min_target_distance()

        # Compute reward
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

        # Store trajectory for rendering
        self.trajectory.append(self.position.copy())
        if len(self.trajectory) > 500:
            self.trajectory.pop(0)

        obs = self._get_obs()
        info = self._get_info()
        info.update(reward_info)
        info["success"] = success
        info["collision"] = collision
        info["out_of_bounds"] = out_of_bounds
        info["resource_depleted"] = resource_depleted

        return obs, reward, terminated, truncated, info

    def _update_body_positions(self):
        """Update all celestial body positions based on current sim_time."""
        self.body_positions = {}
        # First pass: compute positions for bodies orbiting the Sun
        for name, body in CELESTIAL_BODIES.items():
            if name == "Sun":
                self.body_positions[name] = np.zeros(3)
            elif body["parent"] == "Sun":
                self.body_positions[name] = compute_orbital_position(
                    body["orbit_radius"],
                    body["orbit_period"],
                    self.sim_time,
                    body["initial_angle"],
                )
        # Second pass: compute positions for moons (orbit their parent)
        for name, body in CELESTIAL_BODIES.items():
            if body["parent"] and body["parent"] != "Sun":
                parent_pos = self.body_positions.get(body["parent"], np.zeros(3))
                self.body_positions[name] = compute_orbital_position(
                    body["orbit_radius"],
                    body["orbit_period"],
                    self.sim_time,
                    body["initial_angle"],
                    parent_pos=parent_pos,
                )

    def _get_obs(self) -> np.ndarray:
        """Build the 23-dimensional observation vector."""
        obs = np.zeros(23, dtype=np.float32)

        # Position and velocity
        obs[0:3] = self.position.astype(np.float32)
        obs[3:6] = np.clip(self.velocity, -10, 10).astype(np.float32)

        # Target body data
        for i, target_name in enumerate(TARGET_BODIES):
            target_pos = self.body_positions.get(target_name, np.zeros(3))
            dist = np.linalg.norm(target_pos - self.position)
            heading = compute_heading(self.position, target_pos)

            base_idx = 6 + i * 4
            obs[base_idx] = normalize_distance(dist)
            obs[base_idx + 1: base_idx + 4] = heading.astype(np.float32)

        # Resources
        obs[18] = self.fuel
        obs[19] = self.battery

        # SNR
        obs[20] = self._compute_snr()

        # Mission progress
        obs[21] = len(self.biosignatures_found) / self.BIOSIG_SUCCESS_COUNT
        obs[22] = len(self.biosignatures_transmitted) / self.BIOSIG_SUCCESS_COUNT

        return obs

    def _get_info(self) -> dict:
        """Build info dictionary."""
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

    def _compute_snr(self) -> float:
        """Compute signal-to-noise ratio based on proximity to biosignature zones."""
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

    def _attempt_detection(self, instrument_idx: int) -> list:
        """Try to detect biosignatures with the active instrument."""
        detected = []
        instrument = INSTRUMENTS[instrument_idx]

        for target_name in TARGET_BODIES:
            body = CELESTIAL_BODIES[target_name]
            target_pos = self.body_positions.get(target_name, np.zeros(3))
            dist = np.linalg.norm(target_pos - self.position)

            if dist < body["detection_zone_radius"]:
                for biosig in body["biosignatures"]:
                    if (
                        biosig in instrument["detects"]
                        and biosig not in self.biosignatures_found
                    ):
                        self.biosignatures_found.add(biosig)
                        detected.append(biosig)
        return detected

    def _check_collision(self) -> bool:
        """Check if spacecraft collided with any celestial body."""
        for name, body in CELESTIAL_BODIES.items():
            body_pos = self.body_positions.get(name, np.zeros(3))
            dist = np.linalg.norm(body_pos - self.position)
            # Use exaggerated collision radius for gameplay
            collision_radius = max(body["radius"] * 10, 0.001)
            if dist < collision_radius:
                return True
        return False

    def _check_orbital_insertion(self) -> bool:
        """Check if spacecraft achieved orbital insertion around a target body."""
        for target_name in TARGET_BODIES:
            target_pos = self.body_positions.get(target_name, np.zeros(3))
            dist = np.linalg.norm(target_pos - self.position)
            rel_vel = np.linalg.norm(self.velocity)
            # Simplified: close enough and slow enough
            if dist < CELESTIAL_BODIES[target_name]["detection_zone_radius"] * 0.5:
                if rel_vel < 0.01:
                    return True
        return False

    def _min_target_distance(self) -> float:
        """Get minimum distance to any target body."""
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
            from visualization.renderer import PygameRenderer
            self._renderer = PygameRenderer(self)

        return self._renderer.render_frame(self._get_render_state())

    def _get_render_state(self) -> dict:
        """Package current state for the renderer."""
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
            "trajectory": [p.copy() for p in self.trajectory],
            "current_step": self.current_step,
            "max_steps": self.max_episode_steps,
            "cumulative_reward": self.cumulative_reward,
            "thrust_level": 0.0,  # Updated from last action
        }

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
