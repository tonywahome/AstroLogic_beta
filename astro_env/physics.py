"""Physics helpers for orbital mechanics simulation.

Uses normalized units: distances in AU, time in days, masses in solar masses.
G = 4*pi^2 AU^3 / (solar_mass * year^2), converted to day-based units.
"""

import numpy as np

# Gravitational constant in AU^3 / (solar_mass * day^2)
G_NORMALIZED = 4.0 * np.pi**2 / 365.25**2


def gravitational_acceleration(
    pos: np.ndarray, body_pos: np.ndarray, body_mass: float
) -> np.ndarray:
    """Compute gravitational acceleration on spacecraft from a single body.

    F/m = G * M / r^2 in the direction toward the body.
    """
    direction = body_pos - pos
    distance = np.linalg.norm(direction)
    if distance < 1e-8:
        return np.zeros(3)
    return G_NORMALIZED * body_mass / (distance**2) * (direction / distance)


def orientation_to_direction(pitch: float, yaw: float, roll: float) -> np.ndarray:
    """Convert Euler angles (radians) to a unit forward direction vector.

    Convention: forward is initially along +x axis.
    Yaw rotates around z-axis, pitch rotates around y-axis.
    Roll does not affect the forward direction.
    """
    # Yaw (rotation around z-axis)
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    # Pitch (rotation around y-axis)
    cos_pitch, sin_pitch = np.cos(pitch), np.sin(pitch)

    forward = np.array([
        cos_pitch * cos_yaw,
        cos_pitch * sin_yaw,
        sin_pitch,
    ])
    return forward / (np.linalg.norm(forward) + 1e-8)


def compute_orbital_position(
    orbit_radius: float,
    orbit_period: float,
    time: float,
    initial_angle: float,
    parent_pos: np.ndarray | None = None,
) -> np.ndarray:
    """Compute position on a circular orbit at a given time.

    Returns 3D position in the ecliptic plane (z=0 for simplicity).
    """
    if orbit_period <= 0 or orbit_radius <= 0:
        if parent_pos is not None:
            return parent_pos.copy()
        return np.zeros(3)

    angle = initial_angle + 2.0 * np.pi * time / orbit_period
    pos = np.array([
        orbit_radius * np.cos(angle),
        orbit_radius * np.sin(angle),
        0.0,
    ])
    if parent_pos is not None:
        pos += parent_pos
    return pos


def normalize_distance(distance: float, max_distance: float = 50.0) -> float:
    """Normalize distance to [0, 1] range where closer = higher value."""
    return max(0.0, 1.0 - distance / max_distance)


def compute_heading(from_pos: np.ndarray, to_pos: np.ndarray) -> np.ndarray:
    """Compute unit heading vector from one position to another."""
    direction = to_pos - from_pos
    dist = np.linalg.norm(direction)
    if dist < 1e-8:
        return np.zeros(3)
    return direction / dist
