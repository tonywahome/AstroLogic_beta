"""Pygame renderer for the AstroExploration environment.

Renders a 2D top-down view of the solar system with the spacecraft,
celestial bodies, orbital paths, biosignature zones, and trajectory trail.
"""

import numpy as np
import pygame

from astro_env.celestial_bodies import CELESTIAL_BODIES, TARGET_BODIES
from visualization.colors import (
    SPACE_BLACK, STAR_WHITE, SUN_YELLOW, ORBIT_PATH, THRUST_ORANGE,
    TRAJECTORY_WHITE, HUD_WHITE, BODY_COLORS,
    BIOSIG_ZONE_WATER, BIOSIG_ZONE_ORGANIC, BIOSIG_ZONE_INTELLIGENCE,
)
from visualization.ui_overlay import UIOverlay


class PygameRenderer:
    """Renders the AstroExploration environment using pygame."""

    SCREEN_W = 1200
    SCREEN_H = 800

    # Visual scale: pixels per AU
    BASE_SCALE = 60.0
    MIN_SCALE = 5.0
    MAX_SCALE = 500.0

    # Minimum visual sizes
    MIN_BODY_RADIUS = 3
    SPACECRAFT_SIZE = 8
    SUN_VISUAL_RADIUS = 12

    def __init__(self, env):
        self.env = env
        pygame.init()
        self.screen = pygame.display.set_mode((self.SCREEN_W, self.SCREEN_H))
        pygame.display.set_caption("AstroLogic - Astrobiological Exploration")
        self.clock = pygame.time.Clock()
        self.ui = UIOverlay(self.SCREEN_W, self.SCREEN_H)

        # Generate static star field with pre-computed brightness
        self.stars = []
        for _ in range(200):
            x = np.random.randint(0, self.SCREEN_W)
            y = np.random.randint(0, self.SCREEN_H)
            b = np.random.randint(150, 255)
            self.stars.append((x, y, (b, b, min(b + 10, 255))))

        # Camera state
        self.camera_x = 0.0
        self.camera_y = 0.0
        self.scale = self.BASE_SCALE
        self._target_scale = self.BASE_SCALE

        # Cached font for body labels
        self._label_font = pygame.font.SysFont("consolas", 11)

        # Cache for biosignature zone surfaces keyed by (name, radius_px)
        self._zone_surface_cache = {}

        # Pre-computed trajectory color lookup table (index = alpha 0-255)
        self._traj_colors = [
            (
                min(255, TRAJECTORY_WHITE[0] * i // 255),
                min(255, TRAJECTORY_WHITE[1] * i // 255),
                min(255, TRAJECTORY_WHITE[2] * i // 255),
            )
            for i in range(256)
        ]

    def render_frame(self, state: dict):
        """Render one frame. Returns the surface as rgb_array if needed."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None

        # Update camera to follow spacecraft
        pos = state.get("position", np.zeros(3))
        self.camera_x = pos[0]
        self.camera_y = pos[1]

        # Auto-scale: fit spacecraft and nearest target in view
        self._auto_scale(state)

        # Draw layers
        self.screen.fill(SPACE_BLACK)
        self._draw_stars()
        self._draw_orbital_paths(state)
        self._draw_biosignature_zones(state)
        self._draw_celestial_bodies(state)
        self._draw_trajectory(state)
        self._draw_spacecraft(state)
        self.ui.draw(self.screen, state)

        pygame.display.flip()
        self.clock.tick(30)

        if self.env.render_mode == "rgb_array":
            return np.transpose(
                pygame.surfarray.array3d(self.screen), axes=(1, 0, 2)
            )
        return None

    def _world_to_screen(self, wx: float, wy: float) -> tuple[int, int]:
        """Convert world coordinates (AU) to screen pixel coordinates."""
        sx = int(self.SCREEN_W / 2 + (wx - self.camera_x) * self.scale)
        sy = int(self.SCREEN_H / 2 - (wy - self.camera_y) * self.scale)
        return sx, sy

    def _auto_scale(self, state):
        """Adjust zoom to keep spacecraft and at least one target visible."""
        pos = state.get("position", np.zeros(3))
        body_positions = state.get("body_positions", {})

        # Find distance to nearest target
        min_dist = float("inf")
        for target in TARGET_BODIES:
            if target in body_positions:
                dist = np.linalg.norm(body_positions[target][:2] - pos[:2])
                min_dist = min(min_dist, dist)

        # Scale so the nearest target fits in ~60% of screen
        if min_dist > 0.1:
            desired_scale = (self.SCREEN_W * 0.3) / min_dist
            self._target_scale = np.clip(desired_scale, self.MIN_SCALE, self.MAX_SCALE)
        else:
            self._target_scale = self.MAX_SCALE
        # Smooth lerp toward target scale to avoid jarring zoom jumps
        self.scale += (self._target_scale - self.scale) * 0.1

    def _draw_stars(self):
        """Draw static star background."""
        for sx, sy, color in self.stars:
            self.screen.set_at((sx, sy), color)

    def _draw_orbital_paths(self, state):
        """Draw orbital path circles for planets."""
        sun_sx, sun_sy = self._world_to_screen(0, 0)

        for name, body in CELESTIAL_BODIES.items():
            if body["parent"] == "Sun" and body["orbit_radius"] > 0:
                radius_px = int(body["orbit_radius"] * self.scale)
                if 2 < radius_px < 2000:
                    if (sun_sx + radius_px >= 0 and sun_sx - radius_px < self.SCREEN_W
                            and sun_sy + radius_px >= 0 and sun_sy - radius_px < self.SCREEN_H):
                        pygame.draw.circle(
                            self.screen, ORBIT_PATH,
                            (sun_sx, sun_sy), radius_px, 1
                        )

    def _draw_biosignature_zones(self, state):
        """Draw semi-transparent biosignature detection zones."""
        body_positions = state.get("body_positions", {})

        for target in TARGET_BODIES:
            if target not in body_positions:
                continue
            body = CELESTIAL_BODIES[target]
            zone_r = body["detection_zone_radius"]
            if zone_r <= 0:
                continue

            tpos = body_positions[target]
            sx, sy = self._world_to_screen(tpos[0], tpos[1])
            radius_px = max(4, int(zone_r * self.scale))

            # Choose zone color based on biosignatures
            if "signs_of_intelligence" in body["biosignatures"]:
                zone_color = BIOSIG_ZONE_INTELLIGENCE
            elif "liquid_water" in body["biosignatures"]:
                zone_color = BIOSIG_ZONE_WATER
            else:
                zone_color = BIOSIG_ZONE_ORGANIC

            # Draw semi-transparent circle (cached by name + radius to avoid per-frame alloc)
            cache_key = (target, radius_px)
            if cache_key not in self._zone_surface_cache:
                surf = pygame.Surface((radius_px * 2, radius_px * 2), pygame.SRCALPHA)
                pygame.draw.circle(surf, zone_color, (radius_px, radius_px), radius_px)
                self._zone_surface_cache[cache_key] = surf
            self.screen.blit(self._zone_surface_cache[cache_key], (sx - radius_px, sy - radius_px))

    def _draw_celestial_bodies(self, state):
        """Draw all celestial bodies."""
        body_positions = state.get("body_positions", {})

        for name, body in CELESTIAL_BODIES.items():
            if name in body_positions:
                bpos = body_positions[name]
            else:
                bpos = np.zeros(3)

            sx, sy = self._world_to_screen(bpos[0], bpos[1])

            # Skip if off-screen (with margin)
            if not (-100 < sx < self.SCREEN_W + 100 and -100 < sy < self.SCREEN_H + 100):
                continue

            color = BODY_COLORS.get(name, (200, 200, 200))

            if name == "Sun":
                # Draw glow
                glow_surface = pygame.Surface(
                    (self.SUN_VISUAL_RADIUS * 6, self.SUN_VISUAL_RADIUS * 6),
                    pygame.SRCALPHA,
                )
                pygame.draw.circle(
                    glow_surface, (255, 200, 50, 25),
                    (self.SUN_VISUAL_RADIUS * 3, self.SUN_VISUAL_RADIUS * 3),
                    self.SUN_VISUAL_RADIUS * 3,
                )
                self.screen.blit(
                    glow_surface,
                    (sx - self.SUN_VISUAL_RADIUS * 3, sy - self.SUN_VISUAL_RADIUS * 3),
                )
                pygame.draw.circle(self.screen, color, (sx, sy), self.SUN_VISUAL_RADIUS)
            else:
                # Planet/moon
                visual_r = max(
                    self.MIN_BODY_RADIUS,
                    int(body["radius"] * self.scale * 500)  # exaggerated
                )
                if name in ("Europa", "Enceladus"):
                    visual_r = max(2, visual_r)

                pygame.draw.circle(self.screen, color, (sx, sy), visual_r)

                # Label
                label = self._label_font.render(name, True, HUD_WHITE)
                self.screen.blit(label, (sx + visual_r + 3, sy - 6))

    def _draw_trajectory(self, state):
        """Draw spacecraft trajectory trail."""
        trajectory = state.get("trajectory", [])
        if len(trajectory) < 2:
            return

        points = []
        for pos in trajectory:
            sx, sy = self._world_to_screen(pos[0], pos[1])
            points.append((sx, sy))

        # Draw with fading alpha using pre-computed color LUT
        n = len(points)
        for i in range(1, n):
            alpha = int(255 * i / n)
            pygame.draw.line(self.screen, self._traj_colors[alpha], points[i - 1], points[i], 1)

    def _draw_spacecraft(self, state):
        """Draw the spacecraft as a triangle with thrust indicator."""
        pos = state.get("position", np.zeros(3))
        orientation = state.get("orientation", np.zeros(3))

        sx, sy = self._world_to_screen(pos[0], pos[1])
        yaw = orientation[1]

        # Triangle pointing in heading direction
        size = self.SPACECRAFT_SIZE
        forward = np.array([np.cos(yaw), -np.sin(yaw)])
        right = np.array([np.sin(yaw), np.cos(yaw)])

        tip = (sx + forward[0] * size, sy + forward[1] * size)
        left_wing = (
            sx - forward[0] * size * 0.5 - right[0] * size * 0.6,
            sy - forward[1] * size * 0.5 - right[1] * size * 0.6,
        )
        right_wing = (
            sx - forward[0] * size * 0.5 + right[0] * size * 0.6,
            sy - forward[1] * size * 0.5 + right[1] * size * 0.6,
        )

        pygame.draw.polygon(self.screen, (255, 255, 255), [tip, left_wing, right_wing])

        # Thrust flame
        thrust = state.get("thrust_level", 0.0)
        if thrust > 0:
            flame_len = size * thrust * 2
            flame_tip = (
                sx - forward[0] * (size * 0.5 + flame_len),
                sy - forward[1] * (size * 0.5 + flame_len),
            )
            pygame.draw.polygon(
                self.screen, THRUST_ORANGE,
                [left_wing, right_wing, flame_tip],
            )

    def close(self):
        """Clean up pygame."""
        pygame.quit()
