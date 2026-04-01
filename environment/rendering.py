"""Pygame 2D renderer and HUD overlay for the AstroExploration environment.

Consolidates: color constants, UI overlay, and the main PygameRenderer.
"""

import numpy as np
import pygame

from environment.custom_env import CELESTIAL_BODIES, TARGET_BODIES

# ============================================================
# Color constants
# ============================================================

SPACE_BLACK = (5, 5, 15)
STAR_WHITE = (200, 200, 220)
SUN_YELLOW = (255, 223, 0)
SUN_GLOW = (255, 200, 50, 30)
EARTH_BLUE = (70, 130, 200)
MARS_RED = (193, 68, 14)
JUPITER_TAN = (201, 176, 131)
SATURN_GOLD = (210, 180, 100)
EUROPA_ICY = (200, 220, 240)
ENCELADUS_PALE = (180, 200, 220)

THRUST_ORANGE = (255, 140, 0)
TRAJECTORY_WHITE = (150, 150, 180)

BIOSIG_ZONE_WATER = (0, 100, 255, 40)
BIOSIG_ZONE_ORGANIC = (0, 200, 50, 40)
BIOSIG_ZONE_INTELLIGENCE = (200, 0, 255, 40)

HUD_GREEN = (0, 220, 80)
HUD_RED = (220, 50, 30)
HUD_BLUE = (60, 140, 255)
HUD_WHITE = (220, 220, 230)
HUD_YELLOW = (255, 220, 50)
HUD_GRAY = (120, 120, 140)

ORBIT_PATH = (40, 40, 60)

BODY_COLORS = {
    "Sun": SUN_YELLOW, "Earth": EARTH_BLUE, "Mars": MARS_RED,
    "Jupiter": JUPITER_TAN, "Europa": EUROPA_ICY,
    "Saturn": SATURN_GOLD, "Enceladus": ENCELADUS_PALE,
}

# ============================================================
# HUD overlay
# ============================================================


class UIOverlay:
    def __init__(self, screen_width, screen_height):
        self.sw = screen_width
        self.sh = screen_height
        self.font_sm = pygame.font.SysFont("consolas", 14)

    def draw(self, surface, state):
        self._draw_resource_bars(surface, state)
        self._draw_instrument_status(surface, state)
        self._draw_biosignatures(surface, state)
        self._draw_mission_info(surface, state)
        self._draw_position_info(surface, state)

    def _draw_resource_bars(self, surface, state):
        x, y = 15, 15
        bar_w, bar_h = 180, 18
        self._draw_label(surface, "FUEL", x, y, HUD_WHITE)
        fuel = state.get("fuel", 0.0)
        fuel_color = self._lerp_color(HUD_RED, HUD_GREEN, fuel)
        self._draw_bar(surface, x + 55, y, bar_w, bar_h, fuel, fuel_color)
        self._draw_label(surface, f"{fuel*100:.1f}%", x + bar_w + 60, y, HUD_WHITE)
        y += 28
        self._draw_label(surface, "BATT", x, y, HUD_WHITE)
        batt = state.get("battery", 0.0)
        self._draw_bar(surface, x + 55, y, bar_w, bar_h, batt, HUD_BLUE)
        self._draw_label(surface, f"{batt*100:.1f}%", x + bar_w + 60, y, HUD_WHITE)

    def _draw_instrument_status(self, surface, state):
        x, y = self.sw - 230, 15
        instrument = state.get("active_instrument", "None")
        color = HUD_YELLOW if instrument != "None" else HUD_GRAY
        self._draw_label(surface, f"INSTRUMENT: {instrument}", x, y, color)
        y += 22
        biosig_found = state.get("biosig_found", [])
        biosig_transmitted = state.get("biosig_transmitted", [])
        self._draw_label(surface,
                         f"FOUND: {len(biosig_found)}  TX: {len(biosig_transmitted)}",
                         x, y, HUD_GREEN)

    def _draw_biosignatures(self, surface, state):
        x, y = 15, self.sh - 100
        self._draw_label(surface, "BIOSIGNATURES", x, y, HUD_WHITE)
        biosig_found = set(state.get("biosig_found", []))
        biosig_transmitted = set(state.get("biosig_transmitted", []))
        names = [("liquid_water", "Water"), ("ice", "Ice"),
                 ("organic_compounds", "Organic"),
                 ("signs_of_intelligence", "Intelligence")]
        for i, (full, short) in enumerate(names):
            y_row = y + 20 + i * 18
            if full in biosig_transmitted:
                color, status = HUD_GREEN, "[TX]"
            elif full in biosig_found:
                color, status = HUD_YELLOW, "[FOUND]"
            else:
                color, status = HUD_GRAY, "[ ]"
            self._draw_label(surface, f"{status} {short}", x + 10, y_row, color)

    def _draw_mission_info(self, surface, state):
        step = state.get("current_step", 0)
        max_steps = state.get("max_steps", 100000)
        reward = state.get("cumulative_reward", 0.0)
        x, y = self.sw // 2 - 100, self.sh - 50
        self._draw_label(surface, f"Step: {step:,} / {max_steps:,}", x, y, HUD_WHITE)
        self._draw_label(surface, f"Reward: {reward:.2f}", x, y + 20, HUD_YELLOW)

    def _draw_position_info(self, surface, state):
        pos = state.get("position", [0, 0, 0])
        vel = state.get("velocity", [0, 0, 0])
        snr = state.get("snr", 0.0)
        x, y = self.sw - 280, self.sh - 70
        vel_mag = np.linalg.norm(vel)
        self._draw_label(surface,
                         f"POS: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) AU",
                         x, y, HUD_GRAY)
        self._draw_label(surface, f"VEL: {vel_mag:.4f} AU/day", x, y + 18, HUD_GRAY)
        snr_color = HUD_GREEN if snr > 0.5 else (HUD_YELLOW if snr > 0.1 else HUD_GRAY)
        self._draw_label(surface, f"SNR: {snr:.3f}", x, y + 36, snr_color)

    def _draw_bar(self, surface, x, y, w, h, fraction, color):
        pygame.draw.rect(surface, (30, 30, 40), (x, y, w, h))
        fill_w = max(0, int(w * min(1.0, fraction)))
        if fill_w > 0:
            pygame.draw.rect(surface, color, (x, y, fill_w, h))
        pygame.draw.rect(surface, HUD_GRAY, (x, y, w, h), 1)

    def _draw_label(self, surface, text, x, y, color):
        rendered = self.font_sm.render(text, True, color)
        surface.blit(rendered, (x, y))

    @staticmethod
    def _lerp_color(color_a, color_b, t):
        t = max(0.0, min(1.0, t))
        return tuple(int(a + (b - a) * t) for a, b in zip(color_a, color_b))


# ============================================================
# Main pygame renderer
# ============================================================


class PygameRenderer:
    SCREEN_W = 1200
    SCREEN_H = 800
    BASE_SCALE = 60.0
    MIN_SCALE = 5.0
    MAX_SCALE = 500.0
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

        # Pre-computed star field with static brightness
        self.stars = []
        for _ in range(200):
            x = np.random.randint(0, self.SCREEN_W)
            y = np.random.randint(0, self.SCREEN_H)
            b = np.random.randint(150, 255)
            self.stars.append((x, y, (b, b, min(b + 10, 255))))

        self.camera_x = 0.0
        self.camera_y = 0.0
        self.scale = self.BASE_SCALE
        self._target_scale = self.BASE_SCALE

        self._label_font = pygame.font.SysFont("consolas", 11)
        self._zone_surface_cache = {}
        self._traj_colors = [
            (min(255, TRAJECTORY_WHITE[0] * i // 255),
             min(255, TRAJECTORY_WHITE[1] * i // 255),
             min(255, TRAJECTORY_WHITE[2] * i // 255))
            for i in range(256)
        ]

    def render_frame(self, state):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None
        pos = state.get("position", np.zeros(3))
        self.camera_x = pos[0]
        self.camera_y = pos[1]
        self._auto_scale(state)

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
            return np.transpose(pygame.surfarray.array3d(self.screen), axes=(1, 0, 2))
        return None

    def _world_to_screen(self, wx, wy):
        sx = int(self.SCREEN_W / 2 + (wx - self.camera_x) * self.scale)
        sy = int(self.SCREEN_H / 2 - (wy - self.camera_y) * self.scale)
        return sx, sy

    def _auto_scale(self, state):
        pos = state.get("position", np.zeros(3))
        body_positions = state.get("body_positions", {})
        min_dist = float("inf")
        for target in TARGET_BODIES:
            if target in body_positions:
                dist = np.linalg.norm(body_positions[target][:2] - pos[:2])
                min_dist = min(min_dist, dist)
        if min_dist > 0.1:
            self._target_scale = np.clip(
                (self.SCREEN_W * 0.3) / min_dist, self.MIN_SCALE, self.MAX_SCALE)
        else:
            self._target_scale = self.MAX_SCALE
        self.scale += (self._target_scale - self.scale) * 0.1

    def _draw_stars(self):
        for sx, sy, color in self.stars:
            self.screen.set_at((sx, sy), color)

    def _draw_orbital_paths(self, state):
        sun_sx, sun_sy = self._world_to_screen(0, 0)
        for name, body in CELESTIAL_BODIES.items():
            if body["parent"] == "Sun" and body["orbit_radius"] > 0:
                radius_px = int(body["orbit_radius"] * self.scale)
                if 2 < radius_px < 2000:
                    if (sun_sx + radius_px >= 0
                            and sun_sx - radius_px < self.SCREEN_W
                            and sun_sy + radius_px >= 0
                            and sun_sy - radius_px < self.SCREEN_H):
                        pygame.draw.circle(self.screen, ORBIT_PATH,
                                           (sun_sx, sun_sy), radius_px, 1)

    def _draw_biosignature_zones(self, state):
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
            if "signs_of_intelligence" in body["biosignatures"]:
                zone_color = BIOSIG_ZONE_INTELLIGENCE
            elif "liquid_water" in body["biosignatures"]:
                zone_color = BIOSIG_ZONE_WATER
            else:
                zone_color = BIOSIG_ZONE_ORGANIC
            cache_key = (target, radius_px)
            if cache_key not in self._zone_surface_cache:
                surf = pygame.Surface((radius_px * 2, radius_px * 2), pygame.SRCALPHA)
                pygame.draw.circle(surf, zone_color, (radius_px, radius_px), radius_px)
                self._zone_surface_cache[cache_key] = surf
            self.screen.blit(self._zone_surface_cache[cache_key],
                             (sx - radius_px, sy - radius_px))

    def _draw_celestial_bodies(self, state):
        body_positions = state.get("body_positions", {})
        for name, body in CELESTIAL_BODIES.items():
            bpos = body_positions.get(name, np.zeros(3))
            sx, sy = self._world_to_screen(bpos[0], bpos[1])
            if not (-100 < sx < self.SCREEN_W + 100
                    and -100 < sy < self.SCREEN_H + 100):
                continue
            color = BODY_COLORS.get(name, (200, 200, 200))
            if name == "Sun":
                glow_surface = pygame.Surface(
                    (self.SUN_VISUAL_RADIUS * 6, self.SUN_VISUAL_RADIUS * 6),
                    pygame.SRCALPHA)
                pygame.draw.circle(
                    glow_surface, (255, 200, 50, 25),
                    (self.SUN_VISUAL_RADIUS * 3, self.SUN_VISUAL_RADIUS * 3),
                    self.SUN_VISUAL_RADIUS * 3)
                self.screen.blit(glow_surface,
                                 (sx - self.SUN_VISUAL_RADIUS * 3,
                                  sy - self.SUN_VISUAL_RADIUS * 3))
                pygame.draw.circle(self.screen, color, (sx, sy),
                                   self.SUN_VISUAL_RADIUS)
            else:
                visual_r = max(self.MIN_BODY_RADIUS,
                               int(body["radius"] * self.scale * 500))
                if name in ("Europa", "Enceladus"):
                    visual_r = max(2, visual_r)
                pygame.draw.circle(self.screen, color, (sx, sy), visual_r)
                label = self._label_font.render(name, True, HUD_WHITE)
                self.screen.blit(label, (sx + visual_r + 3, sy - 6))

    def _draw_trajectory(self, state):
        trajectory = state.get("trajectory", [])
        if len(trajectory) < 2:
            return
        points = [self._world_to_screen(pos[0], pos[1]) for pos in trajectory]
        n = len(points)
        for i in range(1, n):
            alpha = int(255 * i / n)
            pygame.draw.line(self.screen, self._traj_colors[alpha],
                             points[i - 1], points[i], 1)

    def _draw_spacecraft(self, state):
        pos = state.get("position", np.zeros(3))
        orientation = state.get("orientation", np.zeros(3))
        sx, sy = self._world_to_screen(pos[0], pos[1])
        yaw = orientation[1]
        size = self.SPACECRAFT_SIZE
        forward = np.array([np.cos(yaw), -np.sin(yaw)])
        right = np.array([np.sin(yaw), np.cos(yaw)])
        tip = (sx + forward[0] * size, sy + forward[1] * size)
        left_wing = (sx - forward[0] * size * 0.5 - right[0] * size * 0.6,
                     sy - forward[1] * size * 0.5 - right[1] * size * 0.6)
        right_wing = (sx - forward[0] * size * 0.5 + right[0] * size * 0.6,
                      sy - forward[1] * size * 0.5 + right[1] * size * 0.6)
        pygame.draw.polygon(self.screen, (255, 255, 255),
                            [tip, left_wing, right_wing])
        thrust = state.get("thrust_level", 0.0)
        if thrust > 0:
            flame_len = size * thrust * 2
            flame_tip = (sx - forward[0] * (size * 0.5 + flame_len),
                         sy - forward[1] * (size * 0.5 + flame_len))
            pygame.draw.polygon(self.screen, THRUST_ORANGE,
                                [left_wing, right_wing, flame_tip])

    def close(self):
        pygame.quit()
