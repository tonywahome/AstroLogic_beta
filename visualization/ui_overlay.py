"""HUD overlay for the pygame renderer."""

import numpy as np
import pygame
from visualization.colors import (
    HUD_GREEN, HUD_RED, HUD_BLUE, HUD_WHITE, HUD_YELLOW, HUD_GRAY,
)


class UIOverlay:
    """Draws HUD elements (fuel, battery, biosig counters, etc.)."""

    def __init__(self, screen_width: int, screen_height: int):
        self.sw = screen_width
        self.sh = screen_height
        self.font_sm = pygame.font.SysFont("consolas", 14)
        self.font_md = pygame.font.SysFont("consolas", 16)
        self.font_lg = pygame.font.SysFont("consolas", 20, bold=True)

    def draw(self, surface: pygame.Surface, state: dict):
        """Draw all HUD elements onto the given surface."""
        self._draw_resource_bars(surface, state)
        self._draw_instrument_status(surface, state)
        self._draw_biosignatures(surface, state)
        self._draw_mission_info(surface, state)
        self._draw_position_info(surface, state)

    def _draw_resource_bars(self, surface, state):
        """Draw fuel and battery bars (top-left)."""
        x, y = 15, 15
        bar_w, bar_h = 180, 18

        # Fuel bar
        self._draw_label(surface, "FUEL", x, y, HUD_WHITE)
        fuel = state.get("fuel", 0.0)
        fuel_color = self._lerp_color(HUD_RED, HUD_GREEN, fuel)
        self._draw_bar(surface, x + 55, y, bar_w, bar_h, fuel, fuel_color)
        self._draw_label(surface, f"{fuel*100:.1f}%", x + bar_w + 60, y, HUD_WHITE)

        # Battery bar
        y += 28
        self._draw_label(surface, "BATT", x, y, HUD_WHITE)
        batt = state.get("battery", 0.0)
        self._draw_bar(surface, x + 55, y, bar_w, bar_h, batt, HUD_BLUE)
        self._draw_label(surface, f"{batt*100:.1f}%", x + bar_w + 60, y, HUD_WHITE)

    def _draw_instrument_status(self, surface, state):
        """Draw active instrument and comm status (top-right)."""
        x = self.sw - 230
        y = 15

        instrument = state.get("active_instrument", "None")
        color = HUD_YELLOW if instrument != "None" else HUD_GRAY
        self._draw_label(surface, f"INSTRUMENT: {instrument}", x, y, color)

        y += 22
        biosig_found = state.get("biosig_found", [])
        biosig_transmitted = state.get("biosig_transmitted", [])
        comm_text = f"FOUND: {len(biosig_found)}  TX: {len(biosig_transmitted)}"
        self._draw_label(surface, comm_text, x, y, HUD_GREEN)

    def _draw_biosignatures(self, surface, state):
        """Draw biosignature indicators (bottom-left)."""
        x, y = 15, self.sh - 100
        self._draw_label(surface, "BIOSIGNATURES", x, y, HUD_WHITE)

        biosig_found = set(state.get("biosig_found", []))
        biosig_transmitted = set(state.get("biosig_transmitted", []))

        biosig_names = ["liquid_water", "ice", "organic_compounds", "signs_of_intelligence"]
        short_names = ["Water", "Ice", "Organic", "Intelligence"]

        for i, (full, short) in enumerate(zip(biosig_names, short_names)):
            y_row = y + 20 + i * 18
            if full in biosig_transmitted:
                color = HUD_GREEN
                status = "[TX]"
            elif full in biosig_found:
                color = HUD_YELLOW
                status = "[FOUND]"
            else:
                color = HUD_GRAY
                status = "[ ]"
            self._draw_label(surface, f"{status} {short}", x + 10, y_row, color)

    def _draw_mission_info(self, surface, state):
        """Draw step counter and cumulative reward (bottom-center)."""
        step = state.get("current_step", 0)
        max_steps = state.get("max_steps", 100000)
        reward = state.get("cumulative_reward", 0.0)

        x = self.sw // 2 - 100
        y = self.sh - 50

        self._draw_label(surface, f"Step: {step:,} / {max_steps:,}", x, y, HUD_WHITE)
        self._draw_label(surface, f"Reward: {reward:.2f}", x, y + 20, HUD_YELLOW)

    def _draw_position_info(self, surface, state):
        """Draw position, velocity, SNR (bottom-right)."""
        pos = state.get("position", [0, 0, 0])
        vel = state.get("velocity", [0, 0, 0])
        snr = state.get("snr", 0.0)

        x = self.sw - 280
        y = self.sh - 70

        vel_mag = np.linalg.norm(vel)
        self._draw_label(surface, f"POS: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) AU", x, y, HUD_GRAY)
        self._draw_label(surface, f"VEL: {vel_mag:.4f} AU/day", x, y + 18, HUD_GRAY)

        snr_color = HUD_GREEN if snr > 0.5 else (HUD_YELLOW if snr > 0.1 else HUD_GRAY)
        self._draw_label(surface, f"SNR: {snr:.3f}", x, y + 36, snr_color)

    def _draw_bar(self, surface, x, y, w, h, fraction, color):
        """Draw a filled progress bar."""
        # Background
        pygame.draw.rect(surface, (30, 30, 40), (x, y, w, h))
        # Fill
        fill_w = max(0, int(w * min(1.0, fraction)))
        if fill_w > 0:
            pygame.draw.rect(surface, color, (x, y, fill_w, h))
        # Border
        pygame.draw.rect(surface, HUD_GRAY, (x, y, w, h), 1)

    def _draw_label(self, surface, text, x, y, color):
        """Render a text label."""
        rendered = self.font_sm.render(text, True, color)
        surface.blit(rendered, (x, y))

    @staticmethod
    def _lerp_color(color_a, color_b, t):
        """Linear interpolation between two colors."""
        t = max(0.0, min(1.0, t))
        return tuple(int(a + (b - a) * t) for a, b in zip(color_a, color_b))
