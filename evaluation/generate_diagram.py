"""Generate a publication-quality RL agent-environment interaction diagram.

Shows the RL loop: Agent -> Action -> Environment -> Observation/Reward -> Agent
with detailed breakdowns of observation space, action space, and reward structure.

Usage:
    python evaluation/generate_diagram.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


def generate_diagram():
    """Generate the environment diagram as PNG and PDF."""
    os.makedirs("results/diagrams", exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(18, 13))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 13)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Title
    ax.text(9, 12.5, "AstroLogic: RL Agent-Environment Interaction Loop",
            ha="center", va="center", fontsize=18, fontweight="bold", color="#2c3e50")
    ax.text(9, 12.0, "Reinforcement Learning for Astrobiological Exploration",
            ha="center", va="center", fontsize=12, color="#7f8c8d", style="italic")

    # ============================================================
    # AGENT BOX (top center)
    # ============================================================
    agent_box = FancyBboxPatch((5.5, 9.5), 7, 2, boxstyle="round,pad=0.15",
                                facecolor="#ecf0f1", edgecolor="#2c3e50", linewidth=2)
    ax.add_patch(agent_box)
    ax.text(9, 11.1, "RL Agent", ha="center", va="center",
            fontsize=14, fontweight="bold", color="#2c3e50")

    # Algorithm labels
    algos = [
        ("DQN", "#e74c3c", "Value-Based"),
        ("REINFORCE", "#9b59b6", "Policy Gradient"),
        ("PPO", "#2ecc71", "Proximal Policy Opt."),
    ]
    for i, (name, color, desc) in enumerate(algos):
        x = 6.3 + i * 1.6
        ax.add_patch(FancyBboxPatch((x - 0.6, 9.7), 1.2, 0.9, boxstyle="round,pad=0.05",
                                     facecolor=color, edgecolor="none", alpha=0.2))
        ax.text(x, 10.3, name, ha="center", va="center",
                fontsize=10, fontweight="bold", color=color)
        ax.text(x, 9.95, desc, ha="center", va="center",
                fontsize=7, color="#555")

    # ============================================================
    # ENVIRONMENT BOX (center)
    # ============================================================
    env_box = FancyBboxPatch((3, 4), 12, 4.5, boxstyle="round,pad=0.2",
                              facecolor="#1a1a2e", edgecolor="#16213e", linewidth=2.5)
    ax.add_patch(env_box)
    ax.text(9, 8.2, "AstroExploration Environment", ha="center", va="center",
            fontsize=13, fontweight="bold", color="#f0f0f0")

    # Sun
    sun = plt.Circle((6, 6.2), 0.35, color="#FFD700", zorder=5)
    ax.add_patch(sun)
    ax.text(6, 5.6, "Sun", ha="center", fontsize=8, color="#ccc")

    # Orbit circles
    for r, body, color in [(1.2, "Earth", "#4682C8"), (1.8, "Mars", "#C14A0E"),
                           (3.0, "Jupiter", "#C9B083"), (4.2, "Saturn", "#D2B464")]:
        orbit = plt.Circle((6, 6.2), r, fill=False, color="#333", linewidth=0.5, linestyle="--")
        ax.add_patch(orbit)
        angle = np.random.uniform(0, 2 * np.pi)
        bx = 6 + r * np.cos(angle)
        by = 6.2 + r * np.sin(angle)
        planet = plt.Circle((bx, by), 0.12, color=color, zorder=5)
        ax.add_patch(planet)
        ax.text(bx + 0.2, by + 0.15, body, fontsize=6, color="#aaa")

    # Spacecraft
    spacecraft_x, spacecraft_y = 7.5, 7.0
    ax.plot(spacecraft_x, spacecraft_y, marker="^", color="white", markersize=12, zorder=6)
    ax.text(spacecraft_x + 0.3, spacecraft_y, "Spacecraft", fontsize=7, color="#0f0")

    # Detection zones
    for cx, cy, label in [(7.8, 6.2, "Bio\nZone"), (8.5, 5.2, "Bio\nZone")]:
        zone = plt.Circle((cx, cy), 0.4, color="#00FF00", alpha=0.1, zorder=3)
        ax.add_patch(zone)
        ax.text(cx, cy, label, ha="center", va="center", fontsize=5, color="#0a0")

    # Trajectory trail
    trail_x = [7.5, 7.3, 7.0, 6.8, 6.5]
    trail_y = [7.0, 6.8, 6.5, 6.3, 6.1]
    ax.plot(trail_x, trail_y, color="white", linewidth=0.8, alpha=0.5, linestyle=":")

    # ============================================================
    # OBSERVATION SPACE BOX (left)
    # ============================================================
    obs_box = FancyBboxPatch((0.2, 4), 2.5, 4.5, boxstyle="round,pad=0.1",
                              facecolor="#eaf2f8", edgecolor="#2980b9", linewidth=1.5)
    ax.add_patch(obs_box)
    ax.text(1.45, 8.2, "Observation Space", ha="center", va="center",
            fontsize=10, fontweight="bold", color="#2980b9")
    ax.text(1.45, 7.85, "Box(23,) float32", ha="center", fontsize=8, color="#555")

    obs_items = [
        "[0:3]  Position (x,y,z) AU",
        "[3:6]  Velocity (vx,vy,vz)",
        "[6]    Dist. to Mars",
        "[7:10] Heading to Mars",
        "[10]   Dist. to Europa",
        "[11:14] Heading Europa",
        "[14]   Dist. to Enceladus",
        "[15:18] Heading Enceladus",
        "[18]   Fuel [0,1]",
        "[19]   Battery [0,1]",
        "[20]   SNR Signal",
        "[21]   Biosigs Found/3",
        "[22]   Biosigs TX/3",
    ]
    for i, item in enumerate(obs_items):
        ax.text(0.4, 7.5 - i * 0.27, item, fontsize=6, color="#333",
                fontfamily="monospace")

    # ============================================================
    # ACTION SPACE BOX (right)
    # ============================================================
    act_box = FancyBboxPatch((15.3, 4), 2.5, 4.5, boxstyle="round,pad=0.1",
                              facecolor="#fdf2e9", edgecolor="#e67e22", linewidth=1.5)
    ax.add_patch(act_box)
    ax.text(16.55, 8.2, "Action Space", ha="center", va="center",
            fontsize=10, fontweight="bold", color="#e67e22")
    ax.text(16.55, 7.85, "MultiDiscrete [5,3,3,3,4,2]", ha="center", fontsize=8, color="#555")

    act_items = [
        "[0] Thrust (5 levels)",
        "    {0, 0.25, 0.5, 0.75, 1.0}",
        "[1] Pitch (3 levels)",
        "    {-5, 0, +5} degrees",
        "[2] Yaw (3 levels)",
        "    {-5, 0, +5} degrees",
        "[3] Roll (3 levels)",
        "    {-5, 0, +5} degrees",
        "[4] Instrument (4 options)",
        "    None/Spectro/Thermal/Drill",
        "[5] Communication (2)",
        "    {Off, Transmit}",
    ]
    for i, item in enumerate(act_items):
        ax.text(15.5, 7.5 - i * 0.28, item, fontsize=6, color="#333",
                fontfamily="monospace")

    # ============================================================
    # REWARD TABLE (bottom center)
    # ============================================================
    rew_box = FancyBboxPatch((3.5, 0.3), 11, 3.2, boxstyle="round,pad=0.1",
                              facecolor="#f5f5f5", edgecolor="#27ae60", linewidth=1.5)
    ax.add_patch(rew_box)
    ax.text(9, 3.2, "Reward Structure R(s, a, s')", ha="center", va="center",
            fontsize=11, fontweight="bold", color="#27ae60")

    # Reward table headers
    col_x = [4.5, 7.5, 9.5, 12]
    headers = ["Event", "Value", "Type", "Trigger"]
    for x, h in zip(col_x, headers):
        ax.text(x, 2.85, h, fontsize=8, fontweight="bold", color="#333")

    rewards_data = [
        ("Liquid Water", "+500", "Sparse", "Instrument detects in zone"),
        ("Ice Detection", "+300", "Sparse", "Instrument detects in zone"),
        ("Organic Compounds", "+750", "Sparse", "Instrument detects in zone"),
        ("Signs of Intelligence", "+5000", "Sparse", "Instrument detects in zone"),
        ("Orbital Insertion", "+100", "Sparse", "Close + slow near target"),
        ("Step Penalty", "-(a*fuel+b)", "Dense", "Every timestep"),
        ("Collision/OOB", "-1000", "Terminal", "Body collision or >50 AU"),
    ]
    for i, (event, value, rtype, trigger) in enumerate(rewards_data):
        y = 2.55 - i * 0.3
        color = "#27ae60" if "+" in value else "#e74c3c"
        ax.text(col_x[0], y, event, fontsize=7, color="#333")
        ax.text(col_x[1], y, value, fontsize=7, color=color, fontweight="bold")
        ax.text(col_x[2], y, rtype, fontsize=7, color="#555")
        ax.text(col_x[3], y, trigger, fontsize=6, color="#777")

    # ============================================================
    # ARROWS (RL Loop)
    # ============================================================
    # Agent -> Action -> Environment (right side)
    ax.annotate("", xy=(15.3, 6.5), xytext=(12.5, 10),
                arrowprops=dict(arrowstyle="-|>", color="#e67e22", lw=2.5))
    ax.text(14.5, 8.5, "Action\na(t)", ha="center", fontsize=10,
            fontweight="bold", color="#e67e22", rotation=-50)

    # Environment -> Observation -> Agent (left side)
    ax.annotate("", xy=(5.5, 10), xytext=(2.7, 6.5),
                arrowprops=dict(arrowstyle="-|>", color="#2980b9", lw=2.5))
    ax.text(3.5, 8.5, "Obs s(t)\nReward r(t)", ha="center", fontsize=10,
            fontweight="bold", color="#2980b9", rotation=50)

    # Terminal conditions annotation
    ax.text(9, 4.2, "Terminal: Success (3 biosigs TX) | Resources=0 | Collision | OOB | 100K steps",
            ha="center", fontsize=7, color="#999", style="italic")

    # Save
    plt.savefig("results/diagrams/environment_diagram.png", dpi=200,
                bbox_inches="tight", facecolor="white")
    plt.savefig("results/diagrams/environment_diagram.pdf",
                bbox_inches="tight", facecolor="white")
    plt.close()

    print("Environment diagram saved to:")
    print("  results/diagrams/environment_diagram.png")
    print("  results/diagrams/environment_diagram.pdf")


if __name__ == "__main__":
    generate_diagram()
