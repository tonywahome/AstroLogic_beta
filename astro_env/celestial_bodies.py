"""Celestial body data for the AstroExploration environment.

All distances in AU, masses in solar masses, periods in simulation time units.
The solar system is simplified to circular orbits in the ecliptic plane.
"""

import numpy as np

CELESTIAL_BODIES = {
    "Sun": {
        "mass": 1.0,
        "radius": 0.00465,       # ~1 solar radius in AU
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
        "radius": 0.0000426,     # ~Earth radius in AU
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
        "orbit_radius": 0.0045,   # Europa orbits Jupiter at ~671,000 km ≈ 0.0045 AU
        "orbit_period": 3.55,     # 3.55 days orbital period
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
        "orbit_radius": 0.00159,  # Enceladus orbits Saturn at ~238,000 km ≈ 0.00159 AU
        "orbit_period": 1.37,     # 1.37 days orbital period
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

# Target bodies that the agent needs to explore (those with biosignatures)
TARGET_BODIES = ["Mars", "Europa", "Enceladus"]

# Instruments and which biosignatures they can detect
INSTRUMENTS = {
    0: {"name": "None", "detects": []},
    1: {"name": "Spectrometer", "detects": ["liquid_water", "organic_compounds"]},
    2: {"name": "ThermalImager", "detects": ["ice", "liquid_water"]},
    3: {"name": "Drill", "detects": ["organic_compounds", "ice", "signs_of_intelligence"]},
}
