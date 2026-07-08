"""Shared constants for the booth demo."""

import os
from pathlib import Path

DEMO_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = DEMO_ROOT.parent
BUNDLED_DATA_DIR = DEMO_ROOT / "data"
# SQLite sessions/emails: use a mounted volume in production (Railway, Fly, etc.).
PERSISTENT_DIR = (
    Path(os.environ["DEMO_DATA_DIR"])
    if os.getenv("DEMO_DATA_DIR")
    else BUNDLED_DATA_DIR
)
DATA_DIR = BUNDLED_DATA_DIR
DB_DIR = PERSISTENT_DIR
SCENARIO_NAME = "booth_demo_6drug"
SIMULATION_WEEKS = 5
DRUG_PICKS = 2
FACTOR_PICKS = 3
# Visitor pick order → attention weight (β); unselected factors = 0.
VISITOR_PICK_WEIGHTS = (0.5, 0.35, 0.15)

# Poster-selectable factors (match briefing table rows).
# Runway (QOH ÷ UTZ) is always included in the weight vector.
# erd maps to runway urgency; erd_uncertainty maps to runway_uncertainty in the planner.
SELECTABLE_FACTORS = {
    "qoh": "Quantity on hand",
    "qoh_uncertainty": "QOH uncertainty",
    "usage": "Utilization rate",
    "usage_uncertainty": "Utilization uncertainty",
    "clinical": "Clinical impact",
    "erd": "ERD (weeks)",
    "erd_uncertainty": "ERD uncertainty",
}

FACTOR_WEIGHT_KEY = {
    "qoh": "qoh",
    "qoh_uncertainty": "qoh_uncertainty",
    "usage": "usage",
    "usage_uncertainty": "usage_uncertainty",
    "clinical": "clinical",
    "erd": "runway",
    "erd_uncertainty": "runway_uncertainty",
}

WEIGHT_DISPLAY_LABELS = {
    "runway": "Runway",
    "runway_uncertainty": "ERD uncertainty",
    "qoh": "Quantity on hand",
    "qoh_uncertainty": "QOH uncertainty",
    "usage": "Utilization rate",
    "usage_uncertainty": "Utilization uncertainty",
    "clinical": "Clinical impact",
    "reputation": "Shortage history",
}

ALL_WEIGHT_FEATURES = [
    "runway",
    "runway_uncertainty",
    "qoh",
    "qoh_uncertainty",
    "usage",
    "usage_uncertainty",
    "clinical",
    "reputation",
]

# CogSci poster: each drug column is name → qoh → utz → clinical → erd weeks.
# Format: value / uncertainty (slash = how reliable the intel is).
DRUG_DISPLAY = {
    "Drug A": {
        "qoh": "20ml",
        "qoh_uncertainty": 0.25,
        "utz": "15ml",
        "utz_uncertainty": 0.5,
        "clinical": 1,
        "resupply_weeks": 1,
        "resupply_uncertainty": 0.95,
    },
    "Drug B": {
        "qoh": "17 fl oz",
        "qoh_uncertainty": 0.9,
        "utz": "3 fl oz",
        "utz_uncertainty": 0.1,
        "clinical": 5,
        "resupply_weeks": 3,
        "resupply_uncertainty": 0.5,
    },
    "Drug C": {
        "qoh": "10g",
        "qoh_uncertainty": 0.3,
        "utz": "1g",
        "utz_uncertainty": 0.2,
        "clinical": 3,
        "resupply_weeks": 7,
        "resupply_uncertainty": 0.25,
    },
    "Drug D": {
        "qoh": "123 vials",
        "qoh_uncertainty": 0.6,
        "utz": "10 vials",
        "utz_uncertainty": 0.8,
        "clinical": 6,
        "resupply_weeks": 3,
        "resupply_uncertainty": 0.1,
    },
    "Drug E": {
        "qoh": "55 packs",
        "qoh_uncertainty": 0.625,
        "utz": "5 packs",
        "utz_uncertainty": 0.35,
        "clinical": 2,
        "resupply_weeks": 11,
        "resupply_uncertainty": 0.3,
    },
    "Drug F": {
        "qoh": "1000lbs",
        "qoh_uncertainty": 0.1,
        "utz": "200lbs",
        "utz_uncertainty": 0.95,
        "clinical": 4,
        "resupply_weeks": 6,
        "resupply_uncertainty": 0.8,
    },
}

# Numeric values for simulation (ratio of numeric parts ≈ runway weeks).
SIM_DRUG_NUMERIC = {
    "Drug A": {"qoh": 20, "utz": 15, "qoh_uncertainty": 0.25, "utz_uncertainty": 0.5},
    "Drug B": {"qoh": 17, "utz": 3, "qoh_uncertainty": 0.9, "utz_uncertainty": 0.1},
    "Drug C": {"qoh": 10, "utz": 1, "qoh_uncertainty": 0.3, "utz_uncertainty": 0.2},
    "Drug D": {"qoh": 123, "utz": 10, "qoh_uncertainty": 0.6, "utz_uncertainty": 0.8},
    "Drug E": {"qoh": 55, "utz": 5, "qoh_uncertainty": 0.625, "utz_uncertainty": 0.35},
    "Drug F": {"qoh": 1000, "utz": 200, "qoh_uncertainty": 0.1, "utz_uncertainty": 0.95},
}

BACKGROUND_OPTIONS = [
    "cognitive_science",
    "ai_ml",
    "healthcare",
    "operations",
    "other",
]
