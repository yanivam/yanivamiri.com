"""Build attention weight vectors from visitor factor selections."""

from demo.constants import (
    ALL_WEIGHT_FEATURES,
    FACTOR_PICKS,
    FACTOR_WEIGHT_KEY,
    SELECTABLE_FACTORS,
    VISITOR_PICK_WEIGHTS,
    WEIGHT_DISPLAY_LABELS,
)


def build_visitor_weights(selected_factors: list[str]) -> dict[str, float]:
    """Convert poster factor picks (in click order) into attention weights β."""
    invalid = [f for f in selected_factors if f not in SELECTABLE_FACTORS]
    if invalid:
        raise ValueError(f"Unknown factors: {invalid}")
    if len(selected_factors) != FACTOR_PICKS:
        raise ValueError(f"Exactly {FACTOR_PICKS} factors must be selected")

    weights = {feature: 0.0 for feature in ALL_WEIGHT_FEATURES}
    for factor, pick_weight in zip(selected_factors, VISITOR_PICK_WEIGHTS):
        weight_key = FACTOR_WEIGHT_KEY[factor]
        weights[weight_key] = pick_weight
    return weights


def top_factor_labels(weights: dict[str, float], limit: int = 3) -> list[str]:
    """Human-readable poster-aligned labels for the highest-weighted factors."""
    ranked = sorted(weights.items(), key=lambda item: item[1], reverse=True)
    labels = []
    for key, _ in ranked[:limit]:
        labels.append(WEIGHT_DISPLAY_LABELS.get(key, key))
    return labels
