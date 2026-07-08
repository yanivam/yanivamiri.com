"""Booth-configured LearnedAttentionAgent from the research codebase."""

from __future__ import annotations

import os
from pathlib import Path

from demo.constants import DATA_DIR, SCENARIO_NAME, SIMULATION_WEEKS
from src.agents.rl import LearnedAttentionAgent
from src.environments.belief_state import BeliefEnvState

ATTENTION_WEIGHTS_DIR = DATA_DIR / "attention_weights"


class BoothLearnedAttentionAgent(LearnedAttentionAgent):
    """LearnedAttentionAgent with strict top-k focus for the 6-drug poster demo."""

    def _select_focus_set(self, state: BeliefEnvState) -> list[int]:
        drug_names = list(state.drugs.keys())
        name_to_idx = {name: idx for idx, name in enumerate(drug_names)}
        eligible_names = list(state.drugs.keys())
        eligible_drugs = [state.drugs[name] for name in eligible_names]
        avg_qoh = sum(d.quantity_on_hand for d in eligible_drugs) / max(1, len(eligible_drugs))
        avg_usage = sum(d.utilization_per_week for d in eligible_drugs) / max(1, len(eligible_drugs))
        self.last_feature_activations = {}
        scores = [
            (name_to_idx[name], self._compute_rl_urgency(state.drugs[name], state, avg_qoh, avg_usage))
            for name in eligible_names
        ]
        scores.sort(key=lambda item: item[1], reverse=True)
        above = [idx for idx, score in scores if score >= self.attention_threshold]
        if len(above) >= self.top_k:
            focus_set = above[: self.top_k]
        else:
            need = self.top_k - len(above)
            fallback = [idx for idx, _ in scores if idx not in above][:need]
            focus_set = above + fallback
        self.last_focus_set = focus_set
        return focus_set


def configure_learner_env() -> None:
    """Point the research agent at booth-local weight files."""
    ATTENTION_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["ATTENTION_WEIGHTS_DIR"] = str(ATTENTION_WEIGHTS_DIR)


def create_booth_learner(*, train_during_run: bool = True) -> BoothLearnedAttentionAgent:
    """Instantiate LearnedAttentionAgent used in paper experiments (booth settings)."""
    configure_learner_env()
    os.environ.pop("IGNORE_SAVED_ATTENTION_WEIGHTS", None)

    learner = BoothLearnedAttentionAgent(
        top_k=2,
        attention_threshold=0.45,
        beam_width=6,
        horizon=8,
        n_obs_samples=8,
        max_learning_weeks=SIMULATION_WEEKS if train_during_run else 0,
    )
    learner.start_new_episode()
    return learner


def train_booth_learner(training_weeks: int = 25) -> BoothLearnedAttentionAgent:
    """Train learner on booth scenario via REINFORCE; saves weights to attention_weights/."""
    from src.core.weekly_decisions import make_weekly_decisions

    configure_learner_env()
    os.environ["OVERRIDE_RNG_SEED"] = "42"
    os.environ["IGNORE_SAVED_ATTENTION_WEIGHTS"] = "1"

    learner = BoothLearnedAttentionAgent(
        top_k=2,
        attention_threshold=0.45,
        beam_width=6,
        horizon=8,
        n_obs_samples=8,
        max_learning_weeks=training_weeks,
    )
    learner.start_new_episode()

    prev_true_state = None
    prev_belief_state = None
    for week in range(training_weeks):
        prev_true_state, prev_belief_state, _ = make_weekly_decisions(
            learner,
            SCENARIO_NAME,
            week,
            prev_true_state,
            prev_belief_state,
        )

    learner._save_attention_weights(training_weeks)
    return learner
