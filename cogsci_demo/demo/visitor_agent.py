"""Visitor attention policy agent for booth demo."""

from typing import Dict, List
import math

from src.agents.base import BaseAgent
from src.environments.belief_state import BeliefEnvState
from src.environments.rewards import LAMBDA_ACTION_COUNT
from src.planner.pomdp import PlanningConfig, _plan


class VisitorAttentionAgent(BaseAgent):
    """Attention agent with visitor-chosen weights and week-1 focus set."""

    def __init__(
        self,
        attention_weights: Dict[str, float],
        initial_focus_drugs: List[str] | None = None,
        top_k: int = 2,
        attention_threshold: float = 0.35,
        horizon: int = 8,
        beam_width: int = 6,
        obs_sigma: float = 0.1,
        n_obs_samples: int = 8,
    ):
        super().__init__(name="visitor")
        self.attention_weights = attention_weights.copy()
        self.initial_focus_drugs = initial_focus_drugs or []
        self.top_k = top_k
        self.attention_threshold = attention_threshold
        self.last_focus_set: List[int] = []
        self.last_feature_activations: Dict[int, Dict[str, float]] = {}
        self.lambda_action_count = LAMBDA_ACTION_COUNT
        self.config = PlanningConfig(
            horizon=horizon,
            gamma=0.95,
            beam_width=beam_width,
            obs_sigma=obs_sigma,
            n_obs_samples=n_obs_samples,
            lambda_action_count=self.lambda_action_count,
        )

    def _clinical_urgency(self, drug) -> float:
        return getattr(drug, "clinical_impact", getattr(drug, "clinical_impact_score", 0.5))

    def _compute_urgency(self, drug, state: BeliefEnvState, avg_usage: float) -> float:
        target_runway = 5.0
        erd_weeks = drug.erd_weeks
        critical_threshold = min(erd_weeks * 0.75, target_runway)
        k = 1.5
        runway_urgency = 1.0 / (1.0 + math.exp(k * (drug.runway_weeks - critical_threshold)))
        active_reliability = (
            drug.believed_primary_reliability
            if drug.active_supplier == "primary"
            else drug.believed_alternate_reliability
        )
        supplier_risk_factor = 2.0 - active_reliability
        erd_risk_factor = 1.0 + min(0.5, (erd_weeks - target_runway) / 10.0)
        runway_urgency = min(1.0, runway_urgency * supplier_risk_factor * erd_risk_factor)

        runway_uncertainty_urgency = getattr(drug, "runway_uncertainty", 0.0)
        qoh_urgency = runway_urgency
        qoh_uncertainty_urgency = getattr(drug, "qoh_uncertainty", 0.0)
        usage_urgency = min(1.0, abs(drug.utilization_per_week - avg_usage) / max(avg_usage, 1e-6))
        usage_uncertainty_urgency = getattr(drug, "utz_uncertainty", 0.0)
        clinical_urgency = self._clinical_urgency(drug)
        short_cnt = getattr(drug, "shortage_count", 0.0)
        weeks_trk = max(1.0, getattr(drug, "weeks_tracked", 1.0))
        reputation_urgency = min(1.0, short_cnt / weeks_trk)

        w = self.attention_weights
        features = {
            "runway": runway_urgency,
            "runway_uncertainty": runway_uncertainty_urgency,
            "qoh": qoh_urgency,
            "qoh_uncertainty": qoh_uncertainty_urgency,
            "usage": usage_urgency,
            "usage_uncertainty": usage_uncertainty_urgency,
            "clinical": clinical_urgency,
            "reputation": reputation_urgency,
        }
        total_urgency = sum(w[key] * features[key] for key in w)
        return min(1.0, total_urgency)

    def _select_focus_set(self, state: BeliefEnvState) -> List[int]:
        drug_names = list(state.drugs.keys())
        name_to_idx = {name: idx for idx, name in enumerate(drug_names)}

        if state.week_index == 0 and self.initial_focus_drugs:
            focus_set = [
                name_to_idx[name]
                for name in self.initial_focus_drugs
                if name in name_to_idx
            ]
            self.last_focus_set = focus_set[: self.top_k]
            return self.last_focus_set

        eligible_names = list(state.drugs.keys())
        eligible_drugs = [state.drugs[name] for name in eligible_names]
        avg_usage = sum(d.utilization_per_week for d in eligible_drugs) / max(1, len(eligible_drugs))

        self.last_feature_activations = {}
        scores = [
            (name_to_idx[name], self._compute_urgency(state.drugs[name], state, avg_usage))
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

    def _create_focused_state(self, full_state: BeliefEnvState, focus_indices: List[int]) -> BeliefEnvState:
        drug_names = list(full_state.drugs.keys())
        focused_drugs = {
            drug_names[i]: full_state.drugs[drug_names[i]]
            for i in focus_indices
            if i < len(drug_names)
        }
        return BeliefEnvState(
            drugs=focused_drugs,
            week_index=full_state.week_index,
            rng_seed=full_state.rng_seed,
        )

    def select_actions(self, state):
        focus_indices = self._select_focus_set(state)
        focused_state = self._create_focused_state(state, focus_indices)
        return _plan(focused_state, self.config)
