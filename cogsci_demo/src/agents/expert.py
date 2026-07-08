"""Expert Derived Attention Agent"""

from typing import List
import math
from src.agents.base import BaseAgent
from src.environments.belief_state import BeliefEnvState
from src.environments.rewards import LAMBDA_ACTION_COUNT
from src.planner.pomdp import _plan, PlanningConfig

class ExpertDerivedAttentionAgent(BaseAgent):
    def __init__(self, horizon: int = 15, gamma: float = 0.95, 
                 beam_width: int = 15, obs_sigma: float = 0.1, 
                 n_obs_samples: int = 16, top_k: int = 5, attention_threshold: float = 0.65):
        super().__init__(name="expert_derived")
        self.horizon = horizon
        self.gamma = gamma
        self.beam_width = beam_width
        self.obs_sigma = obs_sigma
        self.n_obs_samples = n_obs_samples
        self.top_k = top_k
        self.attention_threshold = attention_threshold
        self.last_focus_set: List[int] = []
        self.attention_weights = {
            'runway': 0.20,
            'runway_uncertainty': 0.20,
            'qoh': 0.15,
            'qoh_uncertainty': 0.15,
            'usage': 0.10,
            'usage_uncertainty': 0.10,
            'clinical': 0.05,
            'reputation': 0.05
        }
        self.lambda_action_count = LAMBDA_ACTION_COUNT
        self.config = PlanningConfig(
            horizon=self.horizon,
            gamma=self.gamma,
            beam_width=self.beam_width,
            obs_sigma=self.obs_sigma,
            n_obs_samples=self.n_obs_samples,
            lambda_action_count=self.lambda_action_count
        )


    def _select_focus_set(self, state: BeliefEnvState) -> List[int]:
        """Focus Set Selector"""
        drug_names = list(state.drugs.keys())
        name_to_idx = {name: idx for idx, name in enumerate(drug_names)}
        eligible_names = (list(state.drugs.keys()))
        eligible_drugs = [state.drugs[name] for name in eligible_names]
        avg_usage = sum(d.utilization_per_week for d in eligible_drugs) / max(1, len(eligible_drugs))
        scores = [(name_to_idx[name], self._compute_expert_urgency(state.drugs[name], state, None, avg_usage)) 
                 for name in eligible_names]
        scores.sort(key=lambda x: x[1], reverse=True)
        above = [i for i, sc in scores if sc >= self.attention_threshold]
        if len(above) >= self.top_k:
            focus_set = above
        else:
            need = self.top_k - len(above)
            fallback = [i for i, _ in scores if i not in above][:need]
            focus_set = above + fallback
        self.last_focus_set = focus_set
        return focus_set

    def _create_focused_state(self, full_state: BeliefEnvState, focus_indices: List[int]) -> BeliefEnvState:
        """Focused State Getter"""
        drug_names = list(full_state.drugs.keys())
        focused_drugs = {}
        for i in focus_indices:
            if i < len(drug_names):
                focused_drugs[drug_names[i]] = full_state.drugs[drug_names[i]]
        return BeliefEnvState(drugs=focused_drugs, week_index=full_state.week_index, rng_seed=full_state.rng_seed)

    def _compute_expert_urgency(self, drug, state: BeliefEnvState, avg_qoh: float = None, avg_usage: float = None) -> float:
        """Expert Urgency Getter"""
        target_runway = 5.0 
        erd_weeks = drug.erd_weeks
        critical_threshold = min(erd_weeks * 0.75, target_runway)
        k = 1.5
        runway_urgency = 1.0 / (1.0 + math.exp(k * (drug.runway_weeks - critical_threshold)))
        active_reliability = (drug.believed_primary_reliability if drug.active_supplier == "primary" 
                            else drug.believed_alternate_reliability)
        supplier_risk_factor = 2.0 - active_reliability
        erd_risk_factor = 1.0 + min(0.5, (erd_weeks - target_runway) / 10.0)
        runway_urgency = min(1.0, runway_urgency * supplier_risk_factor * erd_risk_factor)
        runway_uncertainty_urgency = getattr(drug, 'runway_uncertainty', 0.0)
        qoh_uncertainty_urgency = getattr(drug, 'qoh_uncertainty', 0.0)
        if avg_usage is None:
            avg_usage = sum(d.utilization_per_week for d in state.drugs.values()) / max(1, len(state.drugs))
        usage_urgency = min(1.0, abs(drug.utilization_per_week - avg_usage) / max(avg_usage, 1e-6))
        usage_uncertainty_urgency = getattr(drug, 'utz_uncertainty', 0.0)
        clinical_urgency = getattr(drug, 'clinical_impact_score', 0.5)
        short_cnt = getattr(drug, 'shortage_count', 0.0)
        weeks_trk = max(1.0, getattr(drug, 'weeks_tracked', 1.0))
        reputation_urgency = min(1.0, short_cnt / weeks_trk)
        total_urgency = runway_urgency
        total_urgency += runway_uncertainty_urgency * 0.20
        total_urgency += qoh_uncertainty_urgency * 0.15
        total_urgency += usage_urgency * 0.10
        total_urgency += usage_uncertainty_urgency * 0.10
        total_urgency += clinical_urgency * 0.05
        total_urgency += reputation_urgency * 0.05
        total_urgency = min(1.0, total_urgency)
        return total_urgency



    def select_actions(self, state):
        """Actions Selector"""
        focus_indices = self._select_focus_set(state)
        focused_state = self._create_focused_state(state, focus_indices)
        actions = _plan(focused_state, self.config)
        
        return actions
