"""RL-Learned Attention Agent"""

from typing import List, Dict
import math
import json
import os

from src.agents.base import BaseAgent
from src.environments.belief_state import BeliefEnvState
from src.environments.rewards import  LAMBDA_ACTION_COUNT
from src.planner.pomdp import _plan, PlanningConfig

class LearnedAttentionAgent(BaseAgent):
    def __init__(self, horizon: int = 15, gamma: float = 0.95, 
                 beam_width: int = 15, obs_sigma: float = 0.1, 
                 n_obs_samples: int = 16, top_k: int = 5, attention_threshold: float = 0.65, 
                 learning_rate: float = 0.01, exploration_rate: float = 0.1, max_learning_weeks: int = None):
        super().__init__(name="learned")
        self.horizon = horizon
        self.gamma = gamma
        self.beam_width = beam_width
        self.obs_sigma = obs_sigma
        self.n_obs_samples = n_obs_samples
        self.top_k = top_k
        self.attention_threshold = attention_threshold
        self.learning_rate = learning_rate * 0.8
        self.initial_learning_rate = self.learning_rate
        self.exploration_rate = exploration_rate
        self.momentum = 0.95
        self.velocity = {}
        self.max_gradient_norm = 0.5
        self.weight_decay = 0.0001
        self.entropy_coeff = 0.01
        self.initial_entropy_coeff = self.entropy_coeff
        self.max_learning_weeks = max_learning_weeks
        self.attention_weights = self._load_learned_weights()
        self.attention_history = []
        self.reward_history = []
        self.convergence_history = []
        self.episode_count = 0
        self.last_feature_activations = {}
        self.last_focus_set = []
        self.lambda_action_count = LAMBDA_ACTION_COUNT
        self.config = PlanningConfig(
            horizon=self.horizon,
            gamma=self.gamma,
            beam_width=self.beam_width,
            obs_sigma=self.obs_sigma,
            n_obs_samples=self.n_obs_samples,
            lambda_action_count=self.lambda_action_count
        )

    def _compute_rl_urgency(self, drug, state: BeliefEnvState, avg_qoh: float = None, avg_usage: float = None) -> float:
        """RL Urgency Getter"""
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
        qoh_urgency = runway_urgency
        qoh_uncertainty_urgency = getattr(drug, 'qoh_uncertainty', 0.0)
        if avg_usage is None:
            avg_usage = sum(d.utilization_per_week for d in state.drugs.values()) / max(1, len(state.drugs))
        usage_urgency = min(1.0, abs(drug.utilization_per_week - avg_usage) / max(avg_usage, 1e-6))
        usage_uncertainty_urgency = getattr(drug, 'utz_uncertainty', 0.0)
        clinical_urgency = getattr(drug, 'clinical_impact_score', 0.5)
        short_cnt = getattr(drug, 'shortage_count', 0.0)
        weeks_trk = max(1.0, getattr(drug, 'weeks_tracked', 1.0))
        reputation_urgency = min(1.0, short_cnt / weeks_trk)
        feature_activations = {
            'runway': runway_urgency,
            'runway_uncertainty': runway_uncertainty_urgency,
            'qoh': qoh_urgency,
            'qoh_uncertainty': qoh_uncertainty_urgency,
            'usage': usage_urgency,
            'usage_uncertainty': usage_uncertainty_urgency,
            'clinical': clinical_urgency,
            'reputation': reputation_urgency
        }
        w = self.attention_weights
        total_urgency = runway_urgency
        total_urgency += runway_uncertainty_urgency * w['runway_uncertainty']
        total_urgency += qoh_uncertainty_urgency * w['qoh_uncertainty']
        total_urgency += usage_urgency * w['usage']
        total_urgency += clinical_urgency * w['clinical']
        total_urgency += reputation_urgency * w['reputation']
        total_urgency = min(1.0, total_urgency)
        drug_idx = self._get_drug_index(drug, state)
        if drug_idx is not None:
            self.last_feature_activations[drug_idx] = feature_activations
        return total_urgency
    
    def _get_drug_index(self, drug, state: BeliefEnvState) -> int:
        """Drug Index Getter"""
        drug_names = list(state.drugs.keys())
        for idx, name in enumerate(drug_names):
            if state.drugs[name] is drug:
                return idx
        return None

    def _select_focus_set(self, state: BeliefEnvState) -> List[int]:
        """Focus Set Selector"""
        drug_names = list(state.drugs.keys())
        name_to_idx = {name: idx for idx, name in enumerate(drug_names)}
        eligible_names = (list(state.drugs.keys()))
        eligible_drugs = [state.drugs[name] for name in eligible_names]
        avg_qoh = sum(d.quantity_on_hand for d in eligible_drugs) / max(1, len(eligible_drugs))
        avg_usage = sum(d.utilization_per_week for d in eligible_drugs) / max(1, len(eligible_drugs))
        self.last_feature_activations = {}
        scores = [(name_to_idx[name], self._compute_rl_urgency(state.drugs[name], state, avg_qoh, avg_usage)) 
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

    def update_attention_weights(self, reward: float, state: BeliefEnvState):
        """Attention Weights Updater"""
        self.reward_history.append(reward)
        self.attention_history.append(self.attention_weights.copy())
        if len(self.reward_history) == 1:
            baseline = self.reward_history[0]
        else:
            alpha = 0.1
            if not hasattr(self, 'baseline'):
                self.baseline = self.reward_history[0]
            self.baseline = alpha * reward + (1 - alpha) * self.baseline
            baseline = self.baseline
        advantage = reward - baseline
        if self.last_feature_activations and self.last_focus_set:
            self._reinforce_update(advantage, state)
        self.exploration_rate = max(0.01, self.exploration_rate * 0.999)
        decay_factor = 0.99999
        self.learning_rate = max(0.0001, self.learning_rate * decay_factor)
        current_week = state.week_index
        if current_week != getattr(self, '_last_saved_week', -1):
            self._save_attention_weights(current_week)
            self._last_saved_week = current_week
    
    def _reinforce_update(self, advantage: float, state: BeliefEnvState):
        """Pure REINFORCE Updater"""
        if self.max_learning_weeks is not None and state.week_index > self.max_learning_weeks:
            return
        gradients = {}
        for feature in self.attention_weights:
            gradients[feature] = self._compute_attention_gradient(feature, state)
        gradient_norm = math.sqrt(sum(g**2 for g in gradients.values()))
        if gradient_norm > self.max_gradient_norm:
            clip_factor = self.max_gradient_norm / gradient_norm
            for feature in gradients:
                gradients[feature] *= clip_factor
        for feature in self.attention_weights:
            if feature not in self.velocity:
                self.velocity[feature] = 0.0
            weight_update = self.learning_rate * advantage * gradients[feature]
            self.velocity[feature] = self.momentum * self.velocity[feature] + weight_update
            weight_decay_update = -self.weight_decay * self.attention_weights[feature]
            current_week = state.week_index
            decayed_entropy_coeff = self.initial_entropy_coeff * max(0.1, 1.0 - (current_week / 100.0))
            entropy_gradient = -(math.log(self.attention_weights[feature] + 1e-8) + 1.0)
            entropy_update = decayed_entropy_coeff * entropy_gradient
            total_update = self.velocity[feature] + weight_decay_update + entropy_update
            self.attention_weights[feature] += total_update * 0.3
            self.attention_weights[feature] = max(0.001, self.attention_weights[feature])
        total_weight = sum(self.attention_weights.values())
        for feature in self.attention_weights:
            self.attention_weights[feature] /= total_weight
        self._update_learning_metrics(state)
    
    def _compute_attention_gradient(self, feature: str, state: BeliefEnvState) -> float:
        """Gradient Getter"""
        if not self.last_feature_activations or not self.last_focus_set:
            return 0.0
        all_drug_indices = list(self.last_feature_activations.keys())
        if not all_drug_indices:
            return 0.0
        weighted_scores = {}
        for drug_idx in all_drug_indices:
            if drug_idx in self.last_feature_activations:
                total_score = 0.0
                for f, weight in self.attention_weights.items():
                    feature_value = self.last_feature_activations[drug_idx].get(f, 0.0)
                    total_score += weight * feature_value
                weighted_scores[drug_idx] = total_score
        if not weighted_scores:
            return 0.0
        max_score = max(weighted_scores.values()) if weighted_scores else 0.0
        exp_scores = {idx: math.exp(score - max_score) for idx, score in weighted_scores.items()}
        total_exp = sum(exp_scores.values())
        if total_exp == 0:
            return 0.0
        softmax_probs = {idx: exp_score / total_exp for idx, exp_score in exp_scores.items()}
        gradient = 0.0
        for drug_idx in self.last_focus_set:
            if drug_idx not in self.last_feature_activations:
                continue
            feature_value = self.last_feature_activations[drug_idx].get(feature, 0.0)
            drug_prob = softmax_probs.get(drug_idx, 0.0)
            gradient += feature_value * (1.0 - drug_prob)
        feature_scales = {
            'runway': 1.0,  
            'runway_uncertainty': 2.0,  
            'qoh': 1.0,
            'qoh_uncertainty': 2.0,
            'usage': 1.0,
            'usage_uncertainty': 2.0,  
            'clinical': 1.0,           
            'reputation': 1.0          
        }
        gradient *= feature_scales.get(feature, 1.0)
        exploration_noise = 0.001 * max(0.0, 1.0 - (state.week_index / 100.0))
        if exploration_noise > 0:
            import random
            gradient += random.gauss(0, exploration_noise)
        return gradient
    
    def _update_learning_metrics(self, state: BeliefEnvState):
        """Learning Metrics Updater"""
        learning_data = {
            'episode': self.episode_count,
            'week': state.week_index,
            'current_weights': self.attention_weights.copy(),
            'learning_rate': self.learning_rate,
            'exploration_rate': self.exploration_rate
        }
        self.convergence_history.append(learning_data)
    
    def start_new_episode(self):
        """New Episode Starter"""
        self.episode_count += 1 
        self.exploration_rate = max(0.01, self.exploration_rate * 0.95)
        self.velocity = {}
        self.learning_rate = self.initial_learning_rate * (0.98 ** self.episode_count)
        self.learning_rate = max(0.001, self.learning_rate)
        # Ensure the initial (week 0) weights are logged for plotting and reproducibility.
        # This is especially important when experiments are run in separate subprocesses.
        self._last_saved_week = -1
        try:
            self._save_attention_weights(0)
            self._last_saved_week = 0
        except Exception:
            # Logging should never break the agent.
            pass
    
    def get_learning_stats(self) -> Dict:
        """Learning Statistics Getter"""
        return {
            'episode_count': self.episode_count,
            'attention_weights': self.attention_weights.copy(),
            'learning_rate': self.learning_rate,
            'exploration_rate': self.exploration_rate,
            'recent_rewards': self.reward_history[-10:] if self.reward_history else [],
            'avg_recent_reward': sum(self.reward_history[-10:]) / min(10, len(self.reward_history)) if self.reward_history else 0.0,
            'learning_history': self.convergence_history[-10:] if self.convergence_history else []
        }

    def _focus_set_changed(self, old_focus: List[int], new_focus: List[int]) -> bool:
        """Focus Set Changed Significantly Getter"""
        if len(old_focus) != len(new_focus):
            return True
        old_set = set(old_focus)
        new_set = set(new_focus)
        changed_drugs = len(old_set.symmetric_difference(new_set))
        return changed_drugs > 0  

    def select_actions(self, state):
        """Actions Selector"""
        self._last_week_index = state.week_index
        focus_indices = self._select_focus_set(state)
        focused_state = self._create_focused_state(state, focus_indices)
        actions = _plan(focused_state, self.config)
        return actions
    
    def _save_attention_weights(self, week: int):
        """Attention Weights Saver"""
        if self.max_learning_weeks is not None and week > self.max_learning_weeks:
            return
        attention_dir = os.environ.get("ATTENTION_WEIGHTS_DIR", "attention_weights")
        os.makedirs(attention_dir, exist_ok=True)
        filename = os.path.join(attention_dir, f"attention_weights_week_{week}.json")
        with open(filename, 'w') as f:
            json.dump(self.attention_weights, f, indent=2)
        history_filename = os.path.join(attention_dir, "attention_history.json")
        with open(history_filename, 'w') as f:
            json.dump(self.attention_history, f, indent=2)
        final_weights_filename = os.path.join(attention_dir, "learned_attention_weights.json")
        with open(final_weights_filename, 'w') as f:
            json.dump(self.attention_weights, f, indent=2)
        convergence_filename = os.path.join(attention_dir, "convergence_history.json")
        with open(convergence_filename, 'w') as f:
            json.dump(self.convergence_history, f, indent=2)
    
    def _load_learned_weights(self):
        """Learned Weights Loader"""
        default_weights = {
            'runway': float(1.0 / 8.0),
            'runway_uncertainty': float(1.0 / 8.0),
            'qoh': float(1.0 / 8.0),
            'qoh_uncertainty': float(1.0 / 8.0),
            'usage': float(1.0 / 8.0),
            'usage_uncertainty': float(1.0 / 8.0),
            'clinical': float(1.0 / 8.0),
            'reputation': float(1.0 / 8.0)
        }
        if os.environ.get("IGNORE_SAVED_ATTENTION_WEIGHTS", "0") == "1":
            return default_weights
        attention_dir = os.environ.get("ATTENTION_WEIGHTS_DIR", "attention_weights")
        learned_weights_file = os.path.join(attention_dir, "learned_attention_weights.json")
        if os.path.exists(learned_weights_file):
            try:
                with open(learned_weights_file, 'r') as f:
                    learned_weights = json.load(f)
                    return learned_weights
            except Exception as e:
                return default_weights
        else:
            return default_weights
