"""External Reward Computation"""

import json
import os
from typing import List, Dict, Any
from src.environments.actions import Action

_REWARD_CFG_PATH = os.environ.get(
    'REWARD_CFG_PATH',
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'configs', 'rewards.json')
)
with open(_REWARD_CFG_PATH, 'r') as _f:
    _REWARD_CFG = json.load(_f)

_COST_CFG_PATH = os.environ.get(
    'COST_CFG_PATH',
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'configs', 'costs.json')
)
with open(_COST_CFG_PATH, 'r') as _f:
    _COST_CFG = json.load(_f)

_RUNWAY = _REWARD_CFG['runway_rewards']
_ACTION_COST_RAW = _COST_CFG['action_costs']

ACTION_COST = {getattr(Action, k): v for k, v in _ACTION_COST_RAW.items() if hasattr(Action, k)}
RUNWAY_REWARDS = _RUNWAY
_DELTA_RUNWAY_W = float(_REWARD_CFG.get('delta_runway_weight', 0.0))
_RECOVERY_BONUS = float(_REWARD_CFG.get('recovery_bonus', 0.0))
_INTERVENTION_SUCCESS_BONUS = float(_REWARD_CFG.get('intervention_success_bonus', 0.0))
_STABILITY_BONUS = float(_REWARD_CFG.get('stability_bonus', 0.0))
_PREVENTION_BONUS = float(_REWARD_CFG.get('prevention_bonus', 0.0))
_SCENARIO_AWARE_REWARDS = _REWARD_CFG.get('scenario_aware_rewards', {})
_SUPPLIER_TRUST_REWARDS = _REWARD_CFG.get('supplier_trust_rewards', {})
_URGENCY_AWARE_REWARDS = _REWARD_CFG.get('urgency_aware_rewards', {})
_SCENARIO_AWARE_PENALTIES = _COST_CFG.get('scenario_aware_penalties', {})
_SUPPLIER_TRUST_PENALTIES = _COST_CFG.get('supplier_trust_penalties', {})
_URGENCY_AWARE_PENALTIES = _COST_CFG.get('urgency_aware_penalties', {})
_SHORTAGE_ANOMALY_PENALTY = float(_COST_CFG.get('shortage_anomaly_penalty', 0.0))
LAMBDA_ACTION_COUNT = float(_COST_CFG.get('lambda_action_count', 15.0))
ENABLE_MULTI = bool(_COST_CFG.get('enable_multi', False))

def _action_cost(actions: List[Action]) -> float:
    """Action Cost Getter"""
    base = 0.0
    for a in actions:
        if isinstance(a, list):
            base += sum(ACTION_COST.get(sub_a.kind if hasattr(sub_a, 'kind') else sub_a, 0.0) for sub_a in a)
        else:
            base += ACTION_COST.get(a.kind if hasattr(a, 'kind') else a, 0.0)
    return base

def compute_week_reward(prev_true_state,
                        actions: Dict[str, List[Action]],
                        next_true_state) -> float:
    """Week Reward Getter"""
    managed_drugs = set(actions.keys()) if actions else None
    state_term = _state_reward_from_true_state(next_true_state, managed_drugs)
    if _DELTA_RUNWAY_W != 0.0 or _RECOVERY_BONUS != 0.0:
        deltas = []
        recovered = 0
        for drug_name in prev_true_state.drugs.keys():
            if drug_name in next_true_state.drugs:
                d_prev = prev_true_state.drugs[drug_name]
                d_next = next_true_state.drugs[drug_name]
                deltas.append(d_next.runway_weeks - d_prev.runway_weeks)
                if d_prev.runway_weeks < 1.0 and d_next.runway_weeks >= 1.0:
                    recovered += 1
        delta_term = _DELTA_RUNWAY_W * (sum(deltas) / max(len(deltas), 1))
        recovery_term = _RECOVERY_BONUS * recovered
    else:
        delta_term = 0.0
        recovery_term = 0.0
    all_actions = []
    for drug_actions in actions.values():
        all_actions.extend(drug_actions)
    action_term = _action_cost(all_actions)
    intervention_success_term = _intervention_success_reward_from_true_state(prev_true_state, next_true_state)
    stability_term = _stability_reward_from_true_state(next_true_state)
    shortage_anomaly_term = _shortage_anomaly_penalty_from_true_state(next_true_state)
    prevention_term = _prevention_reward_from_true_state(prev_true_state, next_true_state)
    return (state_term + action_term + delta_term + recovery_term + 
            intervention_success_term + stability_term + shortage_anomaly_term + prevention_term)

def _compute_drug_reward(drug, drug_actions: List[Action]) -> float:
    """Drug Reward Getter"""
    rw = drug.runway_weeks
    state_reward = 0.0
    if rw < 1.0:
        base_reward = _RUNWAY['lt_1']
        if drug.clinical_impact >= 0.8:
            base_reward += _RUNWAY['high_clinical_extra_under_1']
        state_reward = base_reward
    elif 1.0 <= rw < 2.0:
        state_reward = _RUNWAY['between_1_2']
    elif 2.0 <= rw < 4.0:
        state_reward = _RUNWAY['between_2_4']
    elif 4.0 <= rw <= 8.0:
        state_reward = _RUNWAY['between_4_8']
    elif 8.0 < rw <= 16.0:
        state_reward = _RUNWAY['between_8_16']
    else: 
        state_reward = _RUNWAY['gt_16']
    if drug.quantity_on_hand <= 0.0 or drug.runway_weeks < 1.0:
        state_reward += _RUNWAY['stockout_penalty']
    action_cost = _action_cost(drug_actions)
    
    return state_reward + action_cost

def compute_week_metrics(prev_true_state,
                         actions: Dict[str, List[Action]],
                         next_true_state) -> Dict[str, Any]:
    """Week Metrics Getter"""
    total_reward = compute_week_reward(prev_true_state, actions, next_true_state)
    metrics = {}
    total_stockouts = 0
    total_action_changes = 0
    stockouts = {}
    for drug_name, next_drug in next_true_state.drugs.items():
        is_stockout = (next_drug.quantity_on_hand <= 0.0 or next_drug.runway_weeks < 1.0)
        stockouts[drug_name] = is_stockout
        if is_stockout:
            total_stockouts += 1
    for drug_name, drug_actions in actions.items():
        if drug_name in prev_true_state.drugs and drug_name in next_true_state.drugs:
            prev_drug = prev_true_state.drugs[drug_name]
            next_drug = next_true_state.drugs[drug_name]
            prev_actions = []
            if hasattr(prev_drug, 'action_history') and prev_drug.action_history:
                prev_actions = prev_drug.action_history[-1] if prev_drug.action_history else []
            action_changed = (prev_actions != drug_actions)
            if action_changed:
                total_action_changes += 1
            drug_reward = _compute_drug_reward(next_drug, drug_actions)
            metrics[drug_name] = {
                "reward": round(drug_reward, 2),
                "stockout_next_week": stockouts.get(drug_name, False),
                "actions": drug_actions,
                "action_changed": action_changed,
                "prev_actions": prev_actions,
                "runway_next_week": round(next_drug.runway_weeks, 2),
                "qoh_next_week": round(next_drug.quantity_on_hand, 2),
            }
    metrics["total_reward"] = round(total_reward, 2)
    metrics["stockouts"] = stockouts
    metrics["total_stockouts"] = total_stockouts
    metrics["total_action_changes"] = total_action_changes
    total_actions = 0
    for drug_actions in actions.values():
        for action in drug_actions:
            if action != Action.WAIT_MONITOR:
                total_actions += 1
    metrics["total_actions"] = total_actions
    return metrics

def _state_reward_from_true_state(true_state, managed_drugs=None) -> float:
    """State Reward From True State Getter"""
    total = 0.0
    for drug_name, drug in true_state.drugs.items():
        if managed_drugs is not None and drug_name not in managed_drugs:
            continue
        rw = drug.runway_weeks
        if rw < 1.0:
            base_reward = _RUNWAY['lt_1']
            if drug.clinical_impact >= 0.8:
                base_reward += _RUNWAY['high_clinical_extra_under_1']
            total += base_reward
        elif 1.0 <= rw < 2.0:
            total += _RUNWAY['between_1_2']
        elif 2.0 <= rw < 4.0:
            total += _RUNWAY['between_2_4']
        elif 4.0 <= rw <= 8.0:
            total += _RUNWAY['between_4_8']
        elif 8.0 < rw <= 16.0:
            total += _RUNWAY['between_8_16']
        else: 
            total += _RUNWAY['gt_16']
        if drug.quantity_on_hand <= 0.0 or drug.runway_weeks < 1.0:
            total += _RUNWAY['stockout_penalty']
    return total

def _intervention_success_reward_from_true_state(prev_true_state, next_true_state) -> float:
    """Intervention Success Reward From True State Getter"""
    if _INTERVENTION_SUCCESS_BONUS == 0.0:
        return 0.0
    success_bonus = 0.0
    for drug_name in prev_true_state.drugs.keys():
        if drug_name in next_true_state.drugs:
            d_prev = prev_true_state.drugs[drug_name]
            d_next = next_true_state.drugs[drug_name]
            if d_prev.runway_weeks < 2.0 and d_next.runway_weeks >= 2.0:
                success_bonus += _INTERVENTION_SUCCESS_BONUS
            if d_next.runway_weeks - d_prev.runway_weeks > 1.0:
                success_bonus += _INTERVENTION_SUCCESS_BONUS * 0.5
    return success_bonus

def _stability_reward_from_true_state(true_state) -> float:
    """Stability Reward From True State Getter"""
    if _STABILITY_BONUS == 0.0:
        return 0.0
    stable_drugs = 0
    total_drugs = len(true_state.drugs)
    for drug in true_state.drugs.values():
        if 4.0 <= drug.runway_weeks <= 8.0:
            stable_drugs += 1
    stability_fraction = stable_drugs / max(total_drugs, 1)
    return _STABILITY_BONUS * stability_fraction

def _shortage_anomaly_penalty_from_true_state(true_state) -> float:
    """Shortage Anomaly Penalty From True State Getter"""
    if _SHORTAGE_ANOMALY_PENALTY == 0.0:
        return 0.0
    shortage_count = sum(1 for drug in true_state.drugs.values() if drug.quantity_on_hand <= 0.0 or drug.runway_weeks <= 0.0)
    return _SHORTAGE_ANOMALY_PENALTY * shortage_count

def _prevention_reward_from_true_state(prev_true_state, next_true_state) -> float:
    """Prevention Reward From True State Getter"""
    if _PREVENTION_BONUS == 0.0:
        return 0.0
    prevention_bonus = 0.0
    for drug_name in prev_true_state.drugs.keys():
        if drug_name in next_true_state.drugs:
            d_prev = prev_true_state.drugs[drug_name]
            d_next = next_true_state.drugs[drug_name]
            if 4.0 <= d_prev.runway_weeks <= 8.0 and 4.0 <= d_next.runway_weeks <= 8.0:
                prevention_bonus += _PREVENTION_BONUS * 0.5
            if d_prev.runway_weeks >= 2.0 and d_next.runway_weeks >= 2.0:
                prevention_bonus += _PREVENTION_BONUS * 0.3
    return prevention_bonus

__all__ = [
    'RUNWAY_REWARDS', 'ACTION_COST', 'LAMBDA_ACTION_COUNT', 'ENABLE_MULTI',
    '_SCENARIO_AWARE_REWARDS', '_SCENARIO_AWARE_PENALTIES', 
    '_SUPPLIER_TRUST_REWARDS', '_SUPPLIER_TRUST_PENALTIES',
    '_URGENCY_AWARE_REWARDS', '_URGENCY_AWARE_PENALTIES'
]