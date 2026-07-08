"""POMDP Planning Module"""
import json
import os
import random
import numpy as np
from typing import List, Dict
from dataclasses import dataclass, replace

from src.environments.actions import Action
from src.environments.belief_state import BeliefEnvState, BeliefDrug
from src.agents.types import ParamAction

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
_REWARDS = json.load(open(os.path.join(ROOT, 'configs', 'rewards.json')))
_COSTS = json.load(open(os.path.join(ROOT, 'configs', 'costs.json')))

EMERGENCY = [None, Action.REQUEST_FROM_RESERVE_WAREHOUSE, Action.GREY_MARKET_PURCHASE]
DEMAND = [None, Action.IMPLEMENT_SOFT_LMA, Action.IMPLEMENT_HARD_LMA, Action.REMOVE_LMA]
FOCUS = [None, Action.AUDIT_QOH_AND_UTZ]
SUPPLY = [None, Action.QUERY_ALTERNATE_SUPPLIER, Action.SWITCH_SUPPLIER, Action.SWITCH_BACK_TO_PRIMARY]

WEEKLY_BUDGET = _COSTS['weekly_budget']

@dataclass
class PlanningConfig:
    horizon: int = 10
    gamma: float = 0.95
    beam_width: int = 20 
    obs_sigma: float = 0.1
    n_obs_samples: int = 16 
    lambda_action_count: float = 0.1
    random_seed: int = 42

def legal_bundle(e, d, i, s, drug: BeliefDrug) -> bool:
    """Legal Action Bundle Checker"""
    bundle = [x for x in [e, d, i, s] if x is not None]
    if Action.IMPLEMENT_SOFT_LMA in bundle and Action.IMPLEMENT_HARD_LMA in bundle:
        return False
    if Action.IMPLEMENT_SOFT_LMA in bundle and Action.REMOVE_LMA in bundle:
        return False
    if Action.IMPLEMENT_HARD_LMA in bundle and Action.REMOVE_LMA in bundle:
        return False
    if Action.SWITCH_SUPPLIER in bundle and Action.SWITCH_BACK_TO_PRIMARY in bundle:
        return False
    if drug.runway_weeks >= 12 and d in (Action.IMPLEMENT_SOFT_LMA, Action.IMPLEMENT_HARD_LMA):
        return False
    if drug.runway_weeks > 4 and e in (Action.REQUEST_FROM_RESERVE_WAREHOUSE, Action.GREY_MARKET_PURCHASE):
        return False
    if s == Action.SWITCH_SUPPLIER and drug.active_supplier != "primary":
        return False
    if s == Action.SWITCH_BACK_TO_PRIMARY and drug.active_supplier != "alternate":
        return False
    if d == Action.REMOVE_LMA and drug.current_lma_type == "none":
        return False
    if d == Action.IMPLEMENT_SOFT_LMA and drug.current_lma_type == "soft":
        return False
    if d == Action.IMPLEMENT_HARD_LMA and drug.current_lma_type == "hard":
        return False
    if d == Action.REMOVE_LMA and drug.current_lma_type == "none":
        return False
    return True

def heuristic_score(bundle: List[Action], drug: BeliefDrug) -> float:
    """Heuristic Scorer"""
    if not bundle:
        return 0.0
    score = 0.0
    runway = drug.runway_weeks
    scenario_rewards = _REWARDS['scenario_aware_rewards']
    supplier_rewards = _REWARDS['supplier_trust_rewards']
    urgency_rewards = _REWARDS['urgency_aware_rewards']
    has_evolution_pattern = hasattr(drug, 'base_utz_evolution_config') and drug.base_utz_evolution_config is not None
    high_utilization_risk = (has_evolution_pattern and runway <= 4.0 and runway > 1.0)
    if Action.REQUEST_FROM_RESERVE_WAREHOUSE in bundle:
        if runway <= 0.5:
            score += 10000
        elif runway <= 1.0:
            bonus = urgency_rewards['critical_runway_emergency_action_bonus'] * 100
            score += bonus
        elif runway <= 2.0:
            bonus = urgency_rewards['critical_runway_emergency_action_bonus'] * 20
            score += bonus
        elif high_utilization_risk:
            bonus = urgency_rewards['critical_runway_emergency_action_bonus'] * 10
            score += bonus
    if Action.GREY_MARKET_PURCHASE in bundle:
        if runway <= 0.5:
            score += 9000
        elif runway <= 1.0:
            bonus = urgency_rewards['critical_runway_emergency_action_bonus'] * 80
            score += bonus
        elif runway <= 2.0:
            bonus = urgency_rewards['critical_runway_emergency_action_bonus'] * 15
            score += bonus
        elif high_utilization_risk:
            bonus = urgency_rewards['critical_runway_emergency_action_bonus'] * 8
            score += bonus
    if Action.IMPLEMENT_HARD_LMA in bundle and runway < 2.0:
        bonus = urgency_rewards['low_runway_appropriate_action_bonus'] * 6
        score += bonus
    if Action.IMPLEMENT_SOFT_LMA in bundle and 2.0 <= runway < 5.0:
        bonus = urgency_rewards['low_runway_appropriate_action_bonus'] * 4
        score += bonus
    if Action.AUDIT_QOH_AND_UTZ in bundle:
        if drug.qoh_uncertainty > 0.3:
            score += scenario_rewards['shortage_scenario']['audit_encouragement_bonus'] * 10
        elif drug.qoh_uncertainty > 0.15:
            score += scenario_rewards['shortage_scenario']['audit_encouragement_bonus'] * 6
        elif drug.qoh_uncertainty > 0.05:
            score += scenario_rewards['shortage_scenario']['audit_encouragement_bonus'] * 3
        if hasattr(drug, 'utz_uncertainty') and drug.utz_uncertainty > 0.3:
            score += scenario_rewards['shortage_scenario']['audit_encouragement_bonus'] * 8
        elif hasattr(drug, 'utz_uncertainty') and drug.utz_uncertainty > 0.15:
            score += scenario_rewards['shortage_scenario']['audit_encouragement_bonus'] * 4
        if hasattr(drug, 'weeks_since_audit') and drug.weeks_since_audit > 4:
            score += scenario_rewards['shortage_scenario']['audit_encouragement_bonus'] * 2
        if runway > 8.0 and drug.qoh_uncertainty > 0.2:
            score += scenario_rewards['shortage_scenario']['audit_encouragement_bonus'] * 5
    if Action.SWITCH_SUPPLIER in bundle:
        reliability_diff = drug.believed_alternate_reliability - drug.believed_primary_reliability
        if reliability_diff > 0.05:
            score += supplier_rewards['switch_to_reliable_supplier_bonus']
        if runway <= 4.0:
            score += supplier_rewards['switch_to_reliable_supplier_bonus'] * 0.5
        if drug.believed_primary_reliability < 0.8:
            score += supplier_rewards['switch_from_unreliable_supplier_bonus'] * 0.5
        if drug.erd_weeks > 6.0:
            score += supplier_rewards['switch_to_reliable_supplier_bonus'] * 0.4
        if (drug.active_supplier == "primary" and 
            drug.believed_alternate_reliability > drug.believed_primary_reliability + 0.1):
            score += supplier_rewards['switch_to_reliable_supplier_bonus'] * 0.6
    if Action.QUERY_ALTERNATE_SUPPLIER in bundle:
        if drug.believed_alternate_reliability_uncertainty > 0.2:
            score += scenario_rewards['shortage_scenario']['supplier_query_encouragement_bonus'] * 4
        if runway <= 3.0:
            score += scenario_rewards['shortage_scenario']['supplier_query_encouragement_bonus'] * 2
    if runway >= 12.0 and any(action in bundle for action in [Action.IMPLEMENT_SOFT_LMA, Action.IMPLEMENT_HARD_LMA]):
        score += _COSTS['urgency_aware_penalties']['high_runway_unnecessary_action_penalty']
    return score

def bundles_for_drug(drug: BeliefDrug, config: PlanningConfig) -> List[List[Action]]:
    """Legal Action Bundle Generator"""
    em = EMERGENCY if drug.runway_weeks <= 4 else [None]
    dm = DEMAND if drug.runway_weeks < 8 else [None, Action.REMOVE_LMA]
    inf = FOCUS if (drug.runway_weeks <= 4 or 
                   drug.qoh_uncertainty > 0.1 or 
                   (hasattr(drug, 'utz_uncertainty') and drug.utz_uncertainty > 0.1) or 
                   (hasattr(drug, 'weeks_since_audit') and drug.weeks_since_audit > 2)) else [None]
    sup = SUPPLY
    candidates = []
    for e in em:
        for d in dm:
            for i in inf:
                for s in sup:
                    if legal_bundle(e, d, i, s, drug):
                        bundle = [x for x in [e, d, i, s] if x is not None]
                        if not bundle:
                            bundle = [Action.WAIT_MONITOR]
                        if not _guaranteed_stockout_bundle(bundle, drug):
                            candidates.append(bundle)
    candidates = sorted(candidates, key=lambda b: heuristic_score(b, drug), reverse=True)[:config.beam_width]
    return candidates

def rollout_action(drug: BeliefDrug) -> Action:
    """Rollout Action Selector"""
    if drug.runway_weeks <= 1.0:
        return Action.REQUEST_FROM_RESERVE_WAREHOUSE
    has_evolution_pattern = hasattr(drug, 'base_utz_evolution_config') and drug.base_utz_evolution_config is not None
    if has_evolution_pattern and drug.runway_weeks <= 4.0 and drug.runway_weeks > 1.0:
        if drug.runway_weeks <= 2.0:
            return Action.GREY_MARKET_PURCHASE
        elif drug.runway_weeks <= 3.0:
            return Action.IMPLEMENT_HARD_LMA
    supply_issues = (
        drug.erd_weeks > 8.0 or
        (drug.active_supplier == "primary" and drug.believed_primary_reliability < 0.6) or
        (drug.active_supplier == "alternate" and drug.believed_alternate_reliability < 0.6)
    )
    if supply_issues and drug.runway_weeks < 4.0:
        if (drug.active_supplier == "primary" and 
            drug.believed_alternate_reliability > drug.believed_primary_reliability + 0.1):
            return Action.SWITCH_SUPPLIER
        elif drug.runway_weeks < 2.0:
            return Action.GREY_MARKET_PURCHASE
    if drug.runway_weeks < 2.0:
        return Action.IMPLEMENT_HARD_LMA
    if drug.runway_weeks < 5.0:
        return Action.IMPLEMENT_SOFT_LMA
    if (drug.qoh_uncertainty > 0.15 or 
        (hasattr(drug, 'utz_uncertainty') and drug.utz_uncertainty > 0.15) or
        drug.believed_alternate_reliability_uncertainty > 0.3 or
        (hasattr(drug, 'weeks_since_audit') and drug.weeks_since_audit > 3)):
        return Action.AUDIT_QOH_AND_UTZ
    return Action.WAIT_MONITOR

def rollout_policy_for_all_drugs(state: BeliefEnvState) -> Dict[str, List[ParamAction]]:
    """All Drugs Rollout Action Selector"""
    return {name: [ParamAction(rollout_action(drug))] for name, drug in state.drugs.items()}

def _plan(state: BeliefEnvState, config: PlanningConfig) -> Dict[str, List[ParamAction]]:
    """POMDP Online Planner"""
    seed = state.rng_seed + state.week_index * 1000
    np.random.seed(seed)
    random.seed(seed)
    drug_names = list(state.drugs.keys())
    return _plan_independent_best(state, config, drug_names)

def _plan_independent_best(state: BeliefEnvState, config: PlanningConfig, drug_names: List[str]) -> Dict[str, List[ParamAction]]:
    """Per Drug Independent Planner"""
    best_actions = {}
    total_evaluated = 0
    for drug_name in drug_names:
        drug = state.drugs[drug_name]
        candidate_bundles = bundles_for_drug(drug, config)
        best_val = float('-inf')
        best_bundle = None
        for bundle in candidate_bundles:
            drug_actions = {drug_name: [ParamAction(action) for action in bundle]}
            for other_drug_name, other_drug in state.drugs.items():
                if other_drug_name not in drug_actions:
                    rollout_act = rollout_action(other_drug)
                    drug_actions[other_drug_name] = [ParamAction(rollout_act)]
            val = _simulate_stochastic_future(state, drug_actions, config, focus_drug=drug_name)
            total_evaluated += 1
            if val > best_val:
                best_val = val
                best_bundle = bundle
        if best_bundle is None:
            best_actions[drug_name] = [ParamAction(Action.WAIT_MONITOR)]
        else:
            best_actions[drug_name] = [ParamAction(action) for action in best_bundle]
    for drug_name in state.drugs.keys():
        if drug_name not in best_actions:
            best_actions[drug_name] = [ParamAction(Action.WAIT_MONITOR)]    
    return best_actions

def _simulate_stochastic_future(state: BeliefEnvState, actions: Dict[str, List[ParamAction]], config: PlanningConfig, focus_drug: str = None) -> float:
    """Future Simulation Helper"""
    total_value = 0.0
    for sample_idx in range(config.n_obs_samples):
        current_belief = BeliefEnvState(
            drugs={name: replace(drug) for name, drug in state.drugs.items()},
            week_index=state.week_index,
            rng_seed=config.random_seed + sample_idx
        )
        current_belief = _sample_stochastic_transition(current_belief, actions, config)
        current_belief = _sample_observations_and_update_belief(current_belief, actions, config)
        immediate_reward = _compute_immediate_reward(current_belief, actions, focus_drug=focus_drug)
        total_value += immediate_reward
        parsimony_pen = -config.lambda_action_count * sum(
            1 for acts in actions.values() 
            if any(a.kind != Action.WAIT_MONITOR for a in acts)
        )
        total_value += parsimony_pen
        if _has_stockouts(current_belief):
            stockout_penalty = _REWARDS['runway_rewards']['stockout_penalty']
            total_value += stockout_penalty
            continue
        for step in range(config.horizon):
            future_actions = rollout_policy_for_all_drugs(current_belief)
            current_belief = _sample_stochastic_transition(current_belief, future_actions, config)
            current_belief = _sample_observations_and_update_belief(current_belief, future_actions, config)
            step_reward = _compute_immediate_reward(current_belief, future_actions, focus_drug=focus_drug)
            total_value += step_reward * (config.gamma ** (step + 1))
            if _has_stockouts(current_belief):
                stockout_penalty = _REWARDS['runway_rewards']['stockout_penalty']
                total_value += stockout_penalty * (config.gamma ** (step + 2))
                break
    avg_value = total_value / config.n_obs_samples
    return avg_value

def _has_stockouts(state: BeliefEnvState) -> bool:
    """Stockout Checker"""
    for drug in state.drugs.values():
        if drug.quantity_on_hand <= 0.0:
            return True
    return False

def _guaranteed_stockout_bundle(bundle: List[Action], drug: BeliefDrug) -> bool:
    """Guaranteed Stockout Bundle Checker"""
    if drug.quantity_on_hand <= 0.0:
        has_emergency = any(action in [Action.REQUEST_FROM_RESERVE_WAREHOUSE, Action.GREY_MARKET_PURCHASE] 
                          for action in bundle)
        return not has_emergency
    if drug.runway_weeks <= 0.5:
        has_emergency = any(action in [Action.REQUEST_FROM_RESERVE_WAREHOUSE, Action.GREY_MARKET_PURCHASE] 
                          for action in bundle)
        has_demand_reduction = any(action in [Action.IMPLEMENT_HARD_LMA] for action in bundle)
        return not (has_emergency or has_demand_reduction)
    return False

def _sample_stochastic_transition(state: BeliefEnvState, actions: Dict[str, List[ParamAction]], config: PlanningConfig) -> BeliefEnvState:
    """Transition Sampler Helper"""
    next_state = BeliefEnvState(
        drugs={name: replace(drug) for name, drug in state.drugs.items()},
        week_index=state.week_index + 1,
        rng_seed=state.rng_seed
    )
    for drug_name, drug in next_state.drugs.items():
        act_list = actions.get(drug_name, [ParamAction(Action.WAIT_MONITOR)])
        drug = _apply_actions_by_phase(drug, act_list, config, state.week_index)
        drug = _model_stochastic_consumption(drug, config)
        drug = _model_stochastic_supply(drug, config)
        next_state.drugs[drug_name] = drug
    return next_state

def _apply_actions_by_phase(drug: BeliefDrug, actions: List[ParamAction], config: PlanningConfig, week_index: int) -> BeliefDrug:
    """Action Applicator"""
    emergency_actions = [a for a in actions if a.kind in [Action.REQUEST_FROM_RESERVE_WAREHOUSE, Action.GREY_MARKET_PURCHASE]]
    demand_actions = [a for a in actions if a.kind in [Action.IMPLEMENT_SOFT_LMA, Action.IMPLEMENT_HARD_LMA, Action.REMOVE_LMA]]
    info_actions = [a for a in actions if a.kind in [Action.AUDIT_QOH_AND_UTZ]]
    supply_actions = [a for a in actions if a.kind in [Action.QUERY_ALTERNATE_SUPPLIER, Action.SWITCH_SUPPLIER, Action.SWITCH_BACK_TO_PRIMARY]]
    for action in emergency_actions:
        drug = _apply_stochastic_action(drug, action, config, week_index)
    for action in demand_actions:
        drug = _apply_stochastic_action(drug, action, config, week_index)
    for action in info_actions:
        drug = _apply_stochastic_action(drug, action, config, week_index)
    for action in supply_actions:
        drug = _apply_stochastic_action(drug, action, config, week_index)
    return drug

def _apply_stochastic_action(drug: BeliefDrug, action: ParamAction, config: PlanningConfig, week_index: int) -> BeliefDrug:
    """Simulate Future Action Application Helper"""
    if action.kind == Action.WAIT_MONITOR:
        return drug
    elif action.kind == Action.AUDIT_QOH_AND_UTZ:
        uncertainty_reduction = 0.08 + np.random.normal(0, 0.02)
        uncertainty_reduction = max(0.03, min(0.12, uncertainty_reduction))
        new_qoh_uncertainty = max(0.01, drug.qoh_uncertainty - uncertainty_reduction)
        new_utz_uncertainty = max(0.01, drug.utz_uncertainty - uncertainty_reduction)
        return replace(drug, qoh_uncertainty=new_qoh_uncertainty, utz_uncertainty=new_utz_uncertainty)
    elif action.kind == Action.IMPLEMENT_SOFT_LMA:
        if drug.current_lma_type != "none":
            return drug
        reduction_factor = 0.75 + np.random.normal(0, 0.1)
        reduction_factor = max(0.65, min(0.85, reduction_factor))
        new_utilization = drug.utilization_per_week * reduction_factor
        new_runway = _compute_runway(drug.quantity_on_hand, new_utilization)
        return replace(drug, utilization_per_week=new_utilization, runway_weeks=new_runway, 
                      current_lma_type="soft", lma_implemented_week=week_index)
    elif action.kind == Action.IMPLEMENT_HARD_LMA:
        if drug.current_lma_type == "hard":
            return drug
        reduction_factor = 0.25 + np.random.normal(0, 0.05)
        reduction_factor = max(0.20, min(0.30, reduction_factor))
        new_utilization = drug.utilization_per_week * reduction_factor
        new_runway = _compute_runway(drug.quantity_on_hand, new_utilization)
        return replace(drug, utilization_per_week=new_utilization, runway_weeks=new_runway,
                      current_lma_type="hard", lma_implemented_week=week_index)
    elif action.kind == Action.REMOVE_LMA:
        if drug.current_lma_type == "none":
            return drug
        base_utilization = drug.utilization_per_week * 1.2
        new_runway = _compute_runway(drug.quantity_on_hand, base_utilization)
        return replace(drug, utilization_per_week=base_utilization, 
                      runway_weeks=new_runway, current_lma_type="none", lma_implemented_week=-1)
    elif action.kind == Action.REQUEST_FROM_RESERVE_WAREHOUSE:
        base_boost = 6.0 * drug.utilization_per_week
        boost_factor = 1.0 + np.random.normal(0, 0.1)
        boost_factor = max(0.8, min(1.2, boost_factor))
        boost = base_boost * boost_factor
        new_qoh = drug.quantity_on_hand + boost
        new_runway = _compute_runway(new_qoh, drug.utilization_per_week)
        return replace(drug, quantity_on_hand=new_qoh, runway_weeks=new_runway)
    elif action.kind == Action.GREY_MARKET_PURCHASE:
        base_boost = 14.0 * drug.utilization_per_week
        boost_factor = 1.0 + np.random.normal(0, 0.15)
        boost_factor = max(0.7, min(1.3, boost_factor))
        boost = base_boost * boost_factor
        new_qoh = drug.quantity_on_hand + boost
        new_runway = _compute_runway(new_qoh, drug.utilization_per_week)
        return replace(drug, quantity_on_hand=new_qoh, runway_weeks=new_runway)
    elif action.kind == Action.REQUEST_LOAN_FROM_OTHER_HOSPITALS:
        base_boost = 6.0 * drug.utilization_per_week
        boost_factor = 1.0 + np.random.normal(0, 0.12)
        boost_factor = max(0.7, min(1.3, boost_factor))
        boost = base_boost * boost_factor
        new_qoh = drug.quantity_on_hand + boost
        new_runway = _compute_runway(new_qoh, drug.utilization_per_week)
        return replace(drug, quantity_on_hand=new_qoh, runway_weeks=new_runway)
    elif action.kind == Action.CONTACT_MANUFACTURER_DIRECT:
        base_boost = 6.0 * drug.utilization_per_week
        boost_factor = 1.0 + np.random.normal(0, 0.12)
        boost_factor = max(0.7, min(1.3, boost_factor))
        boost = base_boost * boost_factor
        new_qoh = drug.quantity_on_hand + boost
        new_runway = _compute_runway(new_qoh, drug.utilization_per_week)
        return replace(drug, quantity_on_hand=new_qoh, runway_weeks=new_runway)
    elif action.kind == Action.SWITCH_SUPPLIER:
        supply_improvement = 4.0 + np.random.normal(0, 0.8)
        new_erd_weeks = max(1.0, drug.erd_weeks - supply_improvement)
        reliability_boost = 0.1 + np.random.normal(0, 0.05)
        new_alternate_reliability = min(0.95, drug.believed_alternate_reliability + reliability_boost)
        return replace(drug, 
                      active_supplier='alternate', 
                      use_alternate=True,
                      erd_weeks=new_erd_weeks,
                      believed_alternate_reliability=new_alternate_reliability)
    elif action.kind == Action.SWITCH_BACK_TO_PRIMARY:
        return replace(drug, active_supplier='primary', use_alternate=False)
    elif action.kind == Action.QUERY_ALTERNATE_SUPPLIER:
        uncertainty_reduction = 0.1 + np.random.normal(0, 0.02)
        uncertainty_reduction = max(0.05, min(0.15, uncertainty_reduction))
        new_alternate_uncertainty = max(0.01, drug.believed_alternate_reliability_uncertainty - uncertainty_reduction)
        if np.random.random() < 0.3:
            info_bias = np.random.normal(0, 0.1)
            new_alternate_reliability = max(0.1, min(0.9, drug.believed_alternate_reliability + info_bias))
            return replace(drug, 
                         believed_alternate_reliability=new_alternate_reliability,
                         believed_alternate_reliability_uncertainty=new_alternate_uncertainty)
        else:
            return replace(drug, believed_alternate_reliability_uncertainty=new_alternate_uncertainty)
    else:
        return drug

def _model_stochastic_consumption(drug: BeliefDrug, config: PlanningConfig) -> BeliefDrug:
    """Simulate Consumption Helper"""
    if hasattr(drug, 'base_utz_evolution_config') and drug.base_utz_evolution_config:
        evolution_pattern = drug.base_utz_evolution_config
        current_week = getattr(drug, 'utz_evolution_week', 0)
        if current_week < len(evolution_pattern):
            new_utz = evolution_pattern[current_week]
        else:
            new_utz = evolution_pattern[-1]
    else:
        new_utz = drug.utilization_per_week
    new_utz = drug.utilization_per_week
    new_qoh = max(0.0, drug.quantity_on_hand - new_utz)
    new_runway = _compute_runway(new_qoh, new_utz)
    utz_evolution_week = getattr(drug, 'utz_evolution_week', 0) + 1
    
    return replace(drug, 
                  quantity_on_hand=new_qoh, 
                  runway_weeks=new_runway, 
                  utilization_per_week=new_utz,
                  utz_evolution_week=utz_evolution_week)

def _model_stochastic_supply(drug: BeliefDrug, config: PlanningConfig) -> BeliefDrug:
    """Simulate Supply Events Helper"""
    if drug.active_supplier == "primary":
        supplier_reliability = drug.believed_primary_reliability
    else:
        supplier_reliability = drug.believed_alternate_reliability
    base_prob = 0.4
    supply_prob = base_prob * supplier_reliability
    if drug.runway_weeks <= 1.0:
        supply_prob = min(0.8, supply_prob * 2.0)
    if np.random.random() < supply_prob:
        expected_delivery = drug.expected_delivery_amount or (drug.utilization_per_week * 14)
        if drug.active_supplier == "primary":
            supplier_reliability = drug.believed_primary_reliability
        else:
            supplier_reliability = drug.believed_alternate_reliability
        base_variance = 0.1
        reliability_variance = (1.0 - supplier_reliability) * 0.3
        delivery_variance = base_variance + reliability_variance
        delivery_factor = 1.0 + np.random.normal(0, delivery_variance)
        delivery_amount = expected_delivery * delivery_factor
        delivery_amount = max(0.1 * expected_delivery, delivery_amount)
        new_qoh = drug.quantity_on_hand + delivery_amount
        new_runway = _compute_runway(new_qoh, drug.utilization_per_week)
        return replace(drug, quantity_on_hand=new_qoh, runway_weeks=new_runway)
    return drug

def _sample_observations_and_update_belief(state: BeliefEnvState, actions: Dict[str, List[ParamAction]], config: PlanningConfig) -> BeliefEnvState:
    """Update Belief Helper"""
    updated_state = BeliefEnvState(
        drugs={name: replace(drug) for name, drug in state.drugs.items()},
        week_index=state.week_index,
        rng_seed=state.rng_seed
    )
    for drug_name, action_list in actions.items():
        if drug_name not in updated_state.drugs:
            continue       
        drug = updated_state.drugs[drug_name]        
        if any(action.kind == Action.AUDIT_QOH_AND_UTZ for action in action_list):
            observed_qoh = drug.quantity_on_hand + np.random.normal(0, config.obs_sigma)
            observed_qoh = max(0.0, observed_qoh)
            prior_weight = 0.7 
            obs_weight = 0.3    
            updated_qoh = prior_weight * drug.quantity_on_hand + obs_weight * observed_qoh
            updated_uncertainty = drug.qoh_uncertainty * 0.8
            drug = replace(drug, 
                          quantity_on_hand=updated_qoh,
                          qoh_uncertainty=max(0.01, updated_uncertainty))
        if drug.utilization_per_week > 0:
            drug = replace(drug, runway_weeks=_compute_runway(drug.quantity_on_hand, drug.utilization_per_week))
        updated_state.drugs[drug_name] = drug
    return updated_state

def _compute_immediate_reward(state: BeliefEnvState, actions: Dict[str, List[ParamAction]], focus_drug: str = None) -> float:
    """Reward Helper"""
    total_reward = 0.0
    runway_rewards = _REWARDS['runway_rewards']
    scenario_rewards = _REWARDS['scenario_aware_rewards']
    supplier_trust_rewards = _REWARDS['supplier_trust_rewards']
    urgency_rewards = _REWARDS['urgency_aware_rewards']
    action_costs = _COSTS['action_costs']
    scenario_penalties = _COSTS['scenario_aware_penalties']
    supplier_trust_penalties = _COSTS['supplier_trust_penalties']
    urgency_penalties = _COSTS['urgency_aware_penalties']
    for drug_name, action_list in actions.items():
        if focus_drug and drug_name != focus_drug:
            continue        
        if drug_name not in state.drugs:
            continue
        drug = state.drugs[drug_name]
        if drug.quantity_on_hand <= 0.0:
            total_reward += runway_rewards['stockout_penalty']
        elif drug.runway_weeks < 1.0:
            base_penalty = runway_rewards['lt_1']
            if drug.clinical_impact >= 0.8:
                base_penalty += runway_rewards['high_clinical_extra_under_1']
            total_reward += base_penalty
        elif 1.0 <= drug.runway_weeks < 2.0:
            total_reward += runway_rewards['between_1_2']
        elif 2.0 <= drug.runway_weeks < 4.0:
            total_reward += runway_rewards['between_2_4']
        elif 4.0 <= drug.runway_weeks <= 8.0:
            total_reward += runway_rewards['between_4_8']
        elif 8.0 < drug.runway_weeks <= 16.0:
            total_reward += runway_rewards['between_8_16']
        else:
            total_reward += runway_rewards['gt_16']
        for action in action_list:
            if action.kind.name in action_costs:
                cost = action_costs[action.kind.name]
                total_reward += cost
        lma_config = _REWARDS['lma_effectiveness']
        target_runway = lma_config['target_runway_weeks']
        if drug.runway_weeks < target_runway:
            soft_lma_new_runway = drug.runway_weeks / (1 - lma_config['soft_lma_utilization_reduction'])
            hard_lma_new_runway = drug.runway_weeks / (1 - lma_config['hard_lma_utilization_reduction'])
            has_soft_lma = any(action.kind == Action.IMPLEMENT_SOFT_LMA for action in action_list)
            has_hard_lma = any(action.kind == Action.IMPLEMENT_HARD_LMA for action in action_list)
            if has_soft_lma:
                expected_runway = soft_lma_new_runway
                if expected_runway >= target_runway:
                    total_reward += lma_config['soft_lma_sufficient_bonus']
                else:
                    total_reward += lma_config['soft_lma_insufficient_penalty']
            if has_hard_lma:
                expected_runway = hard_lma_new_runway
                if expected_runway >= target_runway:
                    if soft_lma_new_runway >= target_runway:
                        total_reward += lma_config['hard_lma_overkill_penalty']
                    else:
                        total_reward += lma_config['hard_lma_appropriate_bonus']
                else:
                    total_reward += lma_config['hard_lma_insufficient_bonus']
        if drug.runway_weeks >= 8.0:
            if any(action.kind in [Action.IMPLEMENT_SOFT_LMA, Action.IMPLEMENT_HARD_LMA] for action in action_list):
                total_reward += scenario_penalties['stable_scenario']['unnecessary_lma_penalty']
            if any(action.kind == Action.QUERY_ALTERNATE_SUPPLIER for action in action_list):
                total_reward += scenario_penalties['stable_scenario']['unnecessary_supplier_query_penalty']
            if any(action.kind == Action.AUDIT_QOH_AND_UTZ for action in action_list):
                total_reward += scenario_penalties['stable_scenario']['unnecessary_audit_penalty']
        elif drug.runway_weeks <= 2.0:
            if any(action.kind == Action.AUDIT_QOH_AND_UTZ for action in action_list):
                total_reward += scenario_rewards['shortage_scenario']['audit_encouragement_bonus']
            if any(action.kind == Action.QUERY_ALTERNATE_SUPPLIER for action in action_list):
                total_reward += scenario_rewards['shortage_scenario']['supplier_query_encouragement_bonus']
        if drug.runway_weeks <= 1.0:
            if any(action.kind in [Action.REQUEST_FROM_RESERVE_WAREHOUSE, Action.GREY_MARKET_PURCHASE] for action in action_list):
                total_reward += scenario_rewards['emergency_scenario']['emergency_action_encouragement_bonus']
        if drug.runway_weeks <= 2.0:
            if any(action.kind in [Action.IMPLEMENT_SOFT_LMA, Action.IMPLEMENT_HARD_LMA, Action.QUERY_ALTERNATE_SUPPLIER] for action in action_list):
                total_reward += urgency_rewards['low_runway_appropriate_action_bonus']
        if drug.runway_weeks <= 1.0:
            if any(action.kind in [Action.REQUEST_FROM_RESERVE_WAREHOUSE, Action.GREY_MARKET_PURCHASE] for action in action_list):
                total_reward += urgency_rewards['critical_runway_emergency_action_bonus']
        if drug.runway_weeks >= 8.0:
            if any(action.kind in [Action.IMPLEMENT_SOFT_LMA, Action.IMPLEMENT_HARD_LMA, Action.QUERY_ALTERNATE_SUPPLIER] for action in action_list):
                total_reward += urgency_penalties['high_runway_unnecessary_action_penalty']
        reliability_diff = drug.believed_alternate_reliability - drug.believed_primary_reliability
        if any(action.kind == Action.SWITCH_SUPPLIER for action in action_list):
            if drug.active_supplier == "primary":
                if reliability_diff > 0.3:
                    reward_amount = supplier_trust_rewards['switch_from_unreliable_supplier_bonus']
                    total_reward += reward_amount
                elif reliability_diff > 0.1:
                    reward_amount = supplier_trust_rewards['switch_to_reliable_supplier_bonus']
                    total_reward += reward_amount
                elif reliability_diff < -0.3:
                    penalty_amount = supplier_trust_penalties['switch_from_reliable_supplier_penalty']
                    total_reward += penalty_amount
                elif reliability_diff < -0.1:
                    penalty_amount = supplier_trust_penalties['switch_to_unreliable_supplier_penalty']
                    total_reward += penalty_amount
            else:
                penalty_amount = _REWARDS['supplier_switching']['switch_back_to_primary_penalty']
                total_reward += penalty_amount
        if any(action.kind == Action.QUERY_ALTERNATE_SUPPLIER for action in action_list):
            if drug.believed_primary_reliability > drug.believed_alternate_reliability + 0.2:
                total_reward += supplier_trust_penalties['query_reliable_supplier_penalty']
    return total_reward

def _compute_runway(qoh: float, utilization_per_week: float) -> float:
    """Runway Helper"""
    if utilization_per_week <= 0:
        return float('inf')
    return qoh / utilization_per_week
