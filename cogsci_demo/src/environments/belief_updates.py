"""Belief Update Functions"""

from dataclasses import replace
from typing import Dict, List, Optional
import random

from src.environments.actions import Action
from src.environments.belief_state import BeliefDrug, BeliefEnvState, _belief_decay, _audit_belief, _compute_belief_runway
from src.environments.transition_functions import _get_current_erd
from src.environments.state import SUPPLIER_ERD_PATTERNS
                
def update_belief_from_observation(belief_drug: BeliefDrug, observation_type: str, 
                                 observed_value: float, observation_uncertainty: float) -> BeliefDrug:
    """Belief Updater"""
    if observation_type == "qoh":
        if belief_drug.qoh_uncertainty > 0 and observation_uncertainty > 0:
            new_mean = (belief_drug.quantity_on_hand / belief_drug.qoh_uncertainty + 
                       observed_value / observation_uncertainty) / (1.0 / belief_drug.qoh_uncertainty + 1.0 / observation_uncertainty)
            new_uncertainty = 1.0 / (1.0 / belief_drug.qoh_uncertainty + 1.0 / observation_uncertainty)
        else:
            new_mean = observed_value
            new_uncertainty = observation_uncertainty
        return replace(
            belief_drug,
            quantity_on_hand=new_mean,
            qoh_uncertainty=new_uncertainty,
            runway_weeks=_compute_belief_runway(new_mean, belief_drug.utilization_per_week)
        )
    elif observation_type == "utz":
        if belief_drug.utz_uncertainty > 0 and observation_uncertainty > 0:
            new_mean = (belief_drug.utilization_per_week / belief_drug.utz_uncertainty + 
                       observed_value / observation_uncertainty) / (1.0 / belief_drug.utz_uncertainty + 1.0 / observation_uncertainty)
            new_uncertainty = 1.0 / (1.0 / belief_drug.utz_uncertainty + 1.0 / observation_uncertainty)
        else:
            new_mean = observed_value
            new_uncertainty = observation_uncertainty
        return replace(
            belief_drug,
            utilization_per_week=new_mean,
            utz_uncertainty=new_uncertainty,
            runway_weeks=_compute_belief_runway(belief_drug.quantity_on_hand, new_mean)
        )
    elif observation_type == "supplier_reliability":
        alpha = 0.3
        if "primary" in observation_type:
            new_belief = (1 - alpha) * belief_drug.believed_primary_reliability + alpha * observed_value
            return replace(
                belief_drug,
                believed_primary_reliability=new_belief,
                believed_primary_reliability_uncertainty=max(0.01, belief_drug.believed_primary_reliability_uncertainty * 0.9)
            )
        else:
            new_belief = (1 - alpha) * belief_drug.believed_alternate_reliability + alpha * observed_value
            return replace(
                belief_drug,
                believed_alternate_reliability=new_belief,
                believed_alternate_reliability_uncertainty=max(0.01, belief_drug.believed_alternate_reliability_uncertainty * 0.9)
            )
    elif observation_type == "erd":
        alpha = 0.3
        new_erd = (1 - alpha) * belief_drug.erd_weeks + alpha * observed_value
        new_erd_uncertainty = max(0.1, belief_drug.erd_uncertainty * 0.9)
        return replace(
            belief_drug,
            erd_weeks=new_erd,
            erd_uncertainty=new_erd_uncertainty
        )
    return belief_drug

def update_erd_beliefs(belief_drug: BeliefDrug, true_primary_supplier, true_alternate_supplier, week_index: int) -> BeliefDrug:
    """ERD Belief Updater"""
    primary_erd = _get_current_erd(true_primary_supplier, week_index)
    alternate_erd = _get_current_erd(true_alternate_supplier, week_index)
    alpha = 0.3
    new_primary_erd = (1 - alpha) * belief_drug.erd_weeks + alpha * primary_erd
    new_alternate_erd = (1 - alpha) * belief_drug.alternate_supplier_erd_weeks + alpha * alternate_erd
    if belief_drug.active_supplier == "alternate":
        new_alternate_erd = (1 - alpha * 2) * belief_drug.alternate_supplier_erd_weeks + alpha * 2 * alternate_erd
    else:
        new_primary_erd = (1 - alpha * 2) * belief_drug.erd_weeks + alpha * 2 * primary_erd
    return replace(
        belief_drug,
        erd_weeks=new_primary_erd,
        erd_uncertainty=max(0.1, belief_drug.erd_uncertainty * 0.95),
        alternate_supplier_erd_weeks=new_alternate_erd,
        alternate_supplier_erd_uncertainty=max(0.1, belief_drug.alternate_supplier_erd_uncertainty * 0.95)
    )

def update_action_history(belief_drug: BeliefDrug, actions: List[Action], week_index: int) -> BeliefDrug:
    """Action History Updater"""
    action_names = [action.kind.name for action in actions]
    history_entry = {
        "week": week_index,
        "actions": action_names,
        "runway_before": belief_drug.runway_weeks,
        "erd_before": belief_drug.erd_weeks,
        "lma_type_before": belief_drug.current_lma_type,
        "timestamp": week_index
    }
    new_action_history = belief_drug.action_history[-9:] + [history_entry]
    new_action_frequency = belief_drug.action_frequency.copy()
    for action_name in action_names:
        new_action_frequency[action_name] = new_action_frequency.get(action_name, 0) + 1
    if belief_drug.last_week_actions == action_names and action_names:
        new_consecutive = belief_drug.consecutive_identical_actions + 1
    else:
        new_consecutive = 1 if action_names else 0
    return replace(
        belief_drug,
        last_week_actions=action_names,
        action_history=new_action_history,
        action_frequency=new_action_frequency,
        consecutive_identical_actions=new_consecutive,
        last_action_week=week_index if action_names else belief_drug.last_action_week
    )

def update_audit_results(belief_drug: BeliefDrug, true_qoh: float, true_utz: float, 
                        prev_qoh: float, prev_utz: float) -> BeliefDrug:
    """Audit Results Updater"""
    audit_results = {
        "qoh_change": true_qoh - prev_qoh,
        "utz_change": true_utz - prev_utz,
        "qoh_ratio": true_qoh / max(prev_qoh, 1e-9),
        "utz_ratio": true_utz / max(prev_utz, 1e-9),
        "runway_change": (true_qoh / max(true_utz, 1e-9)) - (prev_qoh / max(prev_utz, 1e-9))
    }
    return replace(
        belief_drug,
        audit_results=audit_results
    )

def update_shipment_notification(belief_drug: BeliefDrug, shipment_arrived: bool, 
                                shipment_amount: float = 0.0) -> BeliefDrug:
    """Shipment Arrival Notification Updater"""
    runway_improvement = 0.0
    if shipment_arrived and shipment_amount > 0:
        runway_improvement = shipment_amount / max(belief_drug.utilization_per_week, 1e-9)
    return replace(
        belief_drug,
        shipment_arrival_notification=shipment_arrived,
        shipment_arrival_amount=shipment_amount,
        audit_results={
            **belief_drug.audit_results,
            "last_shipment_arrived": shipment_arrived,
            "last_shipment_amount": shipment_amount,
            "last_shipment_runway_improvement": runway_improvement
        }
    )

def update_supplier_reliability_from_delivery(belief_drug: BeliefDrug, delivery_arrived: bool, 
                                            supplier_type: str, was_delivery_expected: bool = True) -> BeliefDrug:
    """Supplier Reliability Updater"""
    alpha = 0.1
    if not was_delivery_expected:
        return belief_drug
    if supplier_type == "primary":
        if delivery_arrived:
            new_reliability = min(1.0, belief_drug.believed_primary_reliability + alpha * 0.1)
        else:
            new_reliability = max(0.1, belief_drug.believed_primary_reliability - alpha * 0.2)
        return replace(
            belief_drug,
            believed_primary_reliability=new_reliability,
            believed_primary_reliability_uncertainty=max(0.01, belief_drug.believed_primary_reliability_uncertainty * 0.95)
        )
    else:
        if delivery_arrived:
            new_reliability = min(1.0, belief_drug.believed_alternate_reliability + alpha * 0.1)
        else:
            new_reliability = max(0.1, belief_drug.believed_alternate_reliability - alpha * 0.2)
        return replace(
            belief_drug,
            believed_alternate_reliability=new_reliability,
            believed_alternate_reliability_uncertainty=max(0.01, belief_drug.believed_alternate_reliability_uncertainty * 0.95)
        )

def has_recently_taken_action(belief_drug: BeliefDrug, action_name: str, weeks_back: int = 2) -> bool:
    """Recent Action Checker"""
    recent_weeks = [entry for entry in belief_drug.action_history 
                   if entry["week"] >= belief_drug.last_action_week - weeks_back]
    return any(action_name in entry["actions"] for entry in recent_weeks)

def get_action_frequency(belief_drug: BeliefDrug, action_name: str) -> int:
    """Action Frequency Getter"""
    return belief_drug.action_frequency.get(action_name, 0)

def is_repeating_actions(belief_drug: BeliefDrug, threshold: int = 3) -> bool:
    """Repeating Actions Checker"""
    return belief_drug.consecutive_identical_actions >= threshold

def get_recent_action_pattern(belief_drug: BeliefDrug, weeks_back: int = 4) -> List[str]:
    """Recent Action Pattern Getter"""
    recent_weeks = [entry for entry in belief_drug.action_history 
                   if entry["week"] >= belief_drug.last_action_week - weeks_back]
    return [entry["actions"] for entry in recent_weeks]

def should_avoid_action(belief_drug: BeliefDrug, action_name: str, 
                       max_frequency: int = 5, recent_threshold: int = 2) -> bool:
    """Avoid Action Checker"""
    if get_action_frequency(belief_drug, action_name) >= max_frequency:
        return True
    if has_recently_taken_action(belief_drug, action_name, recent_threshold):
        return True
    if is_repeating_actions(belief_drug) and action_name in belief_drug.last_week_actions:
        return True
    return False

def get_action_effectiveness(belief_drug: BeliefDrug, action_name: str) -> Dict[str, float]:
    """Action Effectiveness Getter"""
    action_entries = [entry for entry in belief_drug.action_history 
                     if action_name in entry["actions"]]
    if not action_entries:
        return {"effectiveness": 0.0, "sample_size": 0}
    effectiveness_scores = []
    for i, entry in enumerate(action_entries):
        if i < len(action_entries) - 1:
            next_entry = belief_drug.action_history[belief_drug.action_history.index(entry) + 1]
            runway_change = next_entry.get("runway_before", 0) - entry.get("runway_before", 0)
            effectiveness_scores.append(max(0, runway_change))
    avg_effectiveness = sum(effectiveness_scores) / len(effectiveness_scores) if effectiveness_scores else 0.0
    return {
        "effectiveness": avg_effectiveness,
        "sample_size": len(action_entries),
        "avg_runway_change": avg_effectiveness
    }

def has_shipment_arrived(belief_drug: BeliefDrug) -> bool:
    """Shipment Arrived Checker"""
    return belief_drug.shipment_arrival_notification

def get_shipment_amount(belief_drug: BeliefDrug) -> float:
    """Shipment Amount Getter"""
    return belief_drug.shipment_arrival_amount

def get_shipment_runway_improvement(belief_drug: BeliefDrug) -> float:
    """Shipment Runway Improvement Getter"""
    return belief_drug.audit_results.get("last_shipment_runway_improvement", 0.0)

def was_shipment_successful(belief_drug: BeliefDrug) -> bool:
    """Shipment Successful Checker"""
    if not has_shipment_arrived(belief_drug):
        return False
    improvement = get_shipment_runway_improvement(belief_drug)
    return improvement >= 1.0

def get_shipment_history(belief_drug: BeliefDrug, weeks_back: int = 4) -> List[Dict[str, any]]:
    """Shipment History Getter"""
    shipment_history = []
    for entry in belief_drug.action_history[-weeks_back:]:
        if "shipment_arrived" in entry:
            shipment_history.append({
                "week": entry["week"],
                "shipment_arrived": entry.get("shipment_arrived", False),
                "shipment_amount": entry.get("shipment_amount", 0.0),
                "runway_improvement": entry.get("runway_improvement", 0.0)
            })
    return shipment_history

def apply_belief_demand_action(belief_drug: BeliefDrug, action: Action, current_week: int) -> BeliefDrug:
    """Demand-Side Actions Applier"""
    if action == Action.IMPLEMENT_SOFT_LMA:
        if belief_drug.current_lma_type != "none":
            return belief_drug
        reduction_factor = 0.75 + random.normalvariate(0, 0.1)
        new_utz = max(
            belief_drug.min_utilization_per_week if hasattr(belief_drug, 'min_utilization_per_week') else belief_drug.utilization_per_week * 0.5,
            belief_drug.utilization_per_week * reduction_factor
        )
        return replace(
            belief_drug,
            utilization_per_week=new_utz,
            runway_weeks=_compute_belief_runway(belief_drug.quantity_on_hand, new_utz),
            current_lma_type="soft",
            lma_implemented_week=current_week
        )
    elif action == Action.IMPLEMENT_HARD_LMA:
        if belief_drug.current_lma_type == "hard":
            return belief_drug
        if current_week - belief_drug.lma_implemented_week < 2:
            return belief_drug
        reduction_factor = 0.25 + random.normalvariate(0, 0.05)
        new_utz = max(
            belief_drug.min_utilization_per_week if hasattr(belief_drug, 'min_utilization_per_week') else belief_drug.utilization_per_week * 0.5,
            belief_drug.utilization_per_week * reduction_factor
        )
        return replace(
            belief_drug,
            utilization_per_week=new_utz,
            runway_weeks=_compute_belief_runway(belief_drug.quantity_on_hand, new_utz),
            current_lma_type="hard",
            lma_implemented_week=current_week
        )
    elif action == Action.REMOVE_LMA:
        if belief_drug.current_lma_type == "none":
            return belief_drug
        baseline_utz = belief_drug.utilization_per_week * 1.2
        return replace(
            belief_drug,
            utilization_per_week=baseline_utz,
            runway_weeks=_compute_belief_runway(belief_drug.quantity_on_hand, baseline_utz),
            current_lma_type="none",
            lma_implemented_week=-1
        )
    return belief_drug

def apply_belief_supply_action(belief_drug: BeliefDrug, action: Action, current_week: int) -> BeliefDrug:
    """Supply-Side Actions Applier"""
    if action == Action.SWITCH_SUPPLIER:
        if belief_drug.active_supplier == "alternate":
            return belief_drug
        return replace(
            belief_drug,
            active_supplier="alternate",
            use_alternate=True,
            switch_pending_until_week=current_week + 2,
            supplier_switch_implemented_week=current_week,
            supplier_switch_week=current_week
        )
    elif action == Action.SWITCH_BACK_TO_PRIMARY:
        if belief_drug.active_supplier == "primary":
            return belief_drug
        return replace(
            belief_drug,
            active_supplier="primary",
            use_alternate=False,
            switch_pending_until_week=current_week + 2,
            supplier_switch_implemented_week=current_week,
            supplier_switch_week=current_week
        )
    elif action in [Action.CONTACT_MANUFACTURER_DIRECT, Action.REQUEST_FROM_RESERVE_WAREHOUSE, 
                   Action.REQUEST_LOAN_FROM_OTHER_HOSPITALS, Action.GREY_MARKET_PURCHASE]:
        boost_multipliers = {
            Action.CONTACT_MANUFACTURER_DIRECT: 2.0,
            Action.REQUEST_FROM_RESERVE_WAREHOUSE: 3.0,
            Action.REQUEST_LOAN_FROM_OTHER_HOSPITALS: 2.5,
            Action.GREY_MARKET_PURCHASE: 4.0
        }
        boost = boost_multipliers[action] * belief_drug.utilization_per_week
        new_qoh = belief_drug.quantity_on_hand + boost
        return replace(
            belief_drug,
            quantity_on_hand=new_qoh,
            runway_weeks=_compute_belief_runway(new_qoh, belief_drug.utilization_per_week)
        )
    elif action == Action.QUERY_ALTERNATE_SUPPLIER:
        return replace(
            belief_drug,
            last_supplier_query_week=current_week
        )
    return belief_drug

def step_belief_state(belief_state: BeliefEnvState, actions: Dict[str, list], 
                     observations: Optional[Dict[str, Dict[str, float]]] = None,
                     true_state=None, delivery_info: Optional[Dict[str, Dict[str, any]]] = None) -> BeliefEnvState:
    """Belief State Stepper"""
    new_drugs = {}
    for drug_name, belief_drug in belief_state.drugs.items():
        drug_actions = actions.get(drug_name, [])
        updated_drug = belief_drug
        updated_drug = update_action_history(updated_drug, drug_actions, belief_state.week_index)
        if true_state and drug_name in true_state.drugs:
            true_drug = true_state.drugs[drug_name]
            updated_drug = update_erd_beliefs(updated_drug, true_drug.primary_supplier, 
                                            true_drug.alternate_supplier, belief_state.week_index)
        if delivery_info and drug_name in delivery_info:
            delivery_data = delivery_info[drug_name]
            if delivery_data['delivery_arrived'] and delivery_data['delivery_amount'] > 0:
                updated_drug = replace(
                    updated_drug,
                    quantity_on_hand=updated_drug.quantity_on_hand + delivery_data['delivery_amount'],
                    runway_weeks=_compute_belief_runway(
                        updated_drug.quantity_on_hand + delivery_data['delivery_amount'], 
                        updated_drug.utilization_per_week
                    )
                )
            updated_drug = update_shipment_notification(
                updated_drug, 
                delivery_data['delivery_arrived'], 
                delivery_data['delivery_amount']
            )
            if true_state and drug_name in true_state.drugs:
                true_drug = true_state.drugs[drug_name]
                supplier_type = "primary" if true_drug.active_supplier == "primary" else "alternate"
                supplier = true_drug.primary_supplier if supplier_type == "primary" else true_drug.alternate_supplier
                patterns = SUPPLIER_ERD_PATTERNS.get(supplier.type.lower(), SUPPLIER_ERD_PATTERNS["accurate"])
                pattern = patterns[supplier.current_erd_week % len(patterns)]
                current_week_in_cycle = belief_state.week_index % len(pattern)
                was_delivery_expected = (current_week_in_cycle == len(pattern) - 1)
                updated_drug = update_supplier_reliability_from_delivery(
                    updated_drug, 
                    delivery_data['delivery_arrived'], 
                    supplier_type,
                    was_delivery_expected
                )
        is_audited = any(action.kind == Action.AUDIT_QOH_AND_UTZ for action in drug_actions)
        for action in drug_actions:
            action_type = action.kind
            if action_type == Action.AUDIT_QOH_AND_UTZ:
                if true_state and drug_name in true_state.drugs:
                    true_drug = true_state.drugs[drug_name]
                    updated_drug = _audit_belief(updated_drug, true_drug.quantity_on_hand, true_drug.utilization_per_week)
                    updated_drug = update_audit_results(updated_drug, true_drug.quantity_on_hand, 
                                                      true_drug.utilization_per_week,
                                                      belief_drug.quantity_on_hand, 
                                                      belief_drug.utilization_per_week)
                else:
                    updated_drug = _audit_belief(updated_drug, 0, 0)
            elif action_type in [Action.IMPLEMENT_SOFT_LMA, Action.IMPLEMENT_HARD_LMA, Action.REMOVE_LMA]:
                updated_drug = apply_belief_demand_action(updated_drug, action_type, belief_state.week_index)
            elif action_type in [Action.QUERY_ALTERNATE_SUPPLIER, Action.SWITCH_SUPPLIER, Action.SWITCH_BACK_TO_PRIMARY, Action.CONTACT_MANUFACTURER_DIRECT, Action.REQUEST_FROM_RESERVE_WAREHOUSE, Action.REQUEST_LOAN_FROM_OTHER_HOSPITALS, Action.GREY_MARKET_PURCHASE]:
                updated_drug = apply_belief_supply_action(updated_drug, action_type, belief_state.week_index)
        if not is_audited:
            updated_drug = _belief_decay(updated_drug)
        if observations and drug_name in observations:
            drug_observations = observations[drug_name]
            for obs_type, obs_value in drug_observations.items():
                if isinstance(obs_value, dict) and "value" in obs_value and "uncertainty" in obs_value:
                    updated_drug = update_belief_from_observation(
                        updated_drug, obs_type, obs_value["value"], obs_value["uncertainty"]
                    )
        new_drugs[drug_name] = updated_drug
    return replace(
        belief_state,
        week_index=belief_state.week_index + 1,
        drugs=new_drugs
    )
