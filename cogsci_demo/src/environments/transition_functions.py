"""Transition Functions"""

from dataclasses import replace
from typing import Dict, Tuple
import math
import random

from src.environments.actions import Action
from src.environments.state import SUPPLIER_ERD_PATTERNS
from src.environments.true_state import TrueDrug, TrueSupplier, TrueEnvState

def _compute_runway(qoh: float, utz: float) -> float:
    """Runway Getter"""
    if utz <= 1e-9:
        return math.inf
    return qoh / utz

def _get_supplier_reliability(supplier: TrueSupplier) -> float:
    """Supplier Reliability Getter"""
    reliability_map = {
        "accurate": 1.0,
        "pushback": 1.0,
        "pushback_reverse": 1.0,
        "random": 1.0
    }
    return reliability_map.get(supplier.type.lower(), 0.9)

def _get_current_erd(supplier: TrueSupplier, week_index: int) -> int:
    """Current ERD Getter"""
    patterns = SUPPLIER_ERD_PATTERNS.get(supplier.type.lower(), SUPPLIER_ERD_PATTERNS["accurate"])
    pattern = patterns[supplier.current_erd_week % len(patterns)]
    current_week_in_cycle = (week_index + supplier.current_erd_week) % len(pattern)
    return pattern[current_week_in_cycle]

def apply_utz_evolution(true_drug: TrueDrug, week_index: int) -> TrueDrug:
    """UTZ Evolution Applier"""
    if true_drug.current_lma_type != "none":
        return true_drug
    if not true_drug.utz_evolution_config:
        return true_drug
    config = true_drug.utz_evolution_config
    evolution_type = config.get("type", "none")
    if evolution_type == "linear_drift":
        return _apply_linear_drift_evolution(true_drug, week_index, config)
    elif evolution_type == "step_change":
        return _apply_step_change_evolution(true_drug, week_index, config)
    elif evolution_type == "seasonal":
        return _apply_seasonal_evolution(true_drug, week_index, config)
    else:
        return true_drug

def _apply_linear_drift_evolution(true_drug: TrueDrug, week_index: int, config: dict) -> TrueDrug:
    """Linear Drift Evolution Applier"""
    start_week = config.get("start_week", 0)
    end_week = config.get("end_week", float('inf'))
    drift_rate = config.get("drift_rate", 0.0)
    if week_index < start_week or week_index > end_week:
        return true_drug
    weeks_into_drift = week_index - start_week + 1
    new_utz = true_drug.base_utilization_per_week * (1 + drift_rate * weeks_into_drift)
    new_runway = _compute_runway(true_drug.quantity_on_hand, new_utz)
    return replace(
        true_drug,
        utilization_per_week=new_utz,
        runway_weeks=new_runway,
        utz_evolution_week=true_drug.utz_evolution_week + 1
    )

def _apply_step_change_evolution(true_drug: TrueDrug, week_index: int, config: dict) -> TrueDrug:
    """Step Change Evolution Executor"""
    change_week = config.get("week", 0)
    new_utz = config.get("new_utz", true_drug.utilization_per_week)
    if week_index != change_week:
        return true_drug
    new_runway = _compute_runway(true_drug.quantity_on_hand, new_utz)
    return replace(
        true_drug,
        utilization_per_week=new_utz,
        runway_weeks=new_runway,
        utz_evolution_week=true_drug.utz_evolution_week + 1
    )

def _apply_seasonal_evolution(true_drug: TrueDrug, week_index: int, config: dict) -> TrueDrug:
    """Seasonal Pattern Evolution Executor"""
    pattern = config.get("pattern", [true_drug.utilization_per_week])
    if not pattern:
        return true_drug
    pattern_index = week_index % len(pattern)
    new_utz = pattern[pattern_index]
    new_runway = _compute_runway(true_drug.quantity_on_hand, new_utz)
    return replace(
        true_drug,
        utilization_per_week=new_utz,
        runway_weeks=new_runway,
        utz_evolution_week=true_drug.utz_evolution_week + 1
    )

def apply_consumption(true_drug: TrueDrug) -> TrueDrug:
    """Consumption Executor"""
    new_qoh = max(0.0, true_drug.quantity_on_hand - true_drug.utilization_per_week)
    new_runway = _compute_runway(new_qoh, true_drug.utilization_per_week)
    return replace(
        true_drug,
        quantity_on_hand=new_qoh,
        runway_weeks=new_runway
    )

def apply_delivery(true_drug: TrueDrug, week_index: int, rng: random.Random, delivery_config: dict = None) -> Tuple[TrueDrug, bool, float]:
    """Delivery Executor"""
    if true_drug.active_supplier == "alternate":
        supplier = true_drug.alternate_supplier
    else:
        supplier = true_drug.primary_supplier
    patterns = SUPPLIER_ERD_PATTERNS.get(supplier.type.lower(), SUPPLIER_ERD_PATTERNS["accurate"])
    pattern = patterns[supplier.current_erd_week % len(patterns)]
    current_week_in_cycle = week_index % len(pattern)
    is_delivery_week = (current_week_in_cycle == len(pattern) - 1)
    delivery_arrived = False
    delivery_amount = 0.0
    if is_delivery_week:
        if delivery_config and "weeks_of_supply" in delivery_config:
            weeks_of_supply = delivery_config["weeks_of_supply"]
            delivery_amount = true_drug.utilization_per_week * weeks_of_supply
        elif delivery_config and "fixed_amount" in delivery_config:
            delivery_amount = delivery_config["fixed_amount"]
        else:
            weeks_of_supply = 14
            delivery_amount = true_drug.utilization_per_week * weeks_of_supply
        reliability = _get_supplier_reliability(supplier)
        if rng.random() < reliability:
            new_qoh = true_drug.quantity_on_hand + delivery_amount
            delivery_arrived = True
        else:
            new_qoh = true_drug.quantity_on_hand
    else:
        new_qoh = true_drug.quantity_on_hand
    new_runway = _compute_runway(new_qoh, true_drug.utilization_per_week)
    patterns = SUPPLIER_ERD_PATTERNS.get(supplier.type.lower(), SUPPLIER_ERD_PATTERNS["accurate"])
    new_supplier = replace(
        supplier,
        current_erd_week=(supplier.current_erd_week + 1) % len(patterns)
    )
    if true_drug.active_supplier == "primary":
        updated_drug = replace(
            true_drug,
            quantity_on_hand=new_qoh,
            runway_weeks=new_runway,
            primary_supplier=new_supplier
        )
    else:
        updated_drug = replace(
            true_drug,
            quantity_on_hand=new_qoh,
            runway_weeks=new_runway,
            alternate_supplier=new_supplier
        )
    return updated_drug, delivery_arrived, delivery_amount

def apply_utilization_reversion(true_drug: TrueDrug, rho: float = 0.2) -> TrueDrug:
    """Utilization Reversion Executor"""
    if true_drug.current_lma_type == "none":
        reversion_rate = rho
    elif true_drug.current_lma_type == "soft":
        reversion_rate = rho * 0.1
    elif true_drug.current_lma_type == "hard":
        reversion_rate = rho * 0.05
    else:
        reversion_rate = rho
    new_utz = max(
        true_drug.min_utilization_per_week,
        (1.0 - reversion_rate) * true_drug.utilization_per_week + reversion_rate * true_drug.base_utilization_per_week,
    )
    new_runway = _compute_runway(true_drug.quantity_on_hand, new_utz)
    return replace(
        true_drug,
        utilization_per_week=new_utz,
        runway_weeks=new_runway
    )

def apply_lma_action(true_drug: TrueDrug, action: Action, current_week: int) -> TrueDrug:
    """LMA Action Executor"""
    if action == Action.IMPLEMENT_SOFT_LMA:
        if true_drug.current_lma_type != "none":
            return true_drug
        new_utz = true_drug.utilization_per_week * (0.75 + random.normalvariate(0, 0.1))
        new_runway = _compute_runway(true_drug.quantity_on_hand, new_utz)
        return replace(
            true_drug,
            utilization_per_week=new_utz,
            runway_weeks=new_runway,
            current_lma_type="soft",
            lma_implemented_week=current_week
        )
    elif action == Action.IMPLEMENT_HARD_LMA:
        if true_drug.current_lma_type == "hard":
            return true_drug
        if true_drug.lma_implemented_week >= 0 and current_week - true_drug.lma_implemented_week < 2:
            return true_drug
        new_utz = true_drug.utilization_per_week * (0.25 + random.normalvariate(0, 0.05))
        new_runway = _compute_runway(true_drug.quantity_on_hand, new_utz)
        return replace(
            true_drug,
            utilization_per_week=new_utz,
            runway_weeks=new_runway,
            current_lma_type="hard",
            lma_implemented_week=current_week
        )
    elif action == Action.REMOVE_LMA:
        if true_drug.current_lma_type == "none":
            return true_drug
        new_runway = _compute_runway(true_drug.quantity_on_hand, true_drug.base_utilization_per_week)
        return replace(
            true_drug,
            utilization_per_week=true_drug.base_utilization_per_week,
            runway_weeks=new_runway,
            current_lma_type="none",
            lma_implemented_week=-1
        )
    return true_drug

def apply_supplier_switch_action(true_drug: TrueDrug, action: Action, current_week: int) -> TrueDrug:
    """Supplier Switching Action Executor"""
    if action == Action.SWITCH_SUPPLIER:
        return replace(
            true_drug,
            active_supplier="alternate",
            supplier_switch_week=current_week
        )
    elif action == Action.SWITCH_BACK_TO_PRIMARY:
        return replace(
            true_drug,
            active_supplier="primary",
            supplier_switch_week=current_week
        )
    return true_drug

def apply_audit_action(true_drug: TrueDrug, action: Action, current_week: int) -> TrueDrug:
    """Audit Action Executor"""
    if action == Action.AUDIT_QOH_AND_UTZ:
        return replace(
            true_drug,
            total_weeks_tracked=true_drug.total_weeks_tracked + 1
        )
    return true_drug

def apply_query_supplier_action(true_drug: TrueDrug, action: Action, current_week: int) -> TrueDrug:
    """Query Supplier Action Executor"""
    if action == Action.QUERY_ALTERNATE_SUPPLIER:
        return replace(
            true_drug,
            total_weeks_tracked=true_drug.total_weeks_tracked + 1
        )
    return true_drug

def apply_emergency_action(true_drug: TrueDrug, action: Action) -> TrueDrug:
    """Emergency Supply Action Executor"""
    if action == Action.CONTACT_MANUFACTURER_DIRECT:
        boost = 6.0 * true_drug.utilization_per_week
    elif action == Action.REQUEST_FROM_RESERVE_WAREHOUSE:
        boost = 6.0 * true_drug.utilization_per_week
    elif action == Action.REQUEST_LOAN_FROM_OTHER_HOSPITALS:
        boost = 6.0 * true_drug.utilization_per_week
    elif action == Action.GREY_MARKET_PURCHASE:
        boost = 14.0 * true_drug.utilization_per_week
    else:
        return true_drug
    new_qoh = true_drug.quantity_on_hand + boost
    new_runway = _compute_runway(new_qoh, true_drug.utilization_per_week)
    return replace(
        true_drug,
        quantity_on_hand=new_qoh,
        runway_weeks=new_runway
    )

def step_true_state(true_state: TrueEnvState, actions: Dict[str, list], delivery_config: dict = None) -> Tuple[TrueEnvState, dict]:
    """True State Executor"""
    rng = random.Random(true_state.rng_seed + true_state.week_index)
    new_drugs = {}
    delivery_info = {}
    for drug_name, true_drug in true_state.drugs.items():
        drug_actions = actions.get(drug_name, [])
        updated_drug = _apply_emergency_actions_only(true_drug, drug_actions, true_state.week_index)
        updated_drug = _apply_lma_actions_only(updated_drug, drug_actions, true_state.week_index)
        updated_drug, delivery_arrived, delivery_amount = _apply_environment_dynamics_with_delivery(updated_drug, true_state.week_index, rng, delivery_config)
        delivery_info[drug_name] = {
            'delivery_arrived': delivery_arrived,
            'delivery_amount': delivery_amount
        }
        updated_drug = _apply_remaining_actions(updated_drug, drug_actions, true_state.week_index)
        new_drugs[drug_name] = updated_drug
    new_true_state = replace(
        true_state,
        week_index=true_state.week_index + 1,
        drugs=new_drugs
    )
    return new_true_state, delivery_info

def _apply_environment_dynamics_with_delivery(true_drug: TrueDrug, week_index: int, rng: random.Random, delivery_config: dict) -> Tuple[TrueDrug, bool, float]:
    """Environment Dynamics Executor"""
    updated_drug = apply_utz_evolution(true_drug, week_index)
    updated_drug, delivery_arrived, delivery_amount = apply_delivery(updated_drug, week_index, rng, delivery_config)
    updated_drug = apply_consumption(updated_drug)
    updated_drug = apply_utilization_reversion(updated_drug)
    return updated_drug, delivery_arrived, delivery_amount

def _apply_emergency_actions_only(true_drug: TrueDrug, actions: list, week_index: int) -> TrueDrug:
    """Emergency Actions Executor"""
    updated_drug = true_drug
    for action in actions:
        if action.kind in [Action.REQUEST_FROM_RESERVE_WAREHOUSE, Action.REQUEST_LOAN_FROM_OTHER_HOSPITALS, Action.GREY_MARKET_PURCHASE, Action.CONTACT_MANUFACTURER_DIRECT]:
            updated_drug = apply_emergency_action(updated_drug, action.kind)
    return updated_drug

def _apply_lma_actions_only(true_drug: TrueDrug, actions: list, week_index: int) -> TrueDrug:
    """LMA Actions Only Executor"""
    updated_drug = true_drug
    for action in actions:
        if action.kind in [Action.IMPLEMENT_SOFT_LMA, Action.IMPLEMENT_HARD_LMA, Action.REMOVE_LMA]:
            updated_drug = apply_lma_action(updated_drug, action.kind, week_index)
    return updated_drug

def _apply_remaining_actions(true_drug: TrueDrug, actions: list, week_index: int) -> TrueDrug:
    """Remaining Non-Emergency Actions Executor"""
    updated_drug = true_drug
    for action in actions:
        if action.kind in [Action.SWITCH_SUPPLIER, Action.SWITCH_BACK_TO_PRIMARY]:
            updated_drug = apply_supplier_switch_action(updated_drug, action.kind, week_index)
        elif action.kind == Action.AUDIT_QOH_AND_UTZ:
            updated_drug = apply_audit_action(updated_drug, action.kind, week_index)
        elif action.kind == Action.QUERY_ALTERNATE_SUPPLIER:
            updated_drug = apply_query_supplier_action(updated_drug, action.kind, week_index)
    return updated_drug
