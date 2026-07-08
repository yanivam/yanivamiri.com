"""Weekly decision orchestration for drug shortage management.

This module orchestrates the weekly decision cycle:
1. Load scenario configuration
2. Create true and belief states
3. Get agent's planned actions based on belief state
4. Apply actions to true state environment
5. Update belief state of the agent based on observations and actions taken
6. Compute metrics and rewards
"""

from __future__ import annotations

import json
import random
import os
from dataclasses import replace

from src.agents.base import BaseAgent
from src.environments.belief_state import BeliefDrug, BeliefSupplier, BeliefEnvState
from src.environments.belief_updates import step_belief_state
from src.environments.rewards import compute_week_metrics
from src.environments.transition_functions import step_true_state
from src.environments.true_state import TrueDrug, TrueSupplier, TrueEnvState
from src.environments.state import SUPPLIER_ERD_PATTERNS

def _get_current_erd_from_supplier_type(supplier_type: str, current_week: int = 0) -> float:
    """ERD Getter"""
    if supplier_type in SUPPLIER_ERD_PATTERNS:
        patterns = SUPPLIER_ERD_PATTERNS[supplier_type]
        pattern = patterns[0] if patterns else [5] * 6
        cycle_position = current_week % len(pattern)
        return pattern[cycle_position]
    else:
        return 5.0

def _scenario_path_for_name(scenario: str) -> str:
    """Scenario Getter"""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    scenarios_dir = os.path.join(project_root, "configs", "scenarios")
    if "_" in scenario:
        directories = sorted([item for item in os.listdir(scenarios_dir) 
                             if os.path.isdir(os.path.join(scenarios_dir, item))], 
                            key=len, reverse=True)
        for item in directories:
            item_path = os.path.join(scenarios_dir, item)
            if scenario.startswith(item):
                name = scenario[len(item)+1:]
                return os.path.join(item_path, f"{name}.json")
        parts = scenario.split("_", 1)
        if len(parts) == 2:
            category, name = parts
            return os.path.join(scenarios_dir, category, f"{name}.json")
    
    return os.path.join(scenarios_dir, f"{scenario}.json")

def create_state_from_config(rng_seed: int, suppliers_cfg: dict, drugs_cfg: list):
    """True State Creator"""
    drugs = {}
    for drug_cfg in drugs_cfg:
        drugs[drug_cfg["name"]] = _create_true_drug(drug_cfg, suppliers_cfg)
    return TrueEnvState(0, drugs, rng_seed)

def get_agent_observable_state(rng_seed: int, suppliers_cfg: dict, drugs_cfg: list):
    """Belief State Creator"""
    drugs = {}
    for drug_cfg in drugs_cfg:
        drugs[drug_cfg["name"]] = _create_belief_drug(drug_cfg, suppliers_cfg)
    return BeliefEnvState(0, drugs, rng_seed)

def get_true_state_for_evaluation(state: TrueEnvState):
    """True State Deep Copy"""
    return replace(state)

def apply_actions_to_state(state: TrueEnvState, actions: dict, delivery_config: dict):
    """Action Applicator"""
    return step_true_state(state, actions, delivery_config)

def update_belief_state_from_observations(belief_state: BeliefEnvState, actions: dict, 
                                        new_true_state: TrueEnvState, delivery_info: dict) -> BeliefEnvState:
    """Belief State Updater"""
    return step_belief_state(
        belief_state=belief_state,
        actions=actions,
        observations=None,
        true_state=new_true_state,
        delivery_info=delivery_info
    )

def make_weekly_decisions(agent_obj: BaseAgent, scenario: str, week: int, 
                         prev_true_state: TrueEnvState = None, prev_belief_state: BeliefEnvState = None):
    """Weekly Decision Maker"""
    config = _load_scenario_config(scenario)
    if week == 0 or prev_true_state is None:
        true_state = create_state_from_config(
            rng_seed=config["rng_seed"],
            suppliers_cfg=config["suppliers_cfg"],
            drugs_cfg=config["drugs_cfg"],
        )
        belief_state = get_agent_observable_state(
            rng_seed=config["rng_seed"],
            suppliers_cfg=config["suppliers_cfg"],
            drugs_cfg=config["drugs_cfg"],
        )
    else:
        true_state = prev_true_state
        belief_state = prev_belief_state
    actions = agent_obj.select_actions(belief_state)
    new_true_state, delivery_info = apply_actions_to_state(true_state, actions, config["delivery_config"])
    new_belief_state = update_belief_state_from_observations(belief_state, actions, new_true_state, delivery_info)
    metrics = compute_week_metrics(true_state, actions, new_true_state)
    if hasattr(agent_obj, 'update_attention_weights') and 'total_reward' in metrics:
        agent_obj.update_attention_weights(metrics['total_reward'], new_belief_state)
    return new_true_state, new_belief_state, metrics

def _load_scenario_config(scenario: str) -> dict:
    """Scenario Loader. OVERRIDE_RNG_SEED in env overrides rng_seed for reproducibility."""
    path = _scenario_path_for_name(scenario)
    with open(path, "r") as f:
        cfg = json.load(f)
    rng_seed = int(cfg.get("rng_seed", 123))
    if os.environ.get("OVERRIDE_RNG_SEED"):
        try:
            rng_seed = int(os.environ["OVERRIDE_RNG_SEED"])
        except ValueError:
            pass
    return {
        "name": cfg.get("name", "toy_test_audit_action"),
        "description": cfg.get("description", "Toy scenario for testing"),
        "rng_seed": rng_seed,
        "delivery_config": cfg.get("delivery_config", {}),
        "suppliers_cfg": cfg.get("suppliers", {}),
        "drugs_cfg": cfg.get("drugs", []),
    }

def _create_true_drug(drug_cfg: dict, suppliers_cfg: dict) -> TrueDrug:
    """True Drug Creator"""
    return TrueDrug(
        name=drug_cfg["name"],
        quantity_on_hand=drug_cfg["qoh"],
        utilization_per_week=drug_cfg["utz"],
        runway_weeks=drug_cfg["qoh"] / drug_cfg["utz"],
        base_utilization_per_week=drug_cfg["base_utilization_per_week"],
        min_utilization_per_week=drug_cfg["min_utilization_per_week"],
        primary_supplier=_create_true_supplier(drug_cfg["primary_supplier"], suppliers_cfg.get("primary", {})),
        alternate_supplier=_create_true_supplier(drug_cfg["alternate_supplier"], suppliers_cfg.get("alternate", {})),
        clinical_impact=drug_cfg["clinical_impact"],
        shortage_count=0,
        total_weeks_tracked=0,
        current_lma_type="none",
        lma_implemented_week=-1,
        active_supplier="primary",
        supplier_switch_week=0,
        utz_evolution_config={"type": "seasonal", "pattern": drug_cfg["base_utz_evolution_config"]},
        utz_evolution_week=0,
    )

def _create_belief_drug(drug_cfg: dict, suppliers_cfg: dict) -> BeliefDrug:
    """Belief Drug Creator"""
    return BeliefDrug(
        name=drug_cfg["name"],
        quantity_on_hand=drug_cfg["qoh"],
        utilization_per_week=drug_cfg["utz"],
        runway_weeks=drug_cfg["qoh"] / drug_cfg["utz"],
        qoh_uncertainty=drug_cfg["qoh_uncertainty"],
        utz_uncertainty=drug_cfg["utz_uncertainty"],
        runway_uncertainty=drug_cfg["runway_uncertainty"],
        primary_supplier=_create_belief_supplier(drug_cfg["primary_supplier"]),
        alternate_supplier=_create_belief_supplier(drug_cfg["alternate_supplier"]),
        use_alternate=False,
        switch_pending_until_week=0,
        believed_primary_reliability=max(0.3, 0.95 - drug_cfg["erd_uncertainty"]),  
        believed_alternate_reliability=max(0.3, 0.95 - drug_cfg["alternate_supplier_erd_uncertainty"]), 
        believed_primary_reliability_uncertainty=drug_cfg["erd_uncertainty"],
        believed_alternate_reliability_uncertainty=drug_cfg["alternate_supplier_erd_uncertainty"],
        last_supplier_query_week=0,
        clinical_impact=drug_cfg["clinical_impact"],
        erd_weeks=_get_current_erd_from_supplier_type(suppliers_cfg.get("primary", {}).get("type", "accurate"), 2),
        erd_uncertainty=drug_cfg["erd_uncertainty"],
        alternate_supplier_erd_weeks=_get_current_erd_from_supplier_type(suppliers_cfg.get("alternate", {}).get("type", "accurate"), 2),
        alternate_supplier_erd_uncertainty=drug_cfg["alternate_supplier_erd_uncertainty"],
        delivery_delay_weeks=drug_cfg["delivery_delay_weeks"],
        delivery_delay_uncertainty=drug_cfg["delivery_delay_uncertainty"],
        expected_delivery_amount=drug_cfg["expected_delivery_amount"],
        expected_delivery_amount_uncertainty=drug_cfg["expected_delivery_amount_uncertainty"],
        drug_reputation=drug_cfg["drug_reputation"],
        shortage_count=0,
        total_weeks_tracked=0,
        current_lma_type="none",
        lma_implemented_week=-1,
        supplier_switch_implemented_week=0,
        active_supplier="primary",
        supplier_switch_week=0,
        last_week_actions=[],
        action_history=[],
        audit_results={},
        shipment_arrival_notification=False,
        shipment_arrival_amount=0.0,
        consecutive_identical_actions=0,
        action_frequency={},
        weeks_since_audit=drug_cfg["weeks_since_audit"],
        emergency_supply_used=False,
        last_action_week=0,
        learned_primary_type=drug_cfg["learned_primary_type"],
        learned_alternate_type=drug_cfg["learned_alternate_type"],
        base_utz_evolution_config=drug_cfg.get("base_utz_evolution_config", None),
        utz_evolution_week=0,
    )

def _create_true_supplier(supplier_name: str, supplier_cfg: dict) -> TrueSupplier:
    """True Supplier Creator"""
    supplier_type = supplier_cfg.get("type", "accurate")
    if supplier_type in SUPPLIER_ERD_PATTERNS:
        erd_pattern = random.choice(SUPPLIER_ERD_PATTERNS[supplier_type])
    else:
        erd_pattern = [5] * 6
    return TrueSupplier(
        name=supplier_name,
        type=supplier_type,
        current_erd_pattern=erd_pattern,
        current_erd_week=0
    )

def _create_belief_supplier(supplier_name: str) -> BeliefSupplier:
    """Belief Supplier Creator"""
    return BeliefSupplier(
        name=supplier_name,
        believed_reliability=0.9,
        believed_erd_uncertainty=0.1
    )
