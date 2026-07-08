"""Agent belief state"""

from dataclasses import dataclass, replace
from typing import List, Dict, Any
import math

@dataclass
class BeliefSupplier:
    name: str
    believed_reliability: float  
    believed_erd_uncertainty: float  
    supplier_type: str = "accurate"  

@dataclass
class BeliefDrug:
    name: str
    quantity_on_hand: float  
    utilization_per_week: float  
    runway_weeks: float 
    qoh_uncertainty: float  
    utz_uncertainty: float  
    runway_uncertainty: float  
    primary_supplier: BeliefSupplier  
    alternate_supplier: BeliefSupplier  
    use_alternate: bool  
    switch_pending_until_week: int  
    believed_primary_reliability: float  
    believed_alternate_reliability: float  
    believed_primary_reliability_uncertainty: float  
    believed_alternate_reliability_uncertainty: float  
    last_supplier_query_week: int  
    clinical_impact: float  
    erd_weeks: float  
    erd_uncertainty: float  
    alternate_supplier_erd_weeks: float  
    alternate_supplier_erd_uncertainty: float  
    delivery_delay_weeks: float  
    delivery_delay_uncertainty: float  
    expected_delivery_amount: float  
    expected_delivery_amount_uncertainty: float  
    drug_reputation: float  
    shortage_count: int  
    total_weeks_tracked: int  
    current_lma_type: str  
    lma_implemented_week: int  
    supplier_switch_implemented_week: int  
    active_supplier: str  
    supplier_switch_week: int  
    last_week_actions: List[str]  
    action_history: List[Dict[str, Any]]  
    audit_results: Dict[str, Any]  
    shipment_arrival_notification: bool  
    shipment_arrival_amount: float  
    consecutive_identical_actions: int  
    action_frequency: Dict[str, int]  
    last_action_week: int  
    learned_primary_type: str  
    learned_alternate_type: str  
    weeks_since_audit: int = 0  
    emergency_supply_used: bool = False
    base_utz_evolution_config: List[float] = None
    utz_evolution_week: int = 0  
    
@dataclass
class BeliefEnvState:
    week_index: int  
    drugs: Dict[str, BeliefDrug]  
    rng_seed: int = 123  

def _compute_belief_runway(qoh: float, utz: float) -> float:
    """Belief Runwway Getter"""
    if utz <= 1e-9:
        return math.inf
    return qoh / utz

def _belief_decay(drug: BeliefDrug) -> BeliefDrug:
    """Belief Decay Applier"""
    updated_qoh = max(0, drug.quantity_on_hand - drug.utilization_per_week)
    
    qoh_uncertainty_growth = 0.02
    utz_uncertainty_growth = 0.05
    
    updated_qoh_uncertainty = drug.qoh_uncertainty + qoh_uncertainty_growth
    updated_utz_uncertainty = drug.utz_uncertainty + utz_uncertainty_growth
    
    return replace(
        drug,
        quantity_on_hand=updated_qoh,
        runway_weeks=_compute_belief_runway(updated_qoh, drug.utilization_per_week),
        qoh_uncertainty=updated_qoh_uncertainty,
        utz_uncertainty=updated_utz_uncertainty,
        weeks_since_audit=drug.weeks_since_audit + 1,
    )

def _audit_belief(drug: BeliefDrug, true_qoh: float, true_utz: float) -> BeliefDrug:
    """Audit Belief Applier"""
    return replace(
        drug,
        quantity_on_hand=true_qoh,
        utilization_per_week=true_utz,
        runway_weeks=_compute_belief_runway(true_qoh, true_utz),
        qoh_uncertainty=0.01,
        utz_uncertainty=0.01,
        weeks_since_audit=0,
    )
