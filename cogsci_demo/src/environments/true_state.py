"""True Environment State"""

from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class TrueSupplier:
    name: str
    type: str
    current_erd_week: int
    current_erd_pattern: List[int]

@dataclass
class TrueDrug:
    name: str
    quantity_on_hand: float 
    utilization_per_week: float 
    runway_weeks: float 
    base_utilization_per_week: float 
    min_utilization_per_week: float 
    primary_supplier: TrueSupplier 
    alternate_supplier: TrueSupplier 
    clinical_impact: float 
    shortage_count: int 
    total_weeks_tracked: int 
    current_lma_type: str 
    lma_implemented_week: int 
    active_supplier: str 
    supplier_switch_week: int 
    utz_evolution_config: Optional[dict] = None 
    utz_evolution_week: int = 0 

@dataclass
class TrueEnvState:
    week_index: int
    drugs: Dict[str, TrueDrug] 
    rng_seed: int = 123 
