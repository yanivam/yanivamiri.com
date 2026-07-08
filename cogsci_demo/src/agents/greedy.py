"""Greedy Agent"""

from src.agents.base import BaseAgent
from src.environments.actions import Action
from src.agents.types import ParamAction

class GreedyAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="greedy")
        
    def should_audit_drug(self, drug):
        """Should Audit Drug Getter"""
        if hasattr(drug, 'qoh_uncertainty') and drug.qoh_uncertainty > 0.25:
            return True
        if hasattr(drug, 'utz_uncertainty') and drug.utz_uncertainty > 0.15:
            return True
        if hasattr(drug, 'weeks_since_audit') and drug.weeks_since_audit > 3:
            return True
        if drug.runway_weeks < 3.0 and (drug.qoh_uncertainty > 0.1 or 
                                       (hasattr(drug, 'utz_uncertainty') and drug.utz_uncertainty > 0.1)):
            return True
        return False
        
    def select_actions(self, state):
        """Actions Selector"""
        actions = {}
        for drug_name, drug in state.drugs.items():
            runway = drug.runway_weeks
            drug_actions = []
            if self.should_audit_drug(drug):
                drug_actions.append(ParamAction(Action.AUDIT_QOH_AND_UTZ))
            if runway < 2.0:
                drug_actions.append(ParamAction(Action.GREY_MARKET_PURCHASE))
            elif runway < 3.0:
                if drug.believed_primary_reliability < 0.7 and drug.active_supplier == "primary":
                    drug_actions.append(ParamAction(Action.SWITCH_SUPPLIER))
                elif drug.active_supplier == "primary":
                    drug_actions.append(ParamAction(Action.QUERY_ALTERNATE_SUPPLIER))
            elif (drug.believed_alternate_reliability - drug.believed_primary_reliability > 0.3 and 
                  drug.active_supplier == "primary"):
                drug_actions.append(ParamAction(Action.SWITCH_SUPPLIER))
            target_runway = 5.0
            soft_lma_extension = runway * 0.33
            hard_lma_extension = runway * 3.0
            if runway < target_runway:
                current_lma = drug.current_lma_type
                if current_lma == "none":
                    if runway + soft_lma_extension >= target_runway:
                        drug_actions.append(ParamAction(Action.IMPLEMENT_SOFT_LMA))
                    elif runway + hard_lma_extension >= target_runway:
                        drug_actions.append(ParamAction(Action.IMPLEMENT_HARD_LMA))
                    else:
                        drug_actions.append(ParamAction(Action.IMPLEMENT_HARD_LMA))
                elif current_lma == "soft":
                    if runway + hard_lma_extension >= target_runway:
                        drug_actions.append(ParamAction(Action.IMPLEMENT_HARD_LMA))
                elif current_lma == "hard":
                    if runway + soft_lma_extension >= target_runway:
                        drug_actions.append(ParamAction(Action.IMPLEMENT_SOFT_LMA))
            if not drug_actions:
                drug_actions = [ParamAction(Action.WAIT_MONITOR)]
            actions[drug_name] = drug_actions
        return actions