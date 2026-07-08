"""Heuristic Agent"""
from src.agents.base import BaseAgent
from src.environments.actions import Action
from src.agents.types import ParamAction
from src.environments.belief_updates import (
    get_action_frequency, is_repeating_actions, 
    get_action_effectiveness,
    has_shipment_arrived, get_shipment_amount, was_shipment_successful,
    get_shipment_runway_improvement
)
import math

class HeuristicAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="heuristic")
    
    def has_recent_shipment(self, drug):
        """Has Recent Shipment Getter"""
        return has_shipment_arrived(drug) and was_shipment_successful(drug)
    
    def should_reduce_urgency_after_shipment(self, drug):
        """Should Reduce Urgency After Shipment Getter"""
        if not self.has_recent_shipment(drug):
            return False
        improvement = get_shipment_runway_improvement(drug)
        return improvement >= 2.0
    
    def should_remove_lma_after_shipment(self, drug):
        """Should Remove LMA After Shipment Getter"""
        if not self.has_recent_shipment(drug) or drug.current_lma_type == "none":
            return False
        improvement = get_shipment_runway_improvement(drug)
        return (improvement >= 3.0 and drug.runway_weeks >= 6.0)
    
    def should_switch_supplier(self, drug):
        """Should Switch Supplier Getter"""
        if drug.active_supplier == "primary":
            return (drug.runway_weeks < 4.0 and 
                    (drug.believed_primary_reliability < 0.8 or drug.erd_weeks > 4.0)) or \
                   (drug.believed_alternate_reliability - drug.believed_primary_reliability > 0.3)
        else:
            return (drug.runway_weeks < 4.0 and 
                    (drug.believed_alternate_reliability < 0.8 or drug.alternate_supplier_erd_weeks > 4.0)) or \
                   (drug.believed_primary_reliability - drug.believed_alternate_reliability > 0.3)
    
    def should_switch_back_to_primary(self, drug):
        """Should Switch Back to Primary Getter"""
        if drug.active_supplier != "alternate":
            return False
        weeks_using_alternate = drug.supplier_switch_week if hasattr(drug, 'supplier_switch_week') else 0
        return (drug.believed_alternate_reliability < 0.6 or 
                weeks_using_alternate > 4 or
                drug.alternate_supplier_erd_weeks > 8.0)
    
    def is_delivery_expected_soon(self, drug):
        """Is Delivery Expected Soon Getter"""
        return drug.erd_weeks <= 2.0 and drug.erd_uncertainty < 1.0
    
    def get_action_effectiveness_score(self, drug, action_name):
        """Get Action Effectiveness Score Getter"""
        effectiveness = get_action_effectiveness(drug, action_name)
        return effectiveness["effectiveness"]
    
    def get_action_frequency_count(self, drug, action_name):
        """Get Action Frequency Count Getter"""
        return get_action_frequency(drug, action_name)
    
    def is_repeating_actions_check(self, drug):
        """Is Repeating Actions Check Getter"""
        return is_repeating_actions(drug, threshold=3)
    
    
    def get_shipment_amount_info(self, drug):
        """Get Shipment Amount Info Getter"""
        return get_shipment_amount(drug)
        
        
    def should_audit_drug(self, drug, confidence_threshold=0.7):
        """Should Audit Drug Getter"""
        if hasattr(drug, 'qoh_uncertainty') and drug.qoh_uncertainty > 0.15:
            return True
        if hasattr(drug, 'utz_uncertainty') and drug.utz_uncertainty > 0.15:
            return True
        if drug.runway_weeks < 3.0 and (drug.qoh_uncertainty > 0.1 or
                                       (hasattr(drug, 'utz_uncertainty') and drug.utz_uncertainty > 0.1) or
                                       drug.erd_uncertainty > 1.5):
            return True
        if (hasattr(drug, 'weeks_since_audit') and drug.weeks_since_audit > 4 and 
            (drug.qoh_uncertainty > 0.1 or (hasattr(drug, 'utz_uncertainty') and drug.utz_uncertainty > 0.1))):
            return True
        if (1.0 <= drug.runway_weeks <= 3.0 and 
            abs(drug.believed_primary_reliability - drug.believed_alternate_reliability) > 0.4):
            return True
        if drug.runway_weeks > 6.0 and (drug.qoh_uncertainty > 0.2 or 
                                       (hasattr(drug, 'utz_uncertainty') and drug.utz_uncertainty > 0.2)):
            return True
        return False
        
    def calculate_action_utility(self, drug, action_name, runway):
        """Calculate Action Utility Getter"""
        effectiveness = self.get_action_effectiveness_score(drug, action_name)
        frequency = self.get_action_frequency_count(drug, action_name)
        cost_penalty = 0.0
        if action_name in ["GREY_MARKET_PURCHASE", "REQUEST_FROM_RESERVE_WAREHOUSE"]:
            cost_penalty = 0.1
        elif action_name in ["REQUEST_LOAN_FROM_OTHER_HOSPITALS", "CONTACT_MANUFACTURER_DIRECT"]:
            cost_penalty = 0.05
        elif action_name in ["IMPLEMENT_HARD_LMA", "IMPLEMENT_SOFT_LMA"]:
            cost_penalty = 0.02
        frequency_penalty = min(0.2, frequency * 0.05)
        urgency_bonus = max(0, (10.0 - runway) / 10.0) * 0.4
        if action_name == "GREY_MARKET_PURCHASE" and runway < 1.0:
            urgency_bonus += 0.3
        strategic_bonus = 0.0
        if action_name == "AUDIT_QOH_AND_UTZ":
            if drug.weeks_since_audit > 6:
                strategic_bonus = 0.15
            else:
                strategic_bonus = 0.05
        elif action_name in ["IMPLEMENT_SOFT_LMA", "IMPLEMENT_HARD_LMA"]:
            if runway < 3.0:
                strategic_bonus = 0.1
            else:
                strategic_bonus = 0.02
        elif action_name == "QUERY_ALTERNATE_SUPPLIER":
            if drug.erd_uncertainty > 2.0:
                strategic_bonus = 0.08
            else:
                strategic_bonus = 0.02
        utility = effectiveness - cost_penalty - frequency_penalty + urgency_bonus + strategic_bonus
        return max(0.0, utility)
    
    def select_best_action(self, drug, candidate_actions, runway):
        """Select Best Action Getter"""
        if not candidate_actions:
            return ParamAction(Action.AUDIT_QOH_AND_UTZ)
        action_utilities = {}
        for action_name in candidate_actions:
            utility = self.calculate_action_utility(drug, action_name, runway)
            action_utilities[action_name] = utility
        best_action = max(action_utilities, key=action_utilities.get)
        return ParamAction(Action[best_action])
    
    def select_actions(self, state):
        """Actions Selector"""
        actions = {}
        for drug_name, drug in state.drugs.items():
            runway = drug.runway_weeks
            drug_actions = []
            urgency_reduction = self.should_reduce_urgency_after_shipment(drug)
            runway_urgency = self.u_runway(runway)
            staleness_urgency = self.u_staleness(drug.weeks_since_audit)
            uncertainty_urgency = self.u_qoh(drug.qoh_uncertainty, 100.0)
            supplier_risk = self.u_supplier_risk(drug.believed_primary_reliability)
            total_urgency = runway_urgency + 0.3 * staleness_urgency + 0.3 * uncertainty_urgency + 0.3 * supplier_risk
            if urgency_reduction:
                shipment_amount = self.get_shipment_amount_info(drug)
                if shipment_amount > drug.utilization_per_week * 4:
                    total_urgency *= 0.3
                else:
                    total_urgency *= 0.5
            if self.should_audit_drug(drug):
                drug_actions.append(ParamAction(Action.AUDIT_QOH_AND_UTZ))
            if self.is_repeating_actions_check(drug):
                pass
            if not self.is_delivery_expected_soon(drug) and not urgency_reduction:
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
            if self.should_remove_lma_after_shipment(drug):
                drug_actions.append(ParamAction(Action.REMOVE_LMA))
            if runway < 1.0 and not urgency_reduction:
                emergency_actions = ["GREY_MARKET_PURCHASE", "REQUEST_FROM_RESERVE_WAREHOUSE", "REQUEST_LOAN_FROM_OTHER_HOSPITALS"]
                best_action = self.select_best_action(drug, emergency_actions, runway)
                drug_actions.append(best_action)
            elif runway < 2.0 and not self.is_delivery_expected_soon(drug) and not urgency_reduction:
                drug_actions.append(ParamAction(Action.GREY_MARKET_PURCHASE))
            elif self.should_switch_back_to_primary(drug):
                drug_actions.append(ParamAction(Action.SWITCH_BACK_TO_PRIMARY))
            elif self.should_switch_supplier(drug):
                drug_actions.append(ParamAction(Action.SWITCH_SUPPLIER))
            elif runway < 3.0 and (drug.erd_uncertainty > 2.0 or drug.erd_weeks > 5.0):
                drug_actions.append(ParamAction(Action.QUERY_ALTERNATE_SUPPLIER))
            elif self.is_delivery_expected_soon(drug):
                drug_actions.append(ParamAction(Action.AUDIT_QOH_AND_UTZ))
            if not drug_actions:
                if total_urgency > 0.3:
                    drug_actions = [ParamAction(Action.AUDIT_QOH_AND_UTZ)]
                else:
                    drug_actions = [ParamAction(Action.WAIT_MONITOR)]
            if len(drug_actions) > 1:
                action_utilities = {}
                for action in drug_actions:
                    action_name = action.kind.name
                    utility = self.calculate_action_utility(drug, action_name, runway)
                    action_utilities[action_name] = (action, utility)
                best_action_name = max(action_utilities, key=lambda k: action_utilities[k][1])
                best_action = action_utilities[best_action_name][0]
                drug_actions = [best_action]
            actions[drug_name] = drug_actions
        return actions

    def clamp01(self, x: float) -> float:
        """Clamp 01 Helper"""
        return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

    def u_runway(self, believed_runway: float) -> float:
        """Runway Urgency Getter"""
        return self.clamp01(math.exp(-believed_runway / 2.0))

    def u_qoh(self, believed_qoh: float, avg_believed_qoh: float) -> float:
        """QOH Urgency Getter"""
        denom = 1.0 + (believed_qoh / max(avg_believed_qoh, 1e-6))
        return self.clamp01(1.0 / denom)

    def u_supplier_risk(self, reliability: float) -> float:
        """Supplier Risk Urgency Getter"""
        return self.clamp01(1.0 - reliability)

    def u_staleness(self, weeks_since_audit: int) -> float:
        """Staleness Urgency Getter"""
        return self.clamp01(weeks_since_audit / 8.0)

