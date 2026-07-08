"""Random Agent"""

import random
from src.agents.base import BaseAgent
from src.environments.actions import all_actions
from src.agents.types import ParamAction

class RandomAgent(BaseAgent):
    
    def __init__(self, random_seed: int = 123):
        super().__init__(name="random")
        random.seed(random_seed)

    def random_actions_for_drug(self):
        """Actions For Drug Getter"""
        actions = []
        
        num_actions = random.randint(0, 4)
        
        if num_actions > 0:
            available_actions = list(all_actions())
            selected_actions = random.sample(available_actions, min(num_actions, len(available_actions)))
            
            for action in selected_actions:
                actions.append(ParamAction(action))
        
        return actions
    
    def select_actions(self, state):
        """Actions Selector"""
        actions = {}
        for drug_name, drug in state.drugs.items():
            actions[drug_name] = self.random_actions_for_drug()
        return actions
        
