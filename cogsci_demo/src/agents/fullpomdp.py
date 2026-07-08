"""Online POMDP Lookahead Agent"""

from src.agents.base import BaseAgent
from src.environments.rewards import LAMBDA_ACTION_COUNT
from src.planner.pomdp import _plan, PlanningConfig

class OnlinePOMDPLookaheadAgent(BaseAgent):
    
    def __init__(self, horizon: int = 15, gamma: float = 0.95, 
                 beam_width: int = 15, obs_sigma: float = 0.1, 
                 n_obs_samples: int = 16):
        super().__init__(name="full_pomdp")
        self.horizon = horizon
        self.gamma = gamma
        self.beam_width = beam_width
        self.obs_sigma = obs_sigma
        self.n_obs_samples = n_obs_samples
        self.lambda_action_count = LAMBDA_ACTION_COUNT
        self.config = PlanningConfig(
            horizon=self.horizon,
            gamma=self.gamma,
            beam_width=self.beam_width,
            obs_sigma=self.obs_sigma,
            n_obs_samples=self.n_obs_samples,
            lambda_action_count=self.lambda_action_count
        )
        
    def select_actions(self, state):
        """Actions Selector"""
        actions = _plan(state, self.config)
        return actions
