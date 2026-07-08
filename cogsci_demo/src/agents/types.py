"""Agent Types"""

from src.environments.actions import Action
from typing import Optional
from dataclasses import dataclass

@dataclass
class ParamAction:
    """Parameterized Action"""
    kind: Action
    idx: Optional[int] = None
    level: Optional[int] = None