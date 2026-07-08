"""Agent Base Class"""

from abc import abstractmethod, ABC
class BaseAgent(ABC):
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def select_actions(self, state):
        """Actions Selector"""
        pass
    