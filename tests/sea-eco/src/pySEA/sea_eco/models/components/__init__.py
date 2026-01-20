"""Model component definitions for SEA-eco."""

from abc import ABC, abstractmethod
from typing import Sequence, Dict, Literal
from numpy.typing import NDArray

def ModelComponent(ABC):
    
    @abstractmethod
    def evalueate(self, coords: Sequence[NDArray] | NDArray, 
                 store_values=True, return_values=True,):
        pass