import abc
from dataclasses import dataclass

@dataclass
class ModelBaseABC(abc.ABC):
    """The base model of objects in our system."""
    id: str
