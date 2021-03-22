from abc import ABC, abstractmethod


class Formula(ABC):
    """Base class for formulas

    You shouldn't use this class directly.
    Use derived classes instead.
    """

    @abstractmethod
    def __call__(self, *args) -> float:
        pass


