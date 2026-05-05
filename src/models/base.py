from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def generate(self, messages, temperature: float, max_tokens: int, **kwargs) -> str:
        pass