from typing import Callable, Any
from abc import ABC, abstractmethod

from asuka.partition.kernel import Kernel


class Builder(ABC):

  def __init__(self, kernel: Kernel) -> None:
    self.kernel = kernel

  @abstractmethod
  def build_callable(self) -> Callable[..., Any]:
    pass

  # @abstractmethod
  # def serialize(self) -> str:
  #   pass

  # @abstractmethod
  # def deserialize(self, buf: str) -> None:
  #   pass
