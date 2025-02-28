import math
from abc import ABC, abstractmethod
from typing import Any


class Op(ABC):
    def __init__(self, op_args: Any) -> None:
        self.op_args = op_args  # graph
        self._cache: Any = None

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def save_to_cache(self, *args: Any):
        self._cache = args

    def retrieve_from_cache(self) -> tuple[Any, ...]:
        assert self._cache is not None
        values, self._cache = self._cache, None
        return values

    @abstractmethod
    def forward(self, *x: float, **kwargs: float) -> float:
        raise NotImplementedError("Subclasses must implement the forward method.")

    @abstractmethod
    def backward(self, dy: float) -> tuple[float, ...]:
        raise NotImplementedError("Subclasses must implement the backward method.")


class Add(Op):
    def forward(self, x1: float, x2: float) -> float:
        y = x1 + x2
        return y

    def backward(self, dy: float) -> tuple[float, ...]:
        dx1 = dy
        dx2 = dy
        return tuple((dx1, dx2))


class Sub(Op):
    def forward(self, x1: float, x2: float) -> float:
        y = x1 - x2
        return y

    def backward(self, dy: float) -> tuple[float, ...]:
        dx1 = dy
        dx2 = -dy
        return tuple((dx1, dx2))


class Mul(Op):
    def forward(self, x1: float, x2: float) -> float:
        y = x1 * x2
        self.save_to_cache(x1, x2)
        return y

    def backward(self, dy: float) -> tuple[float, ...]:
        x1, x2 = self.retrieve_from_cache()
        dx1 = dy * x2
        dx2 = dy * x1
        return tuple((dx1, dx2))


class Div(Op):
    def forward(self, x1: float, x2: float) -> float:
        y = x1 / x2
        self.save_to_cache(x1, x2)
        return y

    def backward(self, dy: float) -> tuple[float, ...]:
        x1, x2 = self.retrieve_from_cache()
        dx1 = dy / x2
        dx2 = -(dy * x1) / (x2 * x2)
        return tuple((dx1, dx2))


class Neg(Op):
    def forward(self, x: float) -> float:
        y = -x
        return y

    def backward(self, dy: float) -> tuple[float, ...]:
        dx = -dy
        return tuple((dx,))


class Pow(Op):
    def forward(self, x: float, power: float) -> float:
        y = x ** power
        self.save_to_cache(x, power)
        return y

    def backward(self, dy: float) -> tuple[float, ...]:
        x, power = self.retrieve_from_cache()
        dx = dy * power * x ** (power - 1)
        return tuple((dx,))


class Exp(Op):
    def forward(self, x: float) -> float:
        y = math.exp(x)
        self.save_to_cache(y)
        return y

    def backward(self, dy: float) -> tuple[float, ...]:
        (y,) = self.retrieve_from_cache()
        dx = dy * y
        return tuple((dx,))


class Log(Op):
    def forward(self, x: float) -> float:
        y = math.log(x)
        self.save_to_cache(x)
        return y

    def backward(self, dy: float) -> tuple[float, ...]:
        (x,) = self.retrieve_from_cache()
        dx = dy / x
        return tuple((dx,))


class Sqrt(Op):
    def forward(self, x: float) -> float:
        y = math.sqrt(x)
        self.save_to_cache(y)
        return y

    def backward(self, dy: float) -> tuple[float, ...]:
        (y,) = self.retrieve_from_cache()
        dx = dy * 0.5 / y
        return tuple((dx,))


class Sin(Op):
    def forward(self, x: float) -> float:
        y = math.sin(x)
        self.save_to_cache(x)
        return y

    def backward(self, dy: float) -> tuple[float, ...]:
        (x,) = self.retrieve_from_cache()
        dx = dy * math.cos(x)
        return tuple((dx,))


class Cos(Op):
    def forward(self, x: float) -> float:
        y = math.cos(x)
        self.save_to_cache(x)
        return y

    def backward(self, dy: float) -> tuple[float, ...]:
        (x,) = self.retrieve_from_cache()
        dx = -dy * math.sin(x)
        return tuple((dx,))


class Tanh(Op):
    def forward(self, x: float) -> float:
        y = math.tanh(x)
        self.save_to_cache(y)
        return y

    def backward(self, dy: float) -> tuple[float, ...]:
        (y,) = self.retrieve_from_cache()
        dx = dy * (1 - y * y)
        return tuple((dx,))


class Relu(Op):
    def forward(self, x: float) -> float:
        y = x if x >= 0.0 else 0.0
        self.save_to_cache(y == x)
        return y

    def backward(self, dy: float) -> tuple[float, ...]:
        (mask,) = self.retrieve_from_cache()
        dx = dy * mask
        return tuple((dx,))
