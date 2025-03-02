from typing import Iterable, Any
from abc import ABC, abstractmethod
import random
from regrad import Var


class Cell(ABC):
    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = 0

    @abstractmethod
    def parameters(self) -> list[Var]:
        raise NotImplementedError("Subclasses must implement the forward method.")


class Neuron(Cell):
    def __init__(self, n_in: int, is_nonlinear: bool = True) -> None:
        self.w = [Var(random.uniform(-1, 1), req_grad=True) for _ in range(n_in)]
        self.b = Var(0.0, req_grad=True)
        self.is_nonlinear = is_nonlinear

    def __call__(self, x: Iterable[Var]) -> Var:
        y = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        if self.is_nonlinear:
            return y.relu()
        return y

    def parameters(self) -> list[Var]:
        return self.w + [self.b]

    def __repr__(self) -> str:
        return f"{'ReLU' if self.is_nonlinear else 'Linear'}-Neuron({len(self.w)})"


class Layer(Cell):
    def __init__(self, n_in: int, n_out: int, **kwargs: Any) -> None:
        self.neurons = [Neuron(n_in, **kwargs) for _ in range(n_out)]

    def __call__(self, x: Iterable[Var]) -> list[Var] | Var:
        out = [n(x) for n in self.neurons]
        if len(out) == 1:
            return out[0]
        return out

    def parameters(self) -> list[Var]:
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self) -> str:
        return f"Layer-[{', '.join(str(n) for n in self.neurons)}]"


class MLP(Cell):
    def __init__(self, n_in: int, n_layers_out: list[int]):
        n_in_out = [n_in] + n_layers_out
        self.layers = [Layer(n_in_out[i], n_in_out[i + 1], is_nonlinear=i != len(n_layers_out) - 1)
                       for i in range(len(n_layers_out))]

    def __call__(self, x: Iterable[Var]) -> Var:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> list[Var]:
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self) -> str:
        return f"MLP-[{', '.join(str(layer) for layer in self.layers)}]"
