from __future__ import annotations

from typing import Optional
from .ops import *


class Var:
    def __init__(
            self,
            val: float,
            op: Optional[Op] = None,
            src: Optional[tuple[Var, ...]] = None,
            req_grad: bool = False
    ) -> None:
        self.val = val
        self.op = op
        self.src = src
        self.req_grad = req_grad

        self.grad: Optional[float] = None

    @property
    def name(self) -> str:
        if self.op is not None:
            return self.op.name
        return self.__class__.__name__

    def __repr__(self) -> str:
        return f"{self.val}"

    def __add__(self, v: float | Var) -> Var:
        return _apply(Add, self, _align(v))

    __radd__ = __add__

    def __sub__(self, v: float | Var) -> Var:
        return _apply(Sub, self, _align(v))

    def __rsub__(self, v: float | Var) -> Var:
        return _apply(Sub, _align(v), self)

    def __mul__(self, v: float | Var) -> Var:
        return _apply(Mul, self, _align(v))

    __rmul__ = __mul__

    def __truediv__(self, v: float | Var) -> Var:
        return _apply(Div, self, _align(v))

    def __rtruediv__(self, v: float | Var) -> Var:
        return _apply(Div, _align(v), self)

    def __neg__(self) -> Var:
        return _apply(Neg, self)

    def __pow__(self, power) -> Var:
        return _apply(Pow, self, power=power)

    def exp(self) -> Var:
        return _apply(Exp, self)

    def log(self) -> Var:
        return _apply(Log, self)

    def sqrt(self) -> Var:
        return _apply(Sqrt, self)

    def sin(self) -> Var:
        return _apply(Sin, self)

    def cos(self) -> Var:
        return _apply(Cos, self)

    def tanh(self) -> Var:
        return _apply(Tanh, self)

    def relu(self) -> Var:
        return _apply(Relu, self)

    def accumulate_grad(self, dy: float) -> None:
        self.grad = dy if self.grad is None else self.grad + dy

    def backward(self, dy: Optional[float] = None):
        assert self.req_grad, "Node is not part of a autograd graph."
        assert self.grad is None, "Cannot run backward multiple times."

        if dy is None:
            self.grad = 1.0
        else:
            self.grad = dy

        node_queue = _computed_graph_dfs(self, [], set())
        for node in reversed(node_queue):
            grads = node.op.backward(node.grad)
            for x, dy in zip(node.src, grads):
                if x.req_grad:
                    x.accumulate_grad(dy)
            # clear context of non-leaf node
            node.grad, node.op, node.src = None, None, None


def _align(v: float | Var) -> Var:
    if isinstance(v, Var):
        return v
    return Var(v)


def _apply(op_: type(Op), *var_args: Var, **kwargs: Any) -> Var:
    op = op_(kwargs)
    fwd_args = [t.val for t in var_args]
    val = op.forward(*fwd_args, **kwargs)
    result_req_grad = any(t.req_grad for t in var_args)
    if result_req_grad:
        return Var(val, op=op, src=var_args, req_grad=True)

    return Var(val)


def _computed_graph_dfs(node: Var, queue: list[Var], visited: set) -> list[Var]:
    if node not in visited:
        visited.add(node)
        if node.src is None:
            return []
        for p in node.src:
            if p.req_grad:
                _ = _computed_graph_dfs(p, queue, visited)
        queue.append(node)
    return queue
