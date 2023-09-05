from typing import Union
import numpy as np

Arrayable = Union[float, list, np.ndarray]


def ensure_array(arrayable: Arrayable) -> np.ndarray:
    if isinstance(arrayable, np.ndarray):
        return arrayable
    else:
        return np.array(arrayable)


class MyTensor:
    def __init__(
        self, data: Arrayable, _children=(), _op="", label="", requires_grad=False
    ) -> None:
        self.data = ensure_array(data)
        self.requires_grad = requires_grad
        self._prev = set(_children)
        self._op: str = _op
        self._backward = lambda: None

        self.grad: np.ndarray = np.zeros_like(self.data)
        self.label = label

    def __repr__(self) -> str:
        return f"\nT({self.data}, rg={self.requires_grad}[0])"

    def __add__(self, other) -> "MyTensor":
        other = other if isinstance(other, MyTensor) else MyTensor(other)
        out = MyTensor(data=self.data + other.data, _children=(self, other), _op="+")

        def _bkwd():
            self.grad = self.grad + (out.grad * 1.0)
            other.grad = other.grad + (out.grad * 1.0)

        out._backward = _bkwd
        return out

    def __mul__(self, other):
        other = other if isinstance(other, MyTensor) else MyTensor(other)

        out = MyTensor(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad = self.grad + (other.data * out.grad)
            other.grad = other.grad + (self.data * out.grad)

        out._backward = _backward
        return out

    def __pow__(self, other):
        if isinstance(other, (float, int)):
            res = np.power(self.data, other)
            out = MyTensor(data=res, _children=(self,), _op=f"**{other}")

            def _bkwd():
                self.grad = self.grad + (
                    (other * np.power(self.data, other - 1)) * out.grad
                )

            out._backward = _bkwd
            return out

        elif isinstance(other, MyTensor):
            out_data = np.power(self.data, other.data)
            out = MyTensor(out_data, (self, other), f"**")

            def _bkwd():
                self.grad = self.grad + (
                    (other.data * np.power(self.data * other.data - 1)) * out.grad
                )
                other.grad = other.grad + (np.log(self.data) * out.grad)

            out._backward = _bkwd
            return out

        else:
            raise TypeError(
                "Unsupported operand type(s) for **: 'Value' and '{}'".format(
                    type(other).__name__
                )
            )

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def exp(self):
        x = self.data
        out = MyTensor(np.exp(x), (self,), "exp")

        def _backward():
            self.grad = self.grad + (out.data * out.grad)

        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        t = np.tanh(x)
        out = MyTensor(t, (self,), "tanh")

        def _backward():
            self.grad = self.grad + ((1 - t**2) * out.grad)

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = np.ones_like(self.data)

        for node in reversed(topo):
            node._backward()
