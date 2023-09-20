import numpy as np
from tensor import MyTensor
import random
from typing import List


class Neuron:
    def __init__(self, nin):
        self.w = [MyTensor(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = MyTensor(random.uniform(-1, 1))

    def __call__(self, x):
        z = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = z.elu()

        return out

    def parameters(self):
        return self.w + [self.b]

    def __str__(self) -> str:
        wb = []
        wb.append(f"w => {self.w} \n")
        wb.append(f"b => {self.b} \n\n")

        return "".join(wb)


class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

    def __str__(self):
        nues = []
        for i, n in enumerate(self.neurons):
            nues.append(f"N {i+1}, {n}")
        return "".join(nues)


class MLP:
    def __init__(self, nin: int, nouts: List[int]):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)

        return x

    def parameters(self):
        ps = [p for layer in self.layers for p in layer.parameters()]
        print(ps)
        return ps


if __name__ == "__main__":
    from pprint import pprint

    xs = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    demonet = MLP(3, [3, 2])

    res = [demonet(x) for x in xs]

    print(np.array(demonet.parameters()).shape)
