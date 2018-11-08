from typing import Sequence, Iterator, Tuple
import numpy as np

from petnet.tensor import Tensor
from petnet.layers import Layer

class NeuralNet:
    def __init__(self, layers: Sequence[Layer]) -> None:
        self.layers = layers

    def forward(self, inputs: Tensor) -> Tensor:
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def forward_single(self, input: Tensor) -> Tensor:
        inputs = np.expand_dims(input, axis=0)
        return self.forward(inputs)[0]

    def backward(self, grad: Tensor) -> Tensor:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def params_and_grads(self) -> Iterator[Tuple[Tensor, Tensor]]:
            for layer in self.layers:
                for name, param in layer.params.items():
                    grad = layer.grads[name]
                    yield param, grad 