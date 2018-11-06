import numpy as np

from petnet.tensor import Tensor
from typing import Dict, Callable

class Layer:
    def __init__(self) -> None:
        self.params: Dict[str, Tensor] = {}
        self.grads: Dict[str, Tensor] = {}
    
    def forward(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError

    def backward(self, grad: Tensor) -> Tensor:
        raise NotImplementedError
    
class Linear(Layer):
    def __init__(self, input_size: int, output_size: int) -> None:
        # inputs will be (batch_size, input_size)
        # outputs will be (batch_size, output_size)
        super().__init__()
        self.params["w"] = np.random.randn(input_size, output_size)
        self.params["b"] = np.random.randn(output_size)

    def forward(self, inputs: Tensor):
        """
        outputs = inputs @ w + b
        """
        self.inputs = inputs
        return inputs @ self.params["w"] + self.params["b"]

    def backward(self, grad: Tensor) -> Tensor:
        """
        f(g(x))/dx = f'(g(x))*g'(x) # g(x) is output
        y = f(x) and x = a @ b + c  # g'(x) is derivation w.r.t next layer
        | dy/da = f'(x) @ b.T
        | dy/db = a.T @ f'(x)
        | dy/dc = f'(x)
        """
        self.grads["b"] = np.sum(grad, axis=0) # w.r.t b = 1 * w.r.t input of next layer
        self.grads["w"] = self.inputs.T @ grad # w.r.t w = input *  w.r.t input of next layer
        return grad @ self.params["w"].T # w.r.t input = w @  w.r.t input of next layer
    
F = Callable[[Tensor], Tensor]
class Activation(Layer):
    def __init__(self, f: F, f_prime: F) -> None:
        super().__init__()
        self.f = f
        self.f_prime = f_prime

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return self.f(inputs)

    def backward(self, grad: Tensor) -> Tensor:
        return self.f_prime(self.inputs) * grad # w.r.t input = f'(input) @ w.r.t input of next layer


def tanh(x: Tensor) -> Tensor:
    return np.tanh(x)

def tanh_prime(x: Tensor) -> Tensor:
    y = tanh(x) 
    return 1 - y ** 2


class Tanh(Activation):
    def __init__(self):
        super().__init__(tanh, tanh_prime)