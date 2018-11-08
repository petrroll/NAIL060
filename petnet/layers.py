import numpy as np

from petnet.tensor import Tensor
from typing import Dict, Callable

# Notes:
# - Shapes are (height, width) not the other way around
# - Layer's output is next layer input

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
        (batch, output) = (batch, input) @ (input, output) 
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
        batch_size = self.inputs.shape[0]

        # d\b = 1 * d\output | for each output
        # (output) = sum_0(batch, output) 
        # -> sum across batch for each output
        self.grads["b"] = np.sum(grad, axis=0) / batch_size

        # d\w = input * d\output  | for each individual edge
        # Note: d\SUM_i(I_i * W_i) -> for specific i -> d\I_i * W_i 
        # (input, output) = (input, batch) @ (batch, output) 
        # -> sums across batch for each edge
        self.grads["w"] = self.inputs.T @ grad / batch_size

        # d\input = weights @  d\output | sum contribs of all edges from each input
        # (batch, input) = (batch, output) @ (output, input)
        # -> maintains batch dimension
        return grad @ self.params["w"].T 
    
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

def sigm(x: Tensor) -> Tensor:
    return 1/(1+np.exp(-x))

def sigm_prime(x: Tensor) -> Tensor:
    s = sigm(x)
    return s * (1 - s)

class Tanh(Activation):
    def __init__(self):
        super().__init__(tanh, tanh_prime)

class Sigm(Activation):
    def __init__(self):
        super().__init__(sigm, sigm_prime)