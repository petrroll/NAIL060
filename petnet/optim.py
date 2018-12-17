from petnet.nn import NeuralNet

class Optimizer:
    def step(self, net: NeuralNet) -> None:
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr
    
    def step(self, net: NeuralNet) -> None:
        for param, grad in net.params_and_grads():
            param -= self.lr * grad

class SGDDecay(Optimizer):
    def __init__(self, lr: float = 0.01, decay: float = 0.999, dec_each: int = 10000) -> None:
        self.lr = lr
        self.decay = decay
        self.dec_each = dec_each

        self.iter = 0
        print(self.dec_each, self.iter)
    
    def step(self, net: NeuralNet) -> None:
        self.iter += 1
        if self.iter % self.dec_each == 0:
            self.lr *= self.decay

        for param, grad in net.params_and_grads():
            param -= self.lr * grad