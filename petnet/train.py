from petnet.tensor import Tensor
from petnet.nn import NeuralNet
from petnet.loss import Loss, MSE
from petnet.optim import Optimizer, SGD
from petnet.data import DataIterator, BatchIterator

def train(
    net: NeuralNet,
    iterator: DataIterator,
    num_epochs: int = 5000,
    loss: Loss = MSE(),
    optimizer: Optimizer = SGD()
) -> None:
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in iterator():
            predicted = net.forward(batch.inputs)
            epoch_loss += loss.loss(predicted, batch.targets)
            grad = loss.grad(predicted, batch.targets)
            net.backward(grad)
            optimizer.step(net)
        print(epoch, ":", epoch_loss)