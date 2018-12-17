from petnet.tensor import Tensor
from petnet.nn import NeuralNet
from petnet.loss import Loss, MSE
from petnet.optim import Optimizer, SGD
from petnet.data import DataIterator, BatchIterator
from typing import Callable, Tuple

def train(
    net: NeuralNet,
    iterator: DataIterator,
    num_epochs: int = 5000,
    loss: Loss = MSE(),
    optimizer: Optimizer = SGD()
) -> None:
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        batch_i = 0
        for batch in iterator():
            predicted = net.forward(batch.inputs)
            epoch_loss += loss.loss(predicted, batch.targets)
            grad = loss.grad(predicted, batch.targets)
            net.backward(grad)
            optimizer.step(net)
            batch_i += 1
        print(epoch, ":", epoch_loss / batch_i)

FCorrect = Callable[[Tensor, Tensor], bool]
def evaluate(
    net: NeuralNet,
    inputs: Tensor,
    targets: Tensor,
    is_correct: FCorrect
) -> None:
    correct = 0
    for i in range(len(inputs)):
        output = net.forward_single(inputs[i])
        gold = targets[i]
        if is_correct(output, gold): correct += 1 

    print(correct / len(inputs))
