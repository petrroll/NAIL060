import numpy as np
from petnet.tensor import Tensor

class Loss:
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        raise NotImplementedError

class MSE(Loss):
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        return np.sum((predicted - actual) ** 2)

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return 2 * (predicted - actual)

class MSEMinOf(Loss):
    # Assumes shape of targets: (batch_size, possible_targets, output_size)

    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        loss = 0
        for batch_i in range(np.size(actual, 0)):
            loss_displacements = (predicted[batch_i] - actual[batch_i]) ** 2
            loss += np.min(np.sum(loss_displacements, axis=1), 0)
        
        return loss
    
    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        batch_grads = np.zeros_like(predicted)

        for batch_i in range(np.size(actual, 0)):
            loss_displacements = (predicted[batch_i] - actual[batch_i]) ** 2
            loss_min_index = np.argmin(np.sum(loss_displacements, axis=1), 0)

            batch_grads[batch_i] = (2 * (predicted[batch_i] - actual[batch_i][loss_min_index]))

        return np.array(batch_grads)






