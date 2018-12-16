from typing import Iterator, NamedTuple

import numpy as np
import random as rnd

from petnet.tensor import Tensor
from typing import Callable, Tuple

Batch = NamedTuple("Batch", [("inputs", Tensor), ("targets", Tensor)])

class DataIterator:
    def __call__(self) -> Iterator[Batch]:
        raise NotImplementedError

    def iterate(self, inputs: Tensor, targets: Tensor, starts: Tensor, batch_size: int) -> Iterator[Batch]:
        for start in starts:
            end = start + batch_size

            batch_inputs = inputs[start:end]
            batch_targets = targets[start:end]

            yield Batch(batch_inputs, batch_targets)



class BatchIterator(DataIterator):
    def __init__(self, inputs: Tensor, targets: Tensor, batch_size: int = 2, shuffle: bool = True) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.inputs = inputs
        self.targets = targets

    def __call__(self) -> Iterator[Batch]:
        starts = np.arange(0, len(self.inputs), self.batch_size)
        if self.shuffle:
            np.random.shuffle(starts)

        return self.iterate(self.inputs, self.targets, starts, self.batch_size)

Epoch = NamedTuple("Batch", [("inputs", Tensor), ("targets", Tensor)])
FGen = Callable[[], Epoch]
class GenIterator(DataIterator):
    def __init__(self, generator: FGen, batch_size: int = 2):
        self.generator = generator
        self.batch_size = batch_size


    def __call__(self) -> Iterator[Batch]:
        inputs, targets = self.generator()
        starts = np.arange(0, len(inputs), self.batch_size)

        return self.iterate(inputs, targets, starts, self.batch_size)


class SampleIterator(DataIterator):
    def __init__(self, inputs: Tensor, targets: Tensor, epoch_size: int = 1000, batch_size: int = 2) -> None:
        self.batch_size = batch_size
        self.epoch_size = epoch_size

        self.inputs = inputs
        self.targets = targets

    def __call__(self) -> Iterator[Batch]:
        epoch_inputs = []
        epoch_targets = []

        for _ in range(self.epoch_size):
            index = rnd.randrange(0, len(self.inputs))
            epoch_inputs.append(self.inputs[index])
            epoch_targets.append(self.targets[index])

        starts = np.arange(0, len(self.inputs), self.batch_size)
        return self.iterate(np.array(epoch_inputs), np.array(epoch_targets), starts, self.batch_size)

class SampleMultInputsIterator(SampleIterator):
    def __call__(self) -> Iterator[Batch]:
        epoch_inputs = []
        epoch_targets = []

        for _ in range(self.epoch_size):
            index = rnd.randrange(0, np.size(self.inputs, 0))
            index_i = rnd.randrange(0, np.size(self.inputs[index], 0))

            epoch_inputs.append(self.inputs[index][index_i])
            epoch_targets.append(self.targets[index])

        starts = np.arange(0, len(self.inputs), self.batch_size)
        return self.iterate(np.array(epoch_inputs), np.array(epoch_targets), starts, self.batch_size)
