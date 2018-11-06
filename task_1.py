"""
The canonical example of a function that can't be
learned with a simple linear model is XOR
"""
import numpy as np

from petnet.train import train
from petnet.nn import NeuralNet
from petnet.layers import Linear, Tanh, Sigm

inputs = np.array([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1]
])

targets = np.array([
    [1, 0],
    [0, 1],
    [0, 1],
    [1, 0]
])

net = NeuralNet([
    Linear(input_size=2, output_size=3),
    Sigm(),
    Linear(input_size=3, output_size=2),
])

train(net, inputs, targets, num_epochs=5000)

for x, y in zip(inputs, targets):
    predicted = net.forward(x)
    print(x, predicted, y)