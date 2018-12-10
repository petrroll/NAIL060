import numpy as np
import random

from PIL import Image

from petnet.loss import Loss, MSE, MSEMinOf
from petnet.optim import Optimizer, SGD
from petnet.data import DataIterator, BatchIterator
from typing import Callable, Tuple

from petnet.tensor import Tensor
from petnet.train import train, evaluate
from petnet.nn import NeuralNet
from petnet.layers import Linear, Tanh, Sigm
from petnet.data import BatchIterator, GenIterator, Epoch, SampleIterator

from img_methods import *

# Sizes
tile_size = 10
pic_size = 100

# Training size
epoch_size = 10000
epochs_num = 800

lr = 0.03

# Prepare data from image
input_pic = Image.open("./task_02_1.png")
input_pic = input_pic.convert('L')

input_tiles = [tile_to_flat_array(x) for x in cut_to_tiles(input_pic, tile_size, tile_size)]
input_tiles_np = np.array(input_tiles)

displ = True
data_iterator = SampleIterator(input_tiles_np, enhance_tiles(input_tiles_np, tile_size, tile_size) if displ else input_tiles_np, epoch_size, 1)

# Create NN
hidden_size = 10
net = NeuralNet([
    Linear(input_size=tile_size ** 2, output_size=hidden_size, name="lin1"),
    Sigm("sigm1"),
    
    Linear(input_size=hidden_size, output_size=hidden_size),
    Sigm("sigm1"),

    Linear(input_size=hidden_size, output_size=tile_size ** 2),
    Sigm()
])

# Train network
train(net, data_iterator, epochs_num, optimizer=SGD(lr), loss=MSEMinOf() if displ else MSE())

# Forward tiles in order
result_tiles_np = net.forward(input_tiles_np)
result_tiles_np = np.around(result_tiles_np)

# Assemble original picture out of fotwarded tiles
result_pic = flat_arrays_to_pic(result_tiles_np, tile_size, tile_size, pic_size, pic_size)

# Unique tiles
result_tiles_np_unique = np.unique(result_tiles_np, axis=0)
input_tiles_np_unique = np.unique(input_tiles_np, axis=0)

print(np.size(result_tiles_np_unique, 0), np.size(input_tiles_np_unique, 0))

# Show picture & original for reference
result_pic.show()
flat_arrays_to_pic(input_tiles_np, tile_size, tile_size, pic_size, pic_size).show()
