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

from img_methods import flat_arrays_to_pic, cut_to_tiles, tile_to_flat_bin_array, enhance_tiles, enh_move_ops, enh_rot_ops

file_name ="./task_02_2.png"

# Sizes
tile_size = 10
pic_size = 100

# Training size
epoch_size = 10000
epochs_num = 800

lr = 0.03

# Prepare data from image
input_pic = Image.open(file_name)
input_pic = input_pic.convert('L')

input_tiles = [tile_to_flat_bin_array(x) for x in cut_to_tiles(input_pic, tile_size, tile_size)]
input_tiles_np = np.array(input_tiles)

# Only one data enh can be true at one time
enh_move = False
enh_rot = True

# Prepare data iterator
if enh_move or enh_rot:
    enh_dta =  enhance_tiles(input_tiles_np, tile_size, tile_size, enh_move_ops if enh_move else enh_rot_ops)
    data_iterator = SampleIterator(input_tiles_np, enh_dta, epoch_size, 1)
else:
    data_iterator = SampleIterator(input_tiles_np, input_tiles_np, epoch_size, 1)

# Create NN
hidden_size = 6
net = NeuralNet([
    Linear(input_size=tile_size ** 2, output_size=hidden_size, name="lin1"),
    Sigm("sigm1"),
    
    Linear(input_size=hidden_size, output_size=hidden_size),
    Sigm("sigm1"),

    Linear(input_size=hidden_size, output_size=tile_size ** 2),
    Sigm()
])

# Train network
train(net, data_iterator, epochs_num, optimizer=SGD(lr), loss=MSEMinOf() if (enh_move or enh_rot) else MSE())

# Forward tiles in order
result_tiles_np = net.forward(input_tiles_np)
result_tiles_np = np.around(result_tiles_np)

# Unique tiles
result_tiles_np_unique = np.unique(result_tiles_np, axis=0)
input_tiles_np_unique = np.unique(input_tiles_np, axis=0)

print(np.size(result_tiles_np_unique, 0), np.size(input_tiles_np_unique, 0))

# Reconstruct the picture
def min_index(predicted, actual):
    # Returns indexes of lowest MSE loss targets for each element in a batch.
    indexes = np.zeros(np.size(predicted, 0))
    for batch_i in range(np.size(actual, 0)):
        loss_displacements = (predicted[batch_i] - actual[batch_i]) ** 2
        loss_min_index = np.argmin(np.sum(loss_displacements, axis=1), 0)
        indexes[batch_i] = loss_min_index
    return indexes

if enh_rot:
    # Use the best rotation of each tile
    rot_indexes = min_index(result_tiles_np, enhance_tiles(input_tiles_np, tile_size, tile_size, enh_rot_ops))
    for i in range(np.size(rot_indexes)):
        tile_flat = result_tiles_np[i]
        tile_2d = np.reshape(tile_flat, (tile_size, tile_size)) 
        rot_coef = (rot_indexes[i]) % 4 # I'm pretty sure -rot_index[i] should be here but that produces wrongly rotated tiles for some reason
        result_tiles_np[i] = np.rot90(tile_2d, rot_coef).flatten()

result_pic = flat_arrays_to_pic(result_tiles_np, tile_size, tile_size, pic_size, pic_size)



# Show picture & original for reference
result_pic.show()
flat_arrays_to_pic(input_tiles_np, tile_size, tile_size, pic_size, pic_size).show()
