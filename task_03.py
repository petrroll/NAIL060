import numpy as np
import random

from PIL import Image
from task_03_dta import dta_mapping as dta 

from petnet.loss import Loss, MSE, MSEMinOf
from petnet.optim import Optimizer, SGD, SGDDecay
from petnet.data import DataIterator, BatchIterator, SampleMultInputsIterator
from typing import Callable, Tuple

from petnet.tensor import Tensor
from petnet.train import train, evaluate
from petnet.nn import NeuralNet
from petnet.layers import Linear, Tanh, Sigm
from petnet.data import BatchIterator, GenIterator, Epoch, SampleIterator

from img_methods import load_img_to_flat_bin_arr, load_img_cut_to_flat_bin_arrs, flat_arrays_to_pic, enh_move_more_ops, enhance_tiles

def get_input_targets(file_to_char_i, input_size, max_char_i):
    '''
    Takes a file_to_char_i dict and returns (inputs, targets)
    Inputs: 2D numpy array (num_of_files, flatten_image_as_bin_array)
    Targets: 2D numpy array (num_of_files, one-hot_encoded_char_i) 
    '''
    inputs_len = len(file_to_char_i)

    inputs = np.zeros((inputs_len, input_size))
    targets = np.zeros((inputs_len, max_char_i))
    char_is = np.zeros(inputs_len)

    input_i = 0
    for path, char_i in file_to_char_i.items():      
        inputs[input_i] = load_img_to_flat_bin_arr(path)
        targets[input_i][char_i] = 1
        char_is[input_i] = char_i

        input_i += 1

    return (inputs, targets, char_is)
    
test_file_name = "./task_03_1.png"
test_file_size = 320


# Sizes
tile_size = 10
pic_size = tile_size ** 2
tiles_per_line = test_file_size // tile_size

# Training size
epoch_size = 1000
epochs_num = 500
batch_size = 1

lr = 0.03
dec_per_epoch = 0.988

# Load data & prapare data
char_i_to_char, char_i_to_files, file_to_char_i = dta.get_data()
inputs, targets, char_is = get_input_targets(file_to_char_i, pic_size, len(char_i_to_char))

enhanced_inputs = enhance_tiles(inputs, tile_size, tile_size, enh_move_more_ops)
data_iterator = SampleMultInputsIterator(enhanced_inputs, targets, epoch_size, batch_size)

# Create NN
hidden_size = 20
net = NeuralNet([
    Linear(input_size=tile_size ** 2, output_size=hidden_size, name="lin1"),
    Tanh(),
    
    #Linear(input_size=hidden_size, output_size=hidden_size),
    #Tanh(),

    Linear(input_size=hidden_size, output_size=len(char_i_to_char)),
    Sigm()
])


# Train network
train(net, data_iterator, epochs_num, optimizer=SGDDecay(lr, dec_per_epoch, epoch_size // batch_size), loss=MSE())

# load testing picture into flat tiles
test_tiles_np = load_img_cut_to_flat_bin_arrs(test_file_name, tile_size, tile_size)
flat_arrays_to_pic(test_tiles_np, tile_size, tile_size, tile_size * tiles_per_line, tile_size * tiles_per_line).show()

# Test original tiles
result_orig_tiles = net.forward(inputs)
result_orig_tiles_chars_i = np.argmax(result_orig_tiles, axis=1)
print(np.sum(result_orig_tiles_chars_i == char_is) / len(result_orig_tiles_chars_i))


result_chars_i_distr = net.forward(test_tiles_np)
result_chars_i = np.argmax(result_chars_i_distr, axis=1)

for i in range(len(result_chars_i)):
    char_i = result_chars_i[i]
    char = char_i_to_char[char_i]

    print(char, end="")
    if i % tiles_per_line == tiles_per_line - 1: print("")

