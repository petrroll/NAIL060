import numpy as np
import random

from PIL import Image
from task_03_dta import dta_mapping as dta 

from petnet.loss import Loss, MSE, MSEMinOf
from petnet.optim import Optimizer, SGD
from petnet.data import DataIterator, BatchIterator
from typing import Callable, Tuple

from petnet.tensor import Tensor
from petnet.train import train, evaluate
from petnet.nn import NeuralNet
from petnet.layers import Linear, Tanh, Sigm
from petnet.data import BatchIterator, GenIterator, Epoch, SampleIterator

from img_methods import load_img_to_flat_bin_arr

def get_input_targets(file_to_char_i, input_size, max_char_i):
    '''
    Takes a file_to_char_i dict and returns (inputs, targets)
    Inputs: 2D numpy array (num_of_files, flatten_image_as_bin_array)
    Targets: 2D numpy array (num_of_files, one-hot_encoded_char_i) 
    '''
    inputs_len = len(file_to_char_i)

    inputs = np.zeros((inputs_len, input_size))
    targets = np.zeros((inputs_len, max_char_i))

    input_i = 0
    for path, char_i in file_to_char_i.items():      
        inputs[input_i] = load_img_to_flat_bin_arr(path)
        targets[input_i][char_i] = 1

        input_i += 1

    return (inputs, targets)
    

# Sizes
tile_size = 16
pic_size = 16 ** 2

# Load data
char_i_to_char, char_i_to_files, file_to_char_i = dta.get_data()
inputs, targets = get_input_targets(file_to_char_i, pic_size, len(char_i_to_char))

print(inputs, targets)

