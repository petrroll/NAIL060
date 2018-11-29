import numpy as np
import random

from PIL import Image

from petnet.loss import Loss, MSE
from petnet.optim import Optimizer, SGD
from petnet.data import DataIterator, BatchIterator
from typing import Callable, Tuple

from petnet.tensor import Tensor
from petnet.train import train, evaluate
from petnet.nn import NeuralNet
from petnet.layers import Linear, Tanh, Sigm
from petnet.data import BatchIterator, GenIterator, Epoch, SampleIterator

# Tranforms image to 2D binary array with custom B/W threshold
def image_to_bin_array(im, treshold=100):
    im_arr = np.array(im)
    im_arr = np.where(im_arr > treshold, 1, 0)
    
    return im_arr

# Cuts tile out of image
def cut_tile(im, x, y, size_x, size_y):
    box = (x, y, x + size_x, y + size_y)
    return im.crop(box)

# Cuts image to tiles of specified size 
# ..could be made significantly more performant through 
# ..not cutting one tile out of the image at a time
# ..but cutting the whole image at the same time.
def cut_to_tiles(im, size_x, size_y):
    tiles = []
    w, h = im.size
    for y in range(0, h, size_y):
        for x in range(0, w, size_x):
            tile = cut_tile(im, x, y, size_x, size_y)
            tiles.append(tile)

    return tiles

# Converts image to flat binary array
def tile_to_flat_array(im):
    im_arr = image_to_bin_array(im)
    return im_arr.flatten()

# Converts binary array to 8bit B/W image
def flat_array_to_tile(im_arr, w, h):
    im_arr_shaped = np.reshape(im_arr, (w, h)) * 255
    return Image.fromarray(im_arr_shaped.astype("int8"), "L")

# Merges list of tiles to one picture
def tiles_to_pic(tiles, w, h):
    new_im = Image.new('L', (w, h))
    tile_w, tile_h = tiles[0].size
    tiles_in_row = w / tile_w

    for i in range(len(tiles)):
        x = i % tiles_in_row
        y = i // tiles_in_row

        new_im.paste(tiles[i], (int(x*tile_w),int(y*tile_h)))

    return new_im

def flat_arrays_to_pic(np_arrays, tile_w, tile_h, pic_w, pic_h):
    result_tiles = [flat_array_to_tile(np_arrays[x], tile_w, tile_h) for x in range(np.size(np_arrays, 0))]
    result_pic = tiles_to_pic(result_tiles, pic_h, pic_h)

    return result_pic

# Prepare data from image
input_pic = Image.open("./task_02_1.png")
input_pic = input_pic.convert('L')

input_tiles = [tile_to_flat_array(x) for x in cut_to_tiles(input_pic, 10, 10)]
input_tiles_np = np.array(input_tiles)

data_iterator = SampleIterator(input_tiles_np, input_tiles_np, 10000, 1)

# Create NN
hidden_size = 6
net = NeuralNet([
    Linear(input_size=10*10, output_size=hidden_size, name="lin1"),
    Sigm("sigm1"),
    Linear(input_size=hidden_size, output_size=10*10),
    Sigm()
])

# Train network
train(net, data_iterator, 1000, optimizer=SGD(0.01))

# Forward tiles in order
result_tiles_np = net.forward(input_tiles_np)
result_tiles_np = np.around(result_tiles_np)

# Assemble original picture out of fotwarded tiles
result_pic = flat_arrays_to_pic(result_tiles_np, 10, 10, 100, 100)

# Unique tiles
result_tiles_np_unique = np.unique(result_tiles_np, axis=0)
input_tiles_np_unique = np.unique(input_tiles_np, axis=0)

print(np.size(result_tiles_np_unique, 0), np.size(input_tiles_np_unique, 0))

# Show picture & original for reference
result_pic.show()
flat_arrays_to_pic(input_tiles_np, 10, 10, 100, 100).show()
