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
from petnet.data import BatchIterator, GenIterator, Epoch

def convert_to_bin_array(im, treshold=100):
    im_arr = np.array(im)
    im_arr = np.where(im_arr > treshold, 1, 0)
    
    return im_arr

def cut_tile(im, x, y, size_x, size_y):
    box = (x, y, x + size_x, y + size_y)
    return im.crop(box)

def cut_to_tiles(im, size_x, size_y):
    tiles = []
    w, h = im.size
    for y in range(0, h, size_y):
        for x in range(0, w, size_x):
            tile = cut_tile(im, x, y, size_x, size_y)
            tiles.append(tile)

    return tiles

def tile_to_flat_array(im):
    im_arr = convert_to_bin_array(im)
    return im_arr.flatten()

def flat_array_to_tile(im_arr, w, h):
    im_arr_shaped = np.reshape(im_arr, (w, h)) * 255
    return Image.fromarray(im_arr_shaped.astype("int8"), "L")

def tiles_to_pic(tiles, w, h):
    new_im = Image.new('L', (w, h))
    tile_w, tile_h = tiles[0].size
    in_row = w / tile_w

    i = 0
    for tile in tiles:
        x = i % in_row
        y = i // in_row

        new_im.paste(tile, (int(x*tile_w),int(y*tile_h)))
        i += 1

    return new_im

im = Image.open("./task_02_1.png")
im = im.convert('L')

flat_tiles = [tile_to_flat_array(x) for x in cut_to_tiles(im, 10, 10)]
inputs = np.array(flat_tiles)

tiles = [flat_array_to_tile(inputs[x], 10, 10) for x in range(np.size(inputs,0))]
pic = tiles_to_pic(tiles, 100, 100)
pic.show()
im.show()


print(inputs.shape)

for i in range(4, 5):
    tile = tile_to_flat_array(tiles[i])
    flat_array_to_tile(tile, 10, 10)


