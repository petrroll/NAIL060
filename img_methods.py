import numpy as np
import random

from PIL import Image

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

def move_right(pic_2d):
    result = np.zeros_like(pic_2d)
    result[:, 1:] = pic_2d[:, :-1]
    result[:, 0] = pic_2d[:, 0] 

    return result

def move_left(pic_2d):
    result = np.zeros_like(pic_2d)
    result[:, :-1] = pic_2d[:, 1:]
    result[:, -1] = pic_2d[:, -1] 

    return result

def move_down(pic_2d):
    result = np.zeros_like(pic_2d)
    result[1:,:] = pic_2d[:-1,:]
    result[0,:] = pic_2d[0,:] 

    return result

def move_up(pic_2d):
    result = np.zeros_like(pic_2d)
    result[:-1,:] = pic_2d[1:,:]
    result[-1,:] = pic_2d[-1,:] 

    return result

def pic_np_to_pic_np2d(pic_np, w, h):
    return 

def enhance_tiles(input, w, h):
    enhance_ops = [move_left, move_right, move_down, move_up,
                    lambda x: move_left(move_down(x)), lambda x: move_left(move_up(x)),
                    lambda x: move_right(move_down(x)), lambda x: move_right(move_up(x))]

    result = []
    for i in range(np.size(input, 0)):
        pic = input[i]
        pic_2d = np.reshape(pic, (w, h)) 

        enhanced_tile = [pic]
        for op in enhance_ops:
            
            enhanced = op(pic_2d).flatten()
            if (
                np.sum(pic) != 0 and 
                np.sum(enhanced) < np.size(enhanced) and 
                np.abs(np.sum(enhanced) - np.sum(pic)) < np.size(pic) // 30
                ):
                enhanced_tile.append(enhanced)

        print(len(enhanced_tile))
        result.append(np.array(enhanced_tile))

    return np.array(result)