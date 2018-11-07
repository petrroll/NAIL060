import numpy as np
from petnet.tensor import Tensor

def char_to_int(char: str) -> int:
    if len(char) != 1: raise ValueError("Must be given one char")
    
    if char == " ": return 0
    elif ord(char) >= ord('a') and ord(char) <= ord('z'): return ord(char) - ord('a') + 1
    else: return -1 # Unknown character

def int_to_bin(number: int, bin_max_len: int) -> Tensor:
    return np.array(list(np.binary_repr(number, bin_max_len))).astype(np.float64)

def word_to_bin_arr(word: str, word_max_len: int = 7, bin_len: int = 5) -> Tensor:
    bin_arr = np.zeros(word_max_len * bin_len)
    for char_i in range(word_max_len):
        char_c = word[char_i] if char_i < len(word) else " "
        char_int = char_to_int(char_c)
        char_bin = int_to_bin(char_int, bin_len)
        for j in range(bin_len):
            bin_i = char_i * bin_len + j
            bin_arr[bin_i] = char_bin[j]
    
    return bin_arr




