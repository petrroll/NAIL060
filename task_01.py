import numpy as np
import random

from petnet.tensor import Tensor
from petnet.train import train
from petnet.nn import NeuralNet
from petnet.layers import Linear, Tanh

def char_to_int(char: str) -> int:
    if len(char) != 1: raise ValueError("Must be given one char")
    
    if char == " ": return 0
    elif ord(char) >= ord('a') and ord(char) <= ord('z'): return ord(char) - ord('a') + 1
    else: return -1 # Unknown character

def int_to_bin(number: int, bin_max_len: int) -> Tensor:
    return np.array(list(np.binary_repr(number, bin_max_len))).astype(np.float64)

def word_to_bin(word: str, word_max_len: int = 7, bin_len: int = 5) -> Tensor:
    bin_arr = np.zeros(word_max_len * bin_len)
    for char_i in range(word_max_len):
        char_c = word[char_i] if char_i < len(word) else " "
        char_int = char_to_int(char_c)
        char_bin = int_to_bin(char_int, bin_len)
        for j in range(bin_len):
            bin_i = char_i * bin_len + j
            bin_arr[bin_i] = char_bin[j]
    
    return bin_arr

# Params
dta_size = 1000

word_max_len = 7
bin_len = 5
repre_size = word_max_len * bin_len

positive_p = 0.7

# Positive words dictionary
words = [
    "hi",
    "zero",
    "deleted",
    "create",
    "eat",
    "busstop"
    "test"
]

# Prepare dictionary & positions translated to binary form
translated_words = np.zeros((len(words), repre_size))
translated_positions = np.zeros((len(words), len(words)))
translated_positions_invalid = np.zeros(len(words))
for i in range(len(words)):
    translated_words[i] = word_to_bin(words[i])
    translated_positions[i][i] = 1

# Prepare input & target data
inputs = np.zeros((dta_size, repre_size))
targets = np.zeros((dta_size, len(words)))
for i in range(5000):
    if random.random() <= positive_p:   
        # Select positive example from translated dictionary
        random_word_i = random.randrange(0, len(words))
        inputs[i] = translated_words[random_word_i]
        targets[i] = translated_positions[random_word_i]
    else:
        # Create negative example
        targets[i] = translated_positions_invalid

        if random.random() <= 0.5:
            # Modification of existing word
            random_word_i = random.randrange(0, len(words))
            word = list(words[random_word_i])
            for j in range(len(word)):
                word[j] = chr(random.randint(ord('a'), ord('z'))) if random.random() < 0.3 else word[j]

            inputs[i] = word_to_bin(str(word), word_max_len, bin_len)
        else:
            # Create random binary array
            input = np.zeros(word_max_len * bin_len)
            for j in range(len(input)):
                input[j] = 1 if random.random() < 0.3 else 0

            inputs[i] = input


net = NeuralNet([
    Linear(input_size=repre_size, output_size=20, name="lin1"),
    Tanh("tanh"),
    Linear(input_size=20, output_size=len(words)),
    Tanh()
])

train(net, inputs, targets, 1000)

for x, y in zip(translated_words, translated_positions):
    predicted = net.forward(x)
    print(np.argmax(predicted), np.argmax(y))

