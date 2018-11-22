import numpy as np
import random

from petnet.tensor import Tensor
from petnet.train import train, evaluate
from petnet.nn import NeuralNet
from petnet.layers import Linear, Tanh, Sigm
from petnet.data import BatchIterator, GenIterator, Epoch

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
dta_size = 10000

word_max_len = 7
bin_len = 5
repre_size = word_max_len * bin_len

positive_p = 0.7

words = [
    "hi",
    "zero",
    "deleted",
    "create",
    "eat",
    "busstop"
    "test",
    "word",
    "on"
    "testing"
]
# Prepare dictionary & positions translated to binary form
translated_words = np.zeros((len(words), repre_size))
translated_positions = np.zeros((len(words), len(words)))
translated_positions_invalid = np.zeros(len(words))
for i in range(len(words)):
    translated_words[i] = word_to_bin(words[i])
    translated_positions[i][i] = 1


# Function that generates data for one epoch
def gen_data(translated_words, translated_positions, translated_positions_invalid):   
    inputs = np.zeros((dta_size, repre_size))
    targets = np.zeros((dta_size, len(words)))
    for i in range(dta_size):
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
    return Epoch(inputs, targets)

# Create NN
net = NeuralNet([
    Linear(input_size=repre_size, output_size=5, name="lin1"),
    Sigm("sigm1"),
    Linear(input_size=5, output_size=len(words)),
    Sigm()
])

# Train network
train(net, GenIterator(lambda: gen_data(translated_words, translated_positions, translated_positions_invalid)), 250)

for x, y in zip(translated_words, translated_positions):
    predicted = net.forward(x)
    print(np.argmax(predicted), np.argmax(y))

def is_correct(output, gold):
    processed_g = np.argmax(gold)
    if np.max(output) < 0.5: # our network guessed that the word isn't from dict
        processed_o = 0 
    else:
        processed_o = np.argmax(output) 
        if np.sum(output > 0.5) > 1: return False # multiple words from dictionary selected (stronger than just using just argmax)

    return processed_g == processed_o

# Generate new set of data for final evaluation
inputs, targets = gen_data(translated_words, translated_positions, translated_positions_invalid)
evaluate(net, inputs, targets, is_correct)
