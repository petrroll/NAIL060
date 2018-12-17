import os 
from os.path import isfile, join

# Tile picture names are in format "<charIndex>_<whatever>"
# ..where charIndex specifies which char should the tile be 
# ..resolved to in picture->ASCII translation. 
# charIndex is 0-based.

# Mapping of charIndex -> specicif ASCII character
char_i_to_char = {
    0: "|",
    1: "_",
    2: "/",
    3: " ",
}

def get_picture_paths():
    '''
    Returns full paths of all pictures in current folder.
    '''
    dir_path = os.path.dirname(os.path.realpath(__file__))
    pic_names = [f for f in os.listdir(dir_path) if isfile(join(dir_path, f)) if len(f) > 4 and f[-4:] == ".png"]

    return pic_names

def get_data():
    '''
    Returns triple of (char_i_to_char, char_i_to_files, file_to_char_i)
    
    char_i_to_char: Map charIndex -> ASCII character
    char_i_to_files: Map charIndex -> list of full file paths
    file_to_char_i: Map file path -> charIndex
    '''
    dir_path = os.path.dirname(os.path.realpath(__file__))
    pic_names = get_picture_paths()    

    char_i_to_files = {}
    file_to_char_i = {}
    for f in pic_names:
        char_index_s = f.split("_")[0]
        char_index = int(char_index_s)

        file_name_full = join(dir_path, f)
        if char_index in char_i_to_files:
            char_i_to_files[char_index].append(file_name_full)
        else:
            char_i_to_files[char_index]= [file_name_full]
        file_to_char_i[file_name_full] = char_index

    return (char_i_to_char, char_i_to_files, file_to_char_i)
