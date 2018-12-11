import os 
from os.path import isfile, join

index_to_char = {
    "1":"|",
    "2":"_",
    "3":"/",
}

dir_path = os.path.dirname(os.path.realpath(__file__))
pic_names = [f for f in os.listdir(dir_path) if isfile(join(dir_path, f)) if len(f) > 4 and f[-4:] == ".png"]


char_i_to_file = {}
file_to_char_i = {}
for f in pic_names:
    char_index, _ = f.split("_")
    file_name_full = join(dir_path, f)
    if char_index in char_i_to_file:
        char_i_to_file[char_index].append(file_name_full)
    else:
        char_i_to_file[char_index]= [file_name_full]
    file_to_char_i[file_name_full] = char_index


print(char_i_to_file)
print(file_to_char_i)