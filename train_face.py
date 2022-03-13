import os
# ↑ import files ↑

file_directory = os.getcwd() + '\\faces_to_train' # getting location of faces folder

for root, dirs, files in os.walk(file_directory):
    for file in files:
        if file.endswith('png') or file.endswith('jpg'):
            path = file_directory + f'\\{file}'
            label = os.path.basename(root)
            print(label, path)