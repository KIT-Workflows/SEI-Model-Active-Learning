import matplotlib.pyplot as plt
import numpy as np, os, tarfile, yaml, sys, shutil


if __name__ == '__main__':

    with open('rendered_wano.yml') as file:
        wano_file = yaml.full_load(file)
    
    #descriptor = wano_file["Descriptor"]
    
    #if not descriptor:
    
    with open('KM.model', 'w') as f:
        f.write('Create KM.model file!')

    with open('FastICA.model', 'w') as f:
        f.write('Create FastICA.model file!')


    fname = "folder1.tar.xz"

    with tarfile.open(fname, 'r:xz') as tar:
    # Extract all contents of the folder
        tar.extractall()

    folder_path = f"{os.getcwd()}/folder1"
    contents = os.listdir(folder_path)
    # Move each item to the parent directory
    for item in contents:
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path):
            shutil.move(item_path, os.path.join(os.path.dirname(folder_path), item))
        elif os.path.isdir(item_path):
            shutil.move(item_path, os.path.dirname(folder_path))
    
    shutil.rmtree("folder1")
    