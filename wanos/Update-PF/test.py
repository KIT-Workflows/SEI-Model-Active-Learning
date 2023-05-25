import os
import shutil

def copy_to_temp_Tdata():
    current_dir = os.getcwd()  # get current directory
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))  # get parent directory
    
    temp_Tdata_dir = os.path.join(parent_dir, 'temp_Tdata')  # create path to temp_Tdata folder
    
    if not os.path.exists(temp_Tdata_dir):  # create temp_Tdata folder if it doesn't exist
        os.mkdir(temp_Tdata_dir)
    
    src_file = os.path.join(current_dir, 'Tdata.json')  # path to source file
    dest_file = os.path.join(temp_Tdata_dir, 'Tdata.json')  # path to destination file
    
    shutil.copy(src_file, dest_file)  # copy file to temp_Tdata folder

copy_to_temp_Tdata()  # call the function
