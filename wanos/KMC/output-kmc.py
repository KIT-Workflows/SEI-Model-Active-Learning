import json, os, yaml, shutil
import numpy as np

def copy_file_starting_with_TOTA():
    current_directory = os.getcwd()
    data_directories = [d for d in os.listdir(current_directory) if d.startswith('data') and os.path.isdir(d)]

    for data_directory in data_directories:
        data_directory_path = os.path.join(current_directory, data_directory)
        files = os.listdir(data_directory_path)

        for file in files:
            if file.startswith('TOTA'):
                src = os.path.join(data_directory_path, file)
                dst = os.path.join(current_directory, file)
                shutil.copy(src, dst)
                print(f"Copied file '{file}' to current directory.")
                return file

    return "No file starting with 'TOTA' found in any 'data*' directories."

def convert_to_single_precision(array):
    return array.astype(np.float32)

if __name__ == '__main__':

    file_var = copy_file_starting_with_TOTA()
    var_TOTA = np.load(file_var, allow_pickle=True)
    var2dict = var_TOTA[-1].astype(np.float32).tolist()
    var2dict_2 = var_TOTA[[0,1,2,5]].astype(np.float32).tolist()
    results_dict = {file_var.split(".", 1)[0]: var2dict,
    file_var.split(".", 1)[0]+'Time': var2dict_2[3],
    file_var.split(".", 1)[0]+'inF': var2dict_2[1],
    file_var.split(".", 1)[0]+'orF': var2dict_2[2]}
    
    #with open("kmc_results.yml",'w') as out:
        #yaml.dump(results_dict, out, default_flow_style=False)

    # Load the first YAML file into a dictionary
    with open('output_kmc.yml', 'r') as f:
        output_kmc = yaml.load(f, Loader=yaml.FullLoader)

    ## Load the second YAML file into a dictionary
    #with open('file2.yml', 'r') as f:
    #data2 = yaml.load(f, Loader=yaml.FullLoader)

    # Merge the dictionaries
    merged_data = {**results_dict, **output_kmc}

    # Dump the merged data to a new YAML file
    with open('kmc_results.yml', 'w') as f:
        yaml.dump(merged_data, f)



