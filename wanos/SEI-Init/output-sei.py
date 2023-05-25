import json, os, yaml, shutil

def copy_models_from_cycle_folders(source_dir, destination_dir):
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    for root, dirs, files in os.walk(source_dir):
        for folder in dirs:
            if folder.startswith("cycle"):
                source_fastica = os.path.join(root, folder, "FastICA.model")
                source_km = os.path.join(root, folder, "KM.model")
                source_kpca = os.path.join(root, folder, "KERNPCA.model")

                if os.path.exists(source_fastica) and os.path.exists(source_km) and os.path.exists(source_kpca) :
                    shutil.copy2(source_fastica, destination_dir)
                    shutil.copy2(source_km, destination_dir)
                    shutil.copy2(source_kpca, destination_dir)



if __name__ == '__main__':

    folder_path = os.getcwd()
    copy_models_from_cycle_folders(folder_path, folder_path)
        
    source_file = "KM.model"
    new_file = "old_KM.model"
    shutil.copy2(source_file, new_file)

    source_file = "FastICA.model"
    new_file = "old_FastICA.model"
    shutil.copy2(source_file, new_file)

    source_file = "KERNPCA.model"
    new_file = "old_KERNPCA.model"
    shutil.copy2(source_file, new_file)

    # Open the JSON file
    # with open('PF.json', 'r') as f:
    #     data = json.load(f)

    # num_index = len(data["e1"])

    # outdict = {"max_index": num_index}

    # with open("output_dict.yml",'w') as out:
    #     yaml.dump(outdict, out, default_flow_style=False)



