import os, glob, yaml

if __name__ == '__main__':

    # Get the current working directory
    cwd = os.getcwd()

    # Get the parent directory of the current working directory
    parent_dir = os.path.abspath(os.path.join(cwd, os.pardir))

    # Use listdir to get a list of all the directories in the parent directory
    all_dirs = [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]
    
    # Remove DB-Generator folder
    substring = 'DB-Generator'
    all_dirs = [i for i in all_dirs if substring not in i]


    wanos_lst = []

    for temp_dir in all_dirs:
        substring = "s-"
        temp_wano = temp_dir.split(substring)
        temp_dict = {}
        temp_dict['Dict-Name'] ='*results.yml'
        temp_dict['Wano-Name'] = temp_wano[1]
        wanos_lst.append(temp_dict)
    
    with open('rendered_wano.yml') as file:
        wano_file = yaml.full_load(file)
    
    if wano_file["Full-DB"]:
        # Files db
        files_db = wanos_lst
    else:
        # Files db
        files_db = wano_file["Wanos"]
    
    # print(files_db)
    name_db = wano_file["DB-Name"]
    
    
    db_dict = {}

    for search_dict in files_db:
        search_string = search_dict["Dict-Name"]
        search_wano = search_dict["Wano-Name"] 
        
        # Create a new list that contains only the strings that contain the substring
        dirs = list(filter(lambda x: search_wano in x, all_dirs))

        # print(search_string)
        # print(search_wano)
        # print(dirs)

        # Use a for loop to search each folder in the list
        for folder in dirs:

            # Use glob to search for files that ends with the search sub-string
            files = glob.glob(os.path.join("../" + folder, search_string))
            print(files)
            if files:
                with open(files[0]) as file:
                    wano_db = yaml.full_load(file)
                elements = files[0].split('/')
                wano_db["id_file"] = elements[2]
                db_dict[elements[1] + '_' + elements[2]] = wano_db
            else:
                add_rendered_wano = os.path.join("../" + folder, 'rendered_wano.yml')
                with open(add_rendered_wano) as file:
                    wano_db = yaml.full_load(file)
                elements = add_rendered_wano.split('/')
                wano_db["id_file"] = elements[2]
                db_dict[elements[1] + '_' + elements[2]] = wano_db
                print("rendered_wano.yml added")
    # Print the list of files    

    with open(name_db + ".yml",'w') as out:
        yaml.dump(db_dict, out,default_flow_style=False)
