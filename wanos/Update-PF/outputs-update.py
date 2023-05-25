import numpy as np
import yaml, sys, joblib, re, os, shutil
import pandas as pd

def find_keys_starting_with_TOTA(dictionary, key_path=None):
    if key_path is None:
        key_path = []

    matching_keys = []

    for key, value in dictionary.items():
        current_key_path = key_path + [key]
        if isinstance(value, dict):
            matching_keys.extend(find_keys_starting_with_TOTA(value, current_key_path))
        elif key.startswith("TOTA"):
            matching_keys.append(current_key_path)

    return matching_keys

def rename_keys_with_TOTA(dictionary):
    new_dict = {}
    pattern = re.compile(r'^TOTA(\d+)$')

    for key, value in dictionary.items():
        new_key = match.group(1) if (match := pattern.match(key)) else key
        new_dict[new_key] = value

    return new_dict

def rename_keys_with_TOTATime(dictionary):
    new_dict = {}

    for key, value in dictionary.items():
        new_key = key.split('TA')[1].split('Time')[0]
        new_dict[new_key] = value

    return new_dict

def rename_keys_with_TOTAinF(dictionary):
    new_dict = {}

    for key, value in dictionary.items():
        new_key = key.split('TA')[1].split('inF')[0]
        new_dict[new_key] = value

    return new_dict

def rename_keys_with_TOTAorF(dictionary):
    new_dict = {}

    for key, value in dictionary.items():
        new_key = key.split('TA')[1].split('orF')[0]
        new_dict[new_key] = value

    return new_dict



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

if __name__ == '__main__':

    with open('kmc_database.yml') as file:
        db_file = yaml.full_load(file)

    key_paths = find_keys_starting_with_TOTA(db_file)
    k1 = []
    kTime = []
    kinF = []
    korF = []
    for i in key_paths:
        if i[1].endswith('Time'):
            kTime.append(i)
            continue
        if i[1].endswith('inF'):
            kinF.append(i)
            continue
        if i[1].endswith('orF'):
            korF.append(i)
            continue
        k1.append(i)

    dict_out = {
        key_list[1]: db_file[key_list[0]][key_list[1]]
        for 
        key_list in k1
    }
    
    dict_out_2 = {
        key_list[1]: db_file[key_list[0]][key_list[1]]
        for
        key_list in kTime
    }

    dict_out_3 = {
        key_list[1]: db_file[key_list[0]][key_list[1]]
        for
        key_list in kinF
    }
    dict_out_4 = {
        key_list[1]: db_file[key_list[0]][key_list[1]]
        for
        key_list in korF
    }

    dict_out = rename_keys_with_TOTA(dict_out)
    dict_out_2 = rename_keys_with_TOTATime(dict_out_2)
    dict_out_3 = rename_keys_with_TOTAinF(dict_out_3)
    dict_out_4 = rename_keys_with_TOTAorF(dict_out_4)

    dict_out_keys = dict_out.keys()
    dict_out_keys = np.array(list(dict_out_keys), dtype=np.int32)
    dict_out_vals = np.array(list(dict_out.values()))
    
    dict_out_keys_2 = dict_out_2.keys()
    dict_out_keys_2 = np.array(list(dict_out_keys_2), dtype=np.int32)
    dict_out_vals_2 = np.array(list(dict_out_2.values()))

    dict_out_keys_3 = dict_out_3.keys()
    dict_out_keys_3 = np.array(list(dict_out_keys_3), dtype=np.int32)
    dict_out_vals_3 = np.array(list(dict_out_3.values()))

    dict_out_keys_4 = dict_out_4.keys()
    dict_out_keys_4 = np.array(list(dict_out_keys_4), dtype=np.int32)
    dict_out_vals_4 = np.array(list(dict_out_3.values()))


    print(dict_out_keys, dict_out_keys_2)
    KM = joblib.load('KM.model')
    ICM = joblib.load('FastICA.model')
    XX = ICM.transform(dict_out_vals)
    KMP = KM.predict(XX)
    KMT = KM.transform(XX)

    tmp = []
    newk = []
    for i in dict_out_keys:
        ind = np.argsort(KMT[i], )[:2]
        two = KMT[i][ind]
        if np.abs(two[1] - np.mean(KMT[i]) ) < 1.2*two[0]:
            tmp.append(KMP[i].tolist())
            newk.append(i)

    KMP = np.array(tmp, dtype=np.int32)

    PF = pd.read_json('PF.json')

    j = np.column_stack((PF.iloc[newk], KMP , dict_out_vals_2[newk], dict_out_vals_3[newk], dict_out_vals_4[newk] ))

    Tdata = pd.read_json('Tdata.json')

    if len( Tdata.columns) == 19 :
        k = np.column_stack((Tdata.iloc[:,:15] , Tdata.KL, Tdata.Time , Tdata.inF, Tdata.orF))
    else:
        k = np.column_stack((Tdata.iloc[:,:15] , Tdata.KL, np.zeros_like(Tdata.KL), np.zeros_like(Tdata.KL), np.zeros_like(Tdata.KL) ))

    columns = ['e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7', 'e8', 'e9', 'e10', 'e11','e12', 'e13', 'e14', 'e15', 'KL', 'Time', 'inF','orF']
    dr = pd.DataFrame( np.row_stack((j,k)) , columns = columns)
    dr = dr.astype({"KL": int})
    dr.fillna(0, inplace=True)
    dr =dr[(dr.Time<2) & (dr.Time> 1e-6) | (dr.Time==0)]
    dr.to_json('TdataUP.json')
    #print(dr)
   
    copy_to_temp_Tdata()

    
