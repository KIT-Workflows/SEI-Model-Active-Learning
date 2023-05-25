from multiprocessing import Process, Queue, Manager
import random
from random import uniform, choice, randrange, shuffle
#from kmcol import KMC
import KMC
import numpy as np
import pandas as pd
import time, yaml, joblib
from glob import glob
import shutil
import json

def XKML(iDX):

    pdfile = args['PDFILE']
    poutdir=args['OUTDIR']

    barr = PF.iloc[iDX].to_list()[:15]

    xdim, ydim, T = 50, 50, 300

    hashh = f'{poutdir}-{iDX}'

    base = 'data_' + hashh

    ol, oi, ct = KMC.SEI( barr, base)

    TOT = KMC.Analy(base, oi, 'list_LAST', barr, iDX, Freq=1, Concentration=1, Porosity=1, Thickness=1, Flux=1)
    np.save(base + f'/TOTA{iDX}', np.array(TOT, dtype=object))

    try:
        return TOT
    except Exception:
        return 0

if __name__ == '__main__':

    #idx = int( sys.argv[1])
    with open('rendered_wano.yml') as file:
        wano_file = yaml.full_load(file)

    idx = int(wano_file["kmc-index"])

    with open('sei_args.yml') as file:
        args = yaml.full_load(file)
    
    args["time"] = wano_file["kmc-time"]

    with open('sei_args.yml', 'w') as file:
        yaml.dump(args, file)

    PF = pd.read_json(args['PDFILE'])
        
            
    tot = XKML(idx)

    #np.save(f'tot{idx}',  tot)
