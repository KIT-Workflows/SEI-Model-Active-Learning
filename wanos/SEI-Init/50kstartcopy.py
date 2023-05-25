from multiprocessing import Process, Queue, Manager
import random
from random import uniform, choice, randrange, shuffle
from kmcol import KMC
import numpy as np
import pandas as pd
import time, yaml, joblib
from glob import glob
import shutil
import json


idx = int( sys.argv[1])

with open('sei_args.yml') as file:
    args = yaml.full_load(file)
file.close()

PF = pd.read_json(args['PDFILE'])

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
    except:
        return 0
    
        

tot = XKML(idx)

np.save(f'tot{idx}',  tot)
