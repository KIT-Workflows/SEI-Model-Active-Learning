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



with open('sei_args.yml') as file:
    args = yaml.full_load(file)
file.close()

PF = pd.read_json(args['PDFILE'])

def XKML(iDX, return_dict):

    pdfile = args['PDFILE']
    poutdir=args['OUTDIR']

    barr = PF.iloc[iDX].to_list()[:15]

    xdim, ydim, T = 50, 50, 300

    hashh = f'{poutdir}-{iDX}'

    base = 'data_' + hashh

    ol, oi, ct = KMC.SEI( barr, base)


    #print('\n', oi, f'sim time: {ct}')

    TOT = KMC.Analy(base, oi, 'list_LAST', barr, iDX, Freq=1, Concentration=1, Porosity=1, Thickness=1, Flux=1)
    np.save(base + f'/TOTA{iDX}', np.array(TOT, dtype=object))
    
    try:
        return_dict[iDX] =  TOT
    except:
        pass
        

manager = Manager()
return_dict = manager.dict()

pp = [Process(target=XKML, args=(i, return_dict)) for i in range(len(PF))]
t0=0
N = args['NCPU']
CHECK = args['CHECKCPU']
while len( [u for u in pp if u.exitcode in [0,1]]) < len(pp):
    alv = [u for u in pp if u.is_alive()]
    lp=[]
    #lp = [u for u in pp if not u.is_alive()]
    for u in pp:
        if not u.is_alive() and not u.pid:
            lp.append(u)
    if lp:
        print('pending: ', len(lp),'alive: ', len(alv))

        for i in np.random.choice(lp,size=min(len(lp), N-len(alv)),replace=False):
            i.start()
    #print(pp)
    time.sleep(CHECK)

for D in glob('*' + args['OUTDIR'] + '*'):
    shutil.rmtree(D)


np.save('dictK',  return_dict.keys())
np.save('dictV',  return_dict.values())
