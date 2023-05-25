import matplotlib.pyplot as plt
import itertools, os, copy, random, sys, time, pickle
from functools import partial
from sklearn.decomposition import KernelPCA, PCA, FastICA
from sklearn.cluster import KMeans
from scipy.spatial import KDTree
import matplotlib
import joblib  
from scipy.stats import qmc
import numpy as np
import seaborn as sns
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, DotProduct, WhiteKernel, ExpSineSquared, RationalQuadratic, Matern, ConstantKernel
import subprocess

from functown import FT2


# Q = np.load('data1.npy')
# C = pd.read_json('data2.json')

# ICM, KM, KL, Q20 = FT2.DataToKmean(Q)
# FT2.PLOTkmean(Q,KL)


# newd, rng, rad = FT2.PLOTData(Q20,KM,KL)
# erl = []

# VRZ = 1000

# print(VRZ)
# rdic= {}

# for i in np.unique(KL):
#     try:
#         rdic.update({i: np.round( rad[np.where( np.array( rng[i]) < VRZ )[0][-1] + 1 ], 3) })
#     except:
#         rdic.update({i: np.round( rad[np.where( np.array( rng[i]) < VRZ )[0][-1]  ], 3) })

# print(rdic)
# Tdata= FT2.DataTrimm(rdic,newd,C,KL, Q)

if __name__ == '__main__':
    with open('rendered_wano.yml') as file:
        wano_file = yaml.full_load(file)
    
    ti = wano_file["TI"] 

    VRZ = 3000
    Tdata = pd.read_json('Tdata.json')
    erl = []
    err ,model, likelihood, matrix = FT2.InvInf(Tdata,TI=ti, EV=1000, XX=[5])
    erl.append(err)
    print()

    np.save(f'errGP-{VRZ}', np.array(erl))

    temp_dict = {}
    temp_dict["err"] = np.array(erl, dtype=np.float64).tolist()
    temp_dict["matrix"] = matrix.tolist()


    with open('sei_al_results.yml', 'w') as f:
        yaml.dump(temp_dict, f) # aqui

    kpca = joblib.load('KERNPCA.model')
    #kpca, kpc_res = FT2.KernPCA(Tdata)

    kpc_res = kpca.transform(Tdata.iloc[:,:15])

    np.save('TDATA_kpc_res', kpc_res)


    G = 13
    PO, pp, invN = FT2.Space([G], kpca, Tdata.iloc[:,:15])

    '''
    R = {0: 0, 1:PO[0] - PO[1]*np.log(2), 2:PO[0] + PO[1]*np.log(2), 3:PO[0] + 6*PO[1]*np.log(10)}
    sampler = qmc.Sobol(d=15, scramble=True)
    sample = sampler.random_base2(m=G)

    LOP = 0
    a = KDTree(kpc_res)
    l_bounds = [-1]*15 #a.mins - LOP
    u_bounds = [1]*15 # a.maxes + LOP
    pp = qmc.scale(sample, l_bounds, u_bounds)
    invN = kpca.inverse_transform(pp)
    indx1 = FT2.Trx(Tdata.iloc[:,:15], invN, L=[R[0], R[1]])
    indx2 = FT2.Trx(Tdata.iloc[:,:15], invN, L=[R[1], R[2]])
    indx3 = FT2.Trx(Tdata.iloc[:,:15], invN, L=[R[2], R[3]])
    print(len(indx1), len(indx2), len(indx3))
    SPACE = {'stage1': [indx1, 0.40], 'stage2': [indx2, 0.35], 'stage3': [indx3, 0.40]}
    '''

    R2= {0:0, 1:PO[0] , 2: 0.6}

    indx1_n = FT2.Trx(Tdata.iloc[:,:15], invN, L=[R2[0], R2[1]])
    indx2_n = FT2.Trx(Tdata.iloc[:,:15], invN, L=[R2[1], R2[2]])

    print(len(indx1_n), len(indx2_n))

    SPACE_n = {'stage1': [indx1_n, 0.40], 'stage2': [indx2_n, 0.55]}

    #m1, lh1 = FT2.Mload()
    tm2=[]
    for i in ['stage1']:
        _, tm1 = FT2.Evaal(model,likelihood,  inputP = invN[SPACE_n[i][0]]  )
        tm2.append( np.where( np.sum( np.square(tm1), axis=0) < SPACE_n[i][1])[0]  )
    final=[]
    for i in tm2:
        for j in i:
            if j not in final:
                final.append(j)
    final.sort()
    print('new points: ', len(final))
    PF =pd.DataFrame(invN[final], columns=Tdata.iloc[:,:15].columns )

    PF.to_json(f'PF.json')
    #subprocess.run("sbatch jobeng.sh",shell=True)
