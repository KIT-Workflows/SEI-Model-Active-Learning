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


Q = np.load('data1.npy')
C = pd.read_json('data2.json')

ICM, KM, KL, Q20 = FT2.DataToKmean(Q)
FT2.PLOTkmean(Q,KL)

try:
    assert (len(KL[KL==2]) < len(KL[KL==0]) ) and (len(KL[KL==2]) < len(KL[KL==1]) ) and (len(KL[KL==2]) < len(KL[KL==3]) )
except AssertionError:
    print('The empty class label now is different. In this file on line 53, and in SEI-AL/folder1.tar.xz/functown/FT2.py lines: 255-258 adjust this condition according to the current Kmean labels: min(len(train_y[train_y == 2]),G). You can delete this assertion afterward.')



newd, rng, rad = FT2.PLOTData(Q20,KM,KL)
erl = []

VRZ = 3000

print(VRZ)
rdic= {}

for i in np.unique(KL):
    try:
        rdic.update({i: np.round( rad[np.where( np.array( rng[i]) < VRZ )[0][-1] + 1 ], 3) })
    except:
        rdic.update({i: np.round( rad[np.where( np.array( rng[i]) < VRZ )[0][-1]  ], 3) })

print(rdic)
Tdata= FT2.DataTrimm(rdic,newd,C,KL, Q)

Tdata = pd.concat( [Tdata[Tdata.KL==0].sample(VRZ), Tdata[Tdata.KL==1].sample(VRZ),
                    Tdata[Tdata.KL==2].sample(min(len(Tdata[Tdata.KL==2]),VRZ)), Tdata[Tdata.KL==3].sample(VRZ)  ], ignore_index=True)

Tdata.to_json('Tdata.json')
kpca, kpc_res = FT2.KernPCA(Tdata)

# erl.append(err)
# print()

# np.save(f'errGP-{VRZ}', np.array(erl))


# kpca, kpc_res = FT2.KernPCA(Tdata)

# G = 10
# PO, pp, invN = FT2.Space([G], kpca, Tdata.iloc[:,:15])
# R = {0: 0, 1:PO[0] - PO[1]*np.log(2), 2:PO[0] + PO[1]*np.log(2), 3:PO[0] + 6*PO[1]*np.log(10)}
# sampler = qmc.Sobol(d=15, scramble=True)
# sample = sampler.random_base2(m=G)

# LOP = 0
# a = KDTree(kpc_res)
# l_bounds = a.mins - LOP
# u_bounds = a.maxes + LOP
# pp = qmc.scale(sample, l_bounds, u_bounds)
# invN = kpca.inverse_transform(pp)

# indx1 = FT2.Trx(Tdata.iloc[:,:15], invN, L=[R[0], R[1]])
# indx2 = FT2.Trx(Tdata.iloc[:,:15], invN, L=[R[1], R[2]])
# indx3 = FT2.Trx(Tdata.iloc[:,:15], invN, L=[R[2], R[3]])

# print(len(indx1), len(indx2), len(indx3))


# SPACE = {'stage1': [indx1, 0.35], 'stage2': [indx2, 0.55], 'stage3': [indx3, 0.75]}
# #m1, lh1 = FT2.Mload()
# tm2=[]
# for i in SPACE:
#     _, tm1 = FT2.Evaalcuda(model,likelihood,  inputP = invN[SPACE[i][0]]  )
#     tm2.append( np.where( np.sum( np.square(tm1), axis=0) < SPACE[i][1])[0]  )
# final=[]
# for i in tm2:
#     for j in i:
#         if j not in final:
#             final.append(j)
# final.sort()

# PF =pd.DataFrame(invN[final], columns=Tdata.iloc[:,:15].columns )

# PF.to_json(f'PF.json')
# #subprocess.run("sbatch jobeng.sh",shell=True)
