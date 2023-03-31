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

Nm = time.ctime().split()[3]
print('model version: ', Nm)
modelout='modelout'

def DataToKmean(Q):
    svd =FastICA(n_components=10, whiten='unit-variance', max_iter=10000)
    Data=svd.fit_transform( Q ) 
    kmeanmodel = KMeans(
            n_clusters=4, init='random',
            n_init=1000, max_iter=90000,
            tol=1e-10, random_state=0, )

    kmeanlable=kmeanmodel.fit_predict(Data)
    joblib.dump(kmeanmodel, f'{modelout}/KM-{Nm}.model')
    joblib.dump(svd, f'{modelout}/FastICA-{Nm}.model')
    return svd, kmeanmodel, kmeanlable, Data


def PLOTkmean(Q, KL):
    pcakmrp = PCA(n_components=2)
    pca_resultkmrp = pcakmrp.fit_transform(Q)
    Co=np.append(matplotlib.cm.rainbow(np.linspace(0, 1, 5)),matplotlib.cm.Accent(np.linspace(0, 1, 5)),axis=0)
    color = iter(Co)

    fig= plt.figure( figsize=(10,8))
    ax = fig.add_subplot(111)
    for k in np.unique(KL):
        c = next(color)
        ax.scatter(
        pca_resultkmrp[KL == k, 0], pca_resultkmrp[KL == k, 1],
         label=f'Region {k+1}', color=c, s=5)
    plt.legend(fontsize=12,markerscale=3)

    
def PLOTData(data, KM, KL, ):
    ww= pd.DataFrame(KM.transform(data))
    ww['KL']=KL
    
    Lis=[]
    rad = np.arange(0,20,0.005)

    for I in np.unique(KL):
        tm=[]
        for r in rad:
            tm.append(len(ww[(ww[I]<r) & (ww['KL']==I)] ) )
        Lis.append(tm)   

    for ii in np.unique(KL):
        plt.figure()
        plt.plot(rad, Lis[ii], label=f'Region {ii+1}')
        plt.legend()
        plt.savefig(f'{modelout}/Kluster-{ii+1}')
        
    return ww, Lis, rad




def DataTrimm(r_dic, data, rawdata, KL, Q):

    indxx =[]
    for TT in np.unique(data.KL):
        I = TT
        indxx.append(data[(data[I]< r_dic[I]) & (data.KL==I)].index.to_list())
    k=[]
    for i in indxx:
        for j in i:
            k.append(j)
    rawdata['KL']= KL
    rawx=rawdata.iloc[k]
    zz= rawx.index
    
    pcakmrp = PCA(n_components=2)
    pca_resultkmrp = pcakmrp.fit_transform(Q)[zz]
    
    Co=np.append(matplotlib.cm.rainbow(np.linspace(0, 1, 5)),matplotlib.cm.Accent(np.linspace(0, 1, 5)),axis=0)
    color = iter(Co)
    fig= plt.figure( figsize=(8,6))
    ax = fig.add_subplot(111)
    for k in [0,1,2,3]:
        c = next(color)
        ax.scatter(
        pca_resultkmrp[KL[zz] == k, 0], pca_resultkmrp[KL[zz] == k, 1],
        label=f'clx {k}', color=c, s=5)
    plt.legend(fontsize=12,markerscale=3)


    return rawx


def KernPCA(data):
    Kpca = KernelPCA(n_components=15, kernel='poly',gamma= 1e-4, 
                        fit_inverse_transform=True,
                        alpha=1e-5,
                        eigen_solver='dense',
                        remove_zero_eig=True,
                        degree=15,coef0=2)

    kpca_result = Kpca.fit_transform(data.iloc[:,:15])
    
    fig = plt.figure(356,figsize=(6, 6))
    sns.scatterplot(x=kpca_result[:,1], y=kpca_result[:,0],
               s=2*(data.KL+1.5)**3, hue=data.KL, alpha= 0.6, palette='Set2')
    plt.show()
    

    joblib.dump(Kpca,f'{modelout}/KERNPCA-{Nm}.model')
    return Kpca , kpca_result


def Trx(data, inv ,L=[0.1, 0.2]):
    
    kd_tree1 = KDTree(data)
    kd_tree2 = KDTree(inv)
    ls=[]
    for I in L:

        indexes = kd_tree1.query_ball_tree(kd_tree2, p=2, r=I)

        fl=[]
        for i in indexes:
            for j in i:
                fl.append(j)

        h,k = np.unique(fl, return_counts=True)
        
        ls.append(h)
    
    hx = [i for i in ls[1] if i not in ls[0]]
    
    return hx

def Space(samplsz, KPCA, data ):
    al=[]
    RANGE =np.arange(0,0.501, 0.02)
    KI = samplsz
    TRXX = partial(Trx,data)
    
   
    for G in KI:
        sampler = qmc.Sobol(d=15, scramble=True)
        sample = sampler.random_base2(m=G)
        LOP = 0
        a = KDTree(data)
        l_bounds = a.mins - LOP
        u_bounds = a.maxes + LOP
        pp = qmc.scale(sample, l_bounds, u_bounds)
        invN = KPCA.inverse_transform(pp)
        L=[]
        print()
        for I in RANGE :
            I= np.round(I,3)

            ii= TRXX( inv=invN, L=[.0,I])
            L.append(len(ii))
            print(I, len(ii), end='\r', flush=True)

        al.append(L)
        
    from scipy.optimize import curve_fit

    def func(x, a, b):
        return 1 / ( np.exp( -(x-a)/(b))  + 1)

        
    lindata=[]
    for T in range(len(al)):
        xdata= RANGE
        iii=T
        ydata= al[iii]/np.max(al[iii])

        popt, pcov = curve_fit(func, xdata, ydata, bounds=([1e-6,1e-6], [3,3]))
        lindata.append(popt)
        print(KI[T], popt, sep='\n' )
        print()
    
    print(np.max(invN,axis=0), np.min(invN,axis=0))

    for j in al:
        plt.plot(RANGE, j/np.max(j),'-x', label=str(np.max(j)), )
        plt.plot(RANGE, func(RANGE, popt[0], popt[1]), label='func')

    plt.legend()
    plt.show()
    return popt, pp, invN
