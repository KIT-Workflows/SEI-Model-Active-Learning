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

import time
import torch
import gpytorch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel, InducingPointKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.variational import VariationalStrategy
from gpytorch.variational import CholeskyVariationalDistribution
from sklearn.metrics import confusion_matrix


class StandardApproximateGP(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, num_classes):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(self, inducing_points, variational_distribution,
                                                                        learn_inducing_locations=True)
        super().__init__(variational_strategy)
        
        
        self.mean_module = ConstantMean(batch_shape=torch.Size((num_classes,)))
        self.covar_module = ScaleKernel(RBFKernel(batch_shape=torch.Size((num_classes,))),batch_shape=torch.Size((num_classes,)),)


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    


Nm = time.ctime().split()[3]
print('model version: ', Nm)

os.mkdir(f'cycle{Nm}')

modelout=f'cycle{Nm}'

def DataToKmean(Q):
    svd =FastICA(n_components=10, whiten='unit-variance', max_iter=10000)
    Data=svd.fit_transform( Q ) 
    kmeanmodel = KMeans(
            n_clusters=4, init='random',
            n_init=1000, max_iter=90000,
            tol=1e-10, random_state=0, )

    kmeanlable=kmeanmodel.fit_predict(Data)
    joblib.dump(kmeanmodel, f'{modelout}/KM.model')
    joblib.dump(svd, f'{modelout}/FastICA.model')
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
    plt.savefig(f'{modelout}/Kmean')


    
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
        label=f'Region {k+1}', color=c, s=5)
    plt.legend(fontsize=12,markerscale=3)
    plt.savefig(f'{modelout}/Kluster-trimmed')
    rawx.to_json(f'{modelout}/Tdata.json')

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
    

    joblib.dump(Kpca,f'{modelout}/KERNPCA.model')
    np.save(f'{modelout}/TDATA_kpc_res', kpca_result)

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


def InvInf(datapd, trsize= 0.7, TI=1000, EV=100, XX= [3, 5, 7, 10, 15, 20] ):

    ld = len(datapd)*trsize/4
    XXX=[]
    for XXi in XX:
        XXX.append(XXi/100*ld)

    for G in XXX:
        G = int(G)
        train_x, test_x, train_y, test_y = train_test_split(datapd.iloc[:,:15], datapd.KL, test_size = 1-trsize)
        
        a0 = train_y[train_y ==0].sample(min(300,G))
        a1 = train_y[train_y ==1].sample(G)
        a2 = train_y[train_y ==2].sample(G)
        a3 = train_y[train_y ==3].sample(G)

        LS = pd.concat([a0,a1,a2,a3]).index.to_list()

        lst = torch.Tensor( train_x.loc[LS].to_numpy() )
        lst = lst.contiguous()

        train_x = torch.Tensor(train_x.to_numpy())
        train_x = train_x.contiguous()
        train_y = torch.Tensor(train_y.to_numpy()).long()
        train_y = train_y.contiguous()
        test_x = torch.Tensor(test_x.to_numpy())
        test_y = torch.Tensor(test_y.to_numpy()).long()

        if torch.cuda.is_available(): train_x, train_y, lst, test_x, test_y = train_x.cuda(), train_y.cuda(), lst.cuda(), test_x.cuda(), test_y.cuda()

        inducing_points = lst

        likelihood = gpytorch.likelihoods.DirichletClassificationLikelihood(train_y, alpha_epsilon= 1e-2, learn_additional_noise=True)

        model = StandardApproximateGP(inducing_points, num_classes=likelihood.num_classes)
        
        if torch.cuda.is_available():
            model = model.cuda()
            likelihood = likelihood.cuda() 

        print(G, f' Ratio: size of (inducing points / training ) = {inducing_points.size(0)/ train_x.size(0)}' )

        model.train()
        likelihood.train()

        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.numel())
        optimizer = torch.optim.Adam(list(model.parameters()) + list(likelihood.parameters()), lr=0.001)
        t0=time.time()
        training_iterations = TI + 1

        lg = [0,0]
        lgm = [0,0]
        Dlg ='not set'
        errcollect=[]
        for i in range(training_iterations):
            # Zero backprop gradients
            optimizer.zero_grad()
            # Get output from model
            output = model(train_x)
            # Calc loss and backprop derivatives
            loss = -mll(output, likelihood.transformed_targets ).sum()
            loss.backward()
            optimizer.step()
            LI = loss.item()
            lg.append(LI)

            lgm.append( np.mean(lg[-3:]) )
            Dlg = np.std(lgm[-3:])

            if i%EV==0:
                print('Iter %d/%d - Loss: %.6f' % (i , training_iterations, LI, ))
                print('Kernel lengthscale:  ',  [i for i in model.covar_module.base_kernel.lengthscale.cpu().detach().numpy().flatten()]) 
                print('Second noise:        ',  [i for i in likelihood.second_noise.cpu().detach().numpy().flatten()])
                print(Dlg)
                print('so far:  ', time.time()-t0)

        
                model.eval()
                likelihood.eval()

                with gpytorch.settings.fast_pred_var(), torch.no_grad():
                    test_dist = model(test_x)

                    pred_means = test_dist.loc

                #ypred = torch.round(pred_means)
                ypred = torch.argmax(pred_means , dim=0).cpu().numpy()

                matrix=confusion_matrix(test_y.cpu().numpy(), ypred)
                print(matrix)
                np.fill_diagonal(matrix,0)
                Err= np.sum(matrix)/test_y.shape[0]
                print(1-Err)
                errcollect.append([i,Err])
                model.train()
                likelihood.train()

        print( 'total time:  ', time.time()-t0)
        torch.save(model.state_dict(), f'{modelout}/GPmodel-{round(G/ld*100)}prc.pth')
        np.save(arr=inducing_points.cpu(), file=f'{modelout}/model-{round(G/ld*100)}prc-indc_point')
        return errcollect
        

def Mload(VV = 5):
    Tdata = pd.read_json('TDATA.json')
    state_dict = torch.load(f'{modelout}/GPmodel-{VV}prc.pth')
    inducing_points = torch.Tensor( np.load(f'{modelout}/model-{VV}prc-indc_point.npy', allow_pickle=True) ).contiguous()

    train_x, test_x, train_y, test_y = train_test_split(Tdata.iloc[:,:15], Tdata.KL, test_size = 0.3)

    train_x = torch.Tensor(train_x.to_numpy())
    train_x = train_x.contiguous()
    train_y = torch.Tensor(train_y.to_numpy()).long()
    train_y = train_y.contiguous()
    test_x = torch.Tensor(test_x.to_numpy())
    test_y = torch.Tensor(test_y.to_numpy()).long()

    likelihood = gpytorch.likelihoods.DirichletClassificationLikelihood(train_y, alpha_epsilon= 1e-2, learn_additional_noise=True)
    model = StandardApproximateGP(inducing_points, num_classes=likelihood.num_classes)
    model.load_state_dict(state_dict)
    return model, likelihood


def Evaal(model, likelihood, inputP):

    model.eval()
    likelihood.eval()
    inputP = torch.Tensor(inputP)
    
    with torch.no_grad():   
        test_dist = model(inputP)
        observed_pred = likelihood(model(inputP))
        mean = observed_pred.mean
        lower, upper = observed_pred.confidence_region()
        pred_means = test_dist.loc

    pred_samples = test_dist.sample(torch.Size((2**15,))).exp()
    probabilities = (pred_samples / pred_samples.sum(-2, keepdim=True)).mean(0)

    ypred = torch.argmax(pred_means , dim=0).cpu().numpy()
    return ypred, probabilities.numpy()

