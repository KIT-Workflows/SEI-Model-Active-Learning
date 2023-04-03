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

#from functown import FT

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
    


def InvInf(datapd, trsize= 0.7, TI=1000, EV=100, XX= [3, 5, 7, 10, 15, 20] ):

    ld = len(datapd)*trsize/4
    XXX=[]
    for XXi in XX:
        XXX.append(XXi/100*ld)

    for G in XXX:
        G = int(G)
        train_x, test_x, train_y, test_y = train_test_split(datapd.iloc[:,:15], datapd.KL, test_size = 1-trsize)
        
        a0 = train_y[train_y ==0].sample(min(len(train_y[train_y ==0]),G))
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
        training_iterations = TI 

        lg = [0,0]
        lgm = [0,0]
        Dlg ='not set'
        errcollect=[]
        for i in range(training_iterations +1):
            # Zero backprop gradients
            optimizer.zero_grad()
            # Get output from model
            output = model(train_x)
            # Calc loss and backprop derivatives
            loss = -mll(output, likelihood.transformed_targets ).sum()
            loss.backward()
            optimizer.step()
            LI = loss.item()
            
            if i%EV==0:
                print('Iter %d/%d - Loss: %.6f' % (i , training_iterations, LI, ))
                print('Kernel lengthscale:  ',  [i for i in model.covar_module.base_kernel.lengthscale.cpu().detach().numpy().flatten()]) 
                print('Second noise:        ',  [i for i in likelihood.second_noise.cpu().detach().numpy().flatten()])
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
                print('accuracy: ', 1-Err)
                errcollect.append([i,Err])
                model.train()
                likelihood.train()

        print( 'total time:  ', time.time()-t0)
        torch.save(model.state_dict(), f'{modelout}/GPmodel-{round(G/ld*100)}prc.pth')
        np.save(arr=inducing_points.cpu(), file=f'{modelout}/model-{round(G/ld*100)}prc-indc_point')
        return errcollect


erl = []

VRZ =int( sys.argv[1])
#os.mkdir(f'cycle{VRZ}')
modelout=f'cycle{VRZ}'

print(VRZ)

Tdata= pd.read_json(f'{modelout}/Tdata.json')
print(len(Tdata))

#assert len(Tdata)==VRZ

err = InvInf(Tdata,TI=50000, EV=500, XX=[5])
erl.append(err)
print()

np.save(f'{modelout}/errGP-{VRZ}', np.array(erl))
torch.cuda.empty_cache()