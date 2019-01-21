import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import combinations

from nfd_definitions.numerical_NFD import NFD_model, InputError
"""
def_lisa = pd.read_csv("definition.csv")

def_lisa = def_lisa[def_lisa.Study == "Godoy & al. 2014"]

a_ij = def_lisa.a_ij_AP.values.reshape(18,18)
lamb = def_lisa.Lambdaj.values[:18]

species = len(lamb)
comb_2_spec = np.array(list(combinations(range(species),2)))
comb_3_spec = np.array(list(combinations(range(species),3)))
comb_4_spec = np.array(list(combinations(range(species),4)))

def annual_plant(N,A,lamb):
    return np.log(lamb/(1+A.dot(N)))

def equi(A,l):
    N_star = np.full((len(l), len(l)),0, dtype = float)
    index = np.arange(len(l))
    for i in range(len(l)):
        # to remove species i
        inds = index[index != i]
        # compute subcommunities equilibrium
        N_star[i,inds] = np.linalg.solve(A[inds[:,np.newaxis],inds],
                        l[inds]-1)
    return N_star
    
ND_2, FD_2 = np.full((2,len(comb_2_spec),2), np.nan)
coex = np.full(a_ij.shape, False, dtype = bool)
for i,comb in enumerate(comb_2_spec):
    print(comb)
    A, l = a_ij[comb[:,None], comb], lamb[comb]
    
    if np.all(np.isfinite(a_ij[comb[:,None], comb])):
        pars_pre = {"N_star": equi(A,l)}
        pars = NFD_model(annual_plant, n_spec = 2,
                         args = (A,l), pars = pars_pre)
        ND_2[i] = pars["ND"]
        FD_2[i] = pars["FD"]
        coex[comb[0], comb[1]] = (pars["r_i"]>0).all()
        
    
ND_3, FD_3 = np.full((2,len(comb_3_spec),3), np.nan)
for i,comb in enumerate(comb_3_spec):
    #print(comb)
    A, l = a_ij[comb[:,None], comb], lamb[comb]
    pars_pre = equi(A,l)
    if np.all(np.isfinite(a_ij[comb[:,None], comb])):
        pars_pre = {"N_star": equi(A,l)}
        try:
            pars = NFD_model(annual_plant,n_spec = 3, args = (A,l),
                         pars = pars_pre)
        except InputError:
            continue
        print(comb)
        ND_3[i] = pars["ND"]
        FD_3[i] = pars["FD"]"""
        
LV_2_spec = pd.read_csv("LV_param_fort2018.csv")

def LV_model(N,A):
    return 1-A.dot(N)

ND_2_LV, FD_2_LV = np.full((2,len(LV_2_spec),2), np.nan)
A = np.ones((len(LV_2_spec),2,2))
A[:,0,1] = LV_2_spec.a_ij
A[:,1,0] = LV_2_spec.a_ji
for i in range(len(LV_2_spec)):
    if np.isfinite(A[i]).all() and (A[i]!=0).all():
        pars = NFD_model(LV_model, args = (A[i],))
        ND_2_LV[i] = pars["ND"]
        FD_2_LV[i] = pars["FD"]
        
LV_multi_spec = pd.read_csv("LV_param_fort2018_multispecies.csv")
huisman = LV_multi_spec.iloc[:4,1:5].values
neill = LV_multi_spec.iloc[5:9,1:5].values
vandermeer = LV_multi_spec.iloc[10:14,1:5].values
picasso = LV_multi_spec.iloc[15:,1:].values

ND = [[] for i in range(7)]
FD = [[] for i in range(7)]
for A_multi in [huisman, neill, vandermeer, picasso]:
    n_max = len(A_multi)
    for n_spec in range(2,n_max):
        for comb in np.array(list(combinations(range(n_max), n_spec))):
            A = A_multi[comb[:,None], comb]
            if np.isfinite(A).all() and (A!=0).all():
                try:
                    pars = NFD_model(LV_model, n_spec = n_spec, args = (A,))
                    ND[n_spec].append(pars["ND"])
                    FD[n_spec].append(pars["FD"])
                except InputError:
                    pass

for i in range(len(ND)):
    plt.scatter(ND[i], -np.array(FD[i]), label = i)
x = np.linspace(-2,2)
plt.plot(x, x/(1-x))
plt.legend()
plt.ylim(-3,1)

plt.figure()
plt.boxplot(ND)
plt.figure()
plt.boxplot(FD)