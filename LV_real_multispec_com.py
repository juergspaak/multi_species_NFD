"""compute the NFD for real multispecies communities
Data are taken from:
    Godoy and Levine 2014 for the annual plant community
    Fort and Segure 2018 for the Lotka volterra community"""

import numpy as np
import pandas as pd
from itertools import combinations
from nfd_definitions.numerical_NFD import InputError, NFD_model

import LV_multi_functions as lmf

# load real LV communities        
fort_2_spec = pd.read_csv("LV_param_fort2018.csv")
LV_multi_spec = pd.read_csv("LV_multispec.csv")

# load all matrices
matrices = {}
ind = np.where(np.isfinite(LV_multi_spec.n_spec))[0]
max_spec = int(np.nanmax(LV_multi_spec.n_spec))
for i in ind:
    n_spec = int(LV_multi_spec.n_spec[i])
    matrices[LV_multi_spec.Source[i]] = LV_multi_spec.iloc[i+1:i+n_spec+1,
            2:2+n_spec].values

LV_pars = {} # stores all parameters for LV systems

# adding keywords to the dictionary
# each entry is a list of length max_spec+1
# LV_pars["key"][i] contains all communities with richness i
LV_pars["matrix"] = [[] for i in range(max_spec+1)] # store the LV matrix

LV_pars["origin"] = [[] for i in range(max_spec+1)] # source of the data
LV_pars["species"] = [[] for i in range(max_spec+1)] # species taken
        
# create matrices from the data
A_fort = np.ones((len(fort_2_spec),2,2))
A_fort[:,0,1] = fort_2_spec.a_ij
A_fort[:,1,0] = fort_2_spec.a_ji
LV_pars["matrix"][2] = list(A_fort)
LV_pars["origin"][2] = len(A_fort)*["Fort et al. 2 spec"]
LV_pars["species"][2] = [[i,i] for i in range(len(A_fort))]

for key in matrices.keys():
    multi = matrices[key]
    cur_spec = len(multi)
    for n_spec in range(2, cur_spec+1):
        for comb in np.array(list(combinations(range(cur_spec), n_spec))):
            LV_pars["matrix"][n_spec].append(multi[comb[:,None], comb])
            LV_pars["origin"][n_spec].append(key)
            LV_pars["species"][n_spec].append(comb)

# convert everything into np.arrays
LV_pars["matrix"] = [np.array(matrix) for matrix in LV_pars["matrix"]]
LV_pars["origin"] = [np.array(origin) for origin in LV_pars["origin"]]
LV_pars["species"] = [np.array(species) for species in LV_pars["species"]]

real_coms = [np.isfinite(mat).all(axis = (1,2)) 
                    for mat in LV_pars["matrix"][2:]]
LV_pars["matrix"] = [[],[]] + [LV_pars["matrix"][i][real_coms[i-2]] 
                            for i in range(2, max_spec+1)]
LV_pars["origin"] = [[],[]] + [LV_pars["origin"][i][real_coms[i-2]] 
                            for i in range(2, max_spec+1)]
LV_pars["species"] = [[],[]] + [LV_pars["species"][i][real_coms[i-2]] 
                            for i in range(2, max_spec+1)]

LV_pars["ND"] = (max_spec+1)*[[]] # to store the ND values
LV_pars["FD"] = (max_spec+1)*[[]] # to store the FD values
LV_pars["c"] = (max_spec+1)*[[]] # c matrix to convert densities
LV_pars["A_NFD"] = (max_spec+1)*[[]] # interaction matrix of NFD_comp
LV_pars["NFD_comp"] = (max_spec+1)*[[]] # whether NFD can be computed
# geometricalmean interaction strength of the offdiagonal entries (diag = 1)
LV_pars["interaction_geom"] = (max_spec+1)*[[]]
# arithmetic mean interaction strength of the offdiagonal entries (diag = 1)
LV_pars["interaction_artm"] = (max_spec+1)*[[]]

def LV_model(N,A):
    return 1-A.dot(N)
        
ND_LV = [[] for i in range(max_spec+1)]
FD_LV = [[] for i in range(max_spec+1)]
comp = [[] for i in range(max_spec+1)]
for n_spec in range(2,7):
    print(n_spec)
    A_n = LV_pars["matrix"][n_spec]
    LV_pars["interaction_geom"][n_spec] = np.prod(np.abs(A_n),
           axis = (1,2))**(1/(n_spec**2))
    LV_pars["interaction_artm"][n_spec] = (np.sum(A_n,
           axis = (1,2)) - n_spec) / (n_spec**2-n_spec)
    
    NFD_comp, sub_equi = lmf.find_NFD_computables(A_n)
    A_comp = A_n[NFD_comp]
    sub_equi = sub_equi[NFD_comp]
    ND, FD = lmf.NFD_LV_multispecies(A_comp,sub_equi)
    LV_pars["ND"][n_spec] = ND
    LV_pars["FD"][n_spec] = FD
    LV_pars["A_NFD"][n_spec] = A_comp
    LV_pars["NFD_comp"][n_spec] = NFD_comp
    
    
    # check whether results are equivalent to the ones from NFD_model
    for i,A_ in enumerate(LV_pars["matrix"][n_spec]):
        try:
            pars = {}
            pars = NFD_model(LV_model, n_spec, args = (A_,), pars = pars)
            ND_LV[n_spec].append(pars["ND"])
            FD_LV[n_spec].append(pars["FD"])
            comp[n_spec].append(True)
            if not LV_pars["NFD_comp"][n_spec][i]:
                print(i, LV_pars["NFD_comp"][n_spec][i], True)
                raise ValueError
        except InputError:
            comp[n_spec].append(False)
            if LV_pars["NFD_comp"][n_spec][i]:
                print(i, LV_pars["NFD_comp"][n_spec][i], False)
                raise
            

ND_LV = LV_pars["ND"]

FD_LV = LV_pars["FD"]
"""
# check whether there is a community in which ND<0 and has coexistence    
for i in range(2, max_spec + 1):
    ND, FD = LV_pars["ND"][i], LV_pars["FD"][i]
    coex = np.all(ND+FD-ND*FD>0, axis = 1)
    try:
        j = np.argmin(np.amin(ND[coex], axis = 1))
        print(i, ND[coex][j], FD[coex][j],(ND+FD-ND*FD)[coex][j] )
    except ValueError:
        continue"""