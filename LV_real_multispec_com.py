"""
@author: J.W.Spaak
compute the NFD for real multispecies communities
Data are taken from real LV communities"""

import numpy as np
import pandas as pd
from itertools import combinations
from collections import Counter
import warnings

import LV_multi_functions as lmf

# load real LV communities        
LV_multi_spec = pd.read_csv("LV_multispec.csv", usecols = np.arange(14))

# load all matrices
matrices = {}
ind = np.where(np.isfinite(LV_multi_spec.n_spec))[0]
max_spec = int(np.nanmax(LV_multi_spec.n_spec))
interaction_index = ["A_{}".format(i) for i in range(1,max_spec + 1)]
for i in ind:
    n_spec = int(LV_multi_spec.n_spec[i])
    matrices[LV_multi_spec.Source[i]] = LV_multi_spec.loc[i+1:i+n_spec,
            interaction_index[:n_spec]].values

LV_pars = {} # stores all parameters for LV systems

# to know community types
Counter(LV_multi_spec.Community_type)
Counter(LV_multi_spec.Experiment_type)

# adding keywords to the dictionary
# each entry is a list of length max_spec+1
# LV_pars["key"][i] contains all communities with richness i
LV_pars["matrix"] = [[] for i in range(max_spec+1)] # store the LV matrix

LV_pars["origin"] = [[] for i in range(max_spec+1)] # source of the data
LV_pars["species"] = [[] for i in range(max_spec+1)] # species taken

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
LV_pars["ND_no_indir"] = (max_spec+1)*[[]] # to store the ND values
LV_pars["FD_no_indir"] = (max_spec+1)*[[]] # to store the FD values
LV_pars["c"] = (max_spec+1)*[[]] # c matrix to convert densities
LV_pars["NO_ij"] = (max_spec+1)*[[]] # c matrix to convert densities
LV_pars["FD_ij"] = (max_spec+1)*[[]] # c matrix to convert densities
LV_pars["sub_equi"] = (max_spec+1)*[[]] # c matrix to convert densities
LV_pars["A_NFD"] = (max_spec+1)*[[]] # interaction matrix of NFD_comp
LV_pars["NFD_comp"] = (max_spec+1)*[[]] # whether NFD can be computed
# geometricalmean interaction strength of the offdiagonal entries (diag = 1)
LV_pars["interaction_geom"] = (max_spec+1)*[[]]
# arithmetic mean interaction strength of the offdiagonal entries (diag = 1)
LV_pars["interaction_artm"] = (max_spec+1)*[[]]
LV_pars["interaction_medi"] = (max_spec+1)*[[]]

# values concerning coexistence
LV_pars["invasion_growth"] = (max_spec+1)*[np.array([])] # the invasion growth rates
LV_pars["coex_invasion"] = (max_spec+1)*[np.array([])] # do all species have r_i>0
LV_pars["real_coex"] = (max_spec+1)*[np.array([])] # is there a stable steady state?
        
for n_spec in range(2,7):
    A_n = LV_pars["matrix"][n_spec]
    
    # to compute average interaction strength remove intraspecific interaction
    B_n = A_n.copy()
    B_n[:,np.arange(n_spec), np.arange(n_spec)] = np.nan
    LV_pars["interaction_geom"][n_spec] = np.nanprod(np.abs(B_n),
           axis = (1,2))**(1/(n_spec*(n_spec-1)))
    LV_pars["interaction_artm"][n_spec] = (np.nansum(B_n,
           axis = (1,2))) / (n_spec*(n_spec-1))
    # compute the median of 
    LV_pars["interaction_medi"][n_spec] = np.nanmedian(B_n, axis = (1,2))
    
    NFD_comp, sub_equi = lmf.find_NFD_computables(A_n)
    A_comp = A_n[NFD_comp]
    sub_equi = sub_equi[NFD_comp]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ND, FD, c, NO_ij, FD_ij, r_i = lmf.NFD_LV_multispecies(A_comp,sub_equi)
    LV_pars["ND"][n_spec] = ND
    LV_pars["FD"][n_spec] = FD
    LV_pars["A_NFD"][n_spec] = A_comp
    LV_pars["NFD_comp"][n_spec] = NFD_comp
    LV_pars["c"][n_spec] = c
    LV_pars["NO_ij"][n_spec] = NO_ij
    LV_pars["FD_ij"][n_spec] = FD_ij
    LV_pars["sub_equi"][n_spec] = sub_equi
    LV_pars["coex_invasion"][n_spec] = np.all(r_i>0, axis = 1)
    LV_pars["invasion_growth"][n_spec] = r_i
    
    # compute which communities have a stable equilibrium
    equi = np.linalg.solve(A_n,np.ones(A_n.shape[:2]))
    # compute eigenvalues at equilibrium
    jacobi = np.linalg.eig(-equi[...,np.newaxis]* A_n)[0]
    LV_pars["real_coex"][n_spec] = np.logical_and(np.all(equi>0,axis = 1),
           np.all(np.real(jacobi)<0, axis = 1))
    # compute which communities are stable
    
    # compute effects in absence of indirect effects
    sub_equi2 = np.ones(sub_equi.shape)
    sub_equi2[:,np.arange(n_spec),np.arange(n_spec)] = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ND2, FD2, c, NO_ij, FD_ij, r_i = lmf.NFD_LV_multispecies(A_comp,
                                                    sub_equi2, check = False)
    LV_pars["ND_no_indir"][n_spec] = ND2
    LV_pars["FD_no_indir"][n_spec] = FD2
    
# create dataframe for testing differences in distributions of ND
ls = []
for i in range(max_spec):
    if np.any(LV_pars["NFD_comp"][i]):
        ND = LV_pars["ND"][i].flatten()
        FD = LV_pars["FD"][i].flatten()
        n_spec = i*np.ones(ND.size)
        origin = LV_pars["origin"][i][LV_pars["NFD_comp"][i]]
        origin = np.repeat(origin, i)
        ls.append(pd.DataFrame({"ND": ND, "FD":FD,
                                "n_spec": n_spec, "origin": origin}))
df = pd.concat(ls)
df.to_csv("NFD_real_LV.csv")
    