"""compute the NFD for real multispecies communities
Data are taken from:
    Godoy and Levine 2014 for the annual plant community
    Fort and Segure 2018 for the Lotka volterra community"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import combinations

from nfd_definitions.numerical_NFD import NFD_model, InputError

# load real LV communities        
def_lisa = pd.read_csv("definition.csv")

def_lisa = def_lisa[def_lisa.Study == "Godoy & al. 2014"]

a_ij = def_lisa.a_ij_AP.values.reshape(18,18)
lamb = def_lisa.Lambdaj.values[:18]

tot_spec = 18 # total of all species
max_spec = 4 # maximum of species for which we might have coexistence
AP_pars = {} # stores all parameters for LV systems

# adding keywords to the dictionary
# each entry is a list of length max_spec+1
# AP_pars["key"][i] contains all communities with richness i
AP_pars["matrix"] = [[] for i in range(max_spec+1)] # store the AP matrix
AP_pars["lamb"] = [[] for i in range(max_spec+1)] # store the lambda

AP_pars["origin"] = [[] for i in range(max_spec+1)] # source of the data
AP_pars["species"] = [[] for i in range(max_spec+1)] # species taken
              

names = ["Godoy"]
for n_spec in range(2, max_spec+1):
    for comb in np.array(list(combinations(range(tot_spec), n_spec))):
        AP_pars["matrix"][n_spec].append(a_ij[comb[:,None], comb])
        AP_pars["origin"][n_spec].append(names[0])
        AP_pars["species"][n_spec].append(comb)
        AP_pars["lamb"][n_spec].append(lamb[comb])

# convert everything into np.arrays
AP_pars["matrix"] = [np.array(matrix) for matrix in AP_pars["matrix"]]
AP_pars["origin"] = [np.array(origin) for origin in AP_pars["origin"]]
AP_pars["species"] = [np.array(species) for species in AP_pars["species"]]
AP_pars["lamb"] = [np.array(lamb) for lamb in AP_pars["lamb"]]

real_coms = [np.isfinite(mat).all(axis = (1,2)) 
                    for mat in AP_pars["matrix"][2:max_spec+1]]
AP_pars["matrix"] = [[],[]] + [AP_pars["matrix"][i][real_coms[i-2]] 
                            for i in range(2, max_spec+1)]
AP_pars["origin"] = [[],[]] + [AP_pars["origin"][i][real_coms[i-2]] 
                            for i in range(2, max_spec+1)]
AP_pars["species"] = [[],[]] + [AP_pars["species"][i][real_coms[i-2]] 
                            for i in range(2, max_spec+1)]
AP_pars["lamb"] = [[],[]] + [AP_pars["lamb"][i][real_coms[i-2]] 
                            for i in range(2, max_spec+1)]

AP_pars["ND"] = [[] for i in range(max_spec+1)] # to store the ND values
AP_pars["FD"] = [[] for i in range(max_spec+1)] # to store the FD values
AP_pars["feasible"] = [[] for i in range(max_spec+1)] # existence of a feasible equilibrium
AP_pars["stable"] = [[] for i in range(max_spec+1)]  # stability of equilibrium
AP_pars["c"] = [[] for i in range(max_spec+1)] # c matrix to convert densities
AP_pars["NFD_comp"] = [[] for i in range(max_spec+1)] # whether NFD can be computed
AP_pars["interaction"] = [[] for i in range(max_spec+1)] # mean interaction strength

        
def AP_model(N,A,lamb):
    return np.log(lamb/(1+A.dot(N)))

ND_AP = [[] for i in range(max_spec+1)]
FD_AP = [[] for i in range(max_spec+1)]
for n_spec in range(2, max_spec+1):
    A_all = AP_pars["matrix"][n_spec]
    lamb_all = AP_pars["lamb"][n_spec]
    equi = np.linalg.solve(A_all,lamb_all-1)
    AP_pars["feasible"][n_spec] = np.all(equi>0, axis = 1)
    #AP_pars["stable"][n_spec] = np.all(np.real(
    #        np.linalg.eigvals(-equi[:,None]*A_all))<0, axis = -1)
    AP_pars["interaction"][n_spec] = np.prod(np.abs(A_all),
           axis = (1,2))**(1/n_spec**2)
    def_pars = {"ND": np.full(n_spec, np.nan), "FD": np.full(n_spec, np.nan),
                "c": np.full((n_spec, n_spec), np.nan)}
    for i,A in enumerate(AP_pars["matrix"][n_spec]):
        try:
            if (A!=0).all():
                pars = NFD_model(AP_model, n_spec, args = (A,lamb_all[i]))
                AP_pars["NFD_comp"][n_spec].append(True)
                ND_AP[n_spec].append(pars["ND"])
                FD_AP[n_spec].append(pars["FD"])
            else:
                pars = def_pars
                AP_pars["NFD_comp"][n_spec].append(False)
        except InputError:
            pars = def_pars
            AP_pars["NFD_comp"][n_spec].append(False)
            
        AP_pars["ND"][n_spec].append(pars["ND"])
        AP_pars["FD"][n_spec].append(pars["FD"])
        AP_pars["c"][n_spec].append(pars["c"])

for key in ["ND", "FD", "c", "NFD_comp"]:
    AP_pars[key] = [np.array(entry) for entry in AP_pars[key]]