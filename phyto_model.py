"""
@author: J.W.Spaak
Investigate the NFD value for the light limited phytoplankton model.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps

from phytoplankton_communities.generate_species import gen_com,dlam
from phytoplankton_communities.I_in_functions import sun_spectrum
import phytoplankton_communities.richness_computation as rc

from nfd_definitions.numerical_NFD import NFD_model, InputError

specs = np.random.choice(np.arange(14),5, replace = True)
phi, l, k_spec, alphas, a = gen_com(specs, 2, n_com_org = 10000)
lux = 40
I_in = lux*sun_spectrum["blue sky"]

equi = rc.multispecies_equi(phi/l, k_spec, I_in = I_in)[0]
print("computed equilibria")
index = np.sum(equi>0,axis = 0)
n_com = sum(index>1)
              
equi = equi[:,index>1]
phi = phi[:,index>1]
l = l[index>1]
k_spec = k_spec[...,index>1]

def phyto_model(N,phi,l,k_spec,I_in):
    if np.sum(N)==0:
        return phi*simps(k_spec*I_in.reshape(-1,1),axis = 0,dx = dlam) - l
    tot_abs = np.sum(N*k_spec, axis = 1, keepdims = True)
    growth = phi*simps(k_spec/tot_abs*(1-np.exp(-tot_abs))*I_in.reshape(-1,1),
                       axis = 0,dx = dlam)
    return growth-l

NO_all, FD_all = np.full((2,n_com,max(index)), np.nan)
for i in range(n_com):
    phi_c = phi[equi[:,i]>0,i]
    k_spec_c = k_spec[:,equi[:,i]>0,i]
    n_spec_c = sum(equi[:,i]>0)
    # pass staring guesses for equilibrium densities
    pars = {"N_star": equi[equi[:,i]>0,i]*np.ones((n_spec_c,1))}
    try:
        pars = NFD_model(phyto_model, n_spec = int(n_spec_c),
                     args = (phi_c,l[i],k_spec_c, I_in),
                    pars = pars)
    except InputError:
        continue
    
    NO_all[i,:n_spec_c] = pars["NO"]
    FD_all[i,:n_spec_c] = pars["FD"]

# reorder NO to group all communities with 2 species, 3 species etc
richness = np.sum(np.isfinite(NO_all),axis = 1)
NO_regroup = [NO_all[richness == i,:i] for i in range(2,5)]
FD_regroup = [FD_all[richness == i,:i] for i in range(2,5)]
fig = plt.figure(figsize = (9,9))
ax_NO = fig.add_subplot(2,2,1)
ax_NO.boxplot(NO_regroup, positions = np.arange(2,5), showfliers = False)

ax_FD = fig.add_subplot(2,2,2)
ax_FD.boxplot(FD_regroup, positions = np.arange(2,5), showfliers = False)

ax_rc_2 = fig.add_subplot(2,3,4)
ax_rc_2.scatter(1-NO_regroup[0],FD_regroup[0],s = 5)

ax_rc_3 = fig.add_subplot(2,3,5)
ax_rc_3.scatter(1-NO_regroup[1],FD_regroup[1],s = 5)

ax_rc_4 = fig.add_subplot(2,3,6)
ax_rc_4.scatter(1-NO_regroup[2],FD_regroup[2],s = 5)
fig.tight_layout()