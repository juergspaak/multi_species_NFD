import numpy as np
import matplotlib.pyplot as plt

from interaction_estimation import resample_short
import LV_multi_functions as lmf
from scipy.stats import mvn


itera = 1000

r_specs = np.arange(2,7)
ND_spaak, FD_spaak = np.full((2,len(r_specs), itera, max(r_specs)), np.nan)
ND_carroll, FD_carroll = np.full((2,len(r_specs), itera), np.nan)
ND_saavedra, FD_saavedra = np.full((2, len(r_specs), itera), np.nan)
ND_chesson = np.full((len(r_specs), itera), np.nan)
FD_chesson = np.full((len(r_specs), itera, max(r_specs)), np.nan)

integral = np.empty(itera)
for i,n in enumerate(r_specs):
    A = resample_short(5*n*n*itera)
    # only competition
    A = A[A>0][:n*n*itera].reshape((itera,n,n))
    A[:,np.arange(n), np.arange(n)] = 1
    
    NFD_comp, sub_equi = lmf.find_NFD_computables(A)
    ND, FD, c, NO_ij, FD_ij, r_i = lmf.NFD_LV_multispecies(A,sub_equi)
    # niche and fitness differences according to Spaak and De Laender
    ND_spaak[i,:,:n] = ND
    FD_spaak[i,:,:n] = FD
    
    # according to Carroll et al.
    S_i = np.log(1-r_i) # sensitivities
    ND_carroll[i] = 1 - np.exp(np.mean(S_i, axis = 1))
    FD_carroll[i] = np.exp(np.std(S_i, axis = 1))
    FD_carroll[i] = (1-r_i[:,0])/(1-ND_carroll[i])
    
    # according to Chesson2003
    ND_chesson[i] = np.mean(r_i, axis = 1)
    FD_chesson[i,:,:n] = r_i - np.mean(r_i, axis = 1, keepdims = True)
    
    # compute saavedra
    for j in range(itera):
        sig = np.linalg.inv(A[j].T.dot(A[j]))
        ND_saavedra[i][j] = mvn.mvnun(np.zeros(len(A[j])),
                                      np.full(len(A[j]),np.inf),
                               np.zeros(len(A[j])), sig)[0]
    
    # compute fitness differences
    norm = np.sqrt(np.sum(A**2, axis = 1, keepdims = True))
    rc = np.mean(A/norm, axis = -1)
    FD_saavedra[i] = np.arccos(np.sum(rc, axis = 1)/np.sqrt(n)/
                               np.sqrt(np.sum(rc**2, axis = 1)))

fig , ax = plt.subplots(2,4, figsize = (11,9), sharex = True, sharey = False)


ax[0,0].boxplot([X[np.isfinite(X)] for X in ND_spaak], showfliers = False,
                positions = r_specs)
ax[1,0].boxplot([X[np.isfinite(X)] for X in FD_spaak], showfliers = False,
                positions = r_specs)    

ax[0,1].boxplot(ND_carroll.T, showfliers = False,
                positions = r_specs)
ax[1,1].boxplot(FD_carroll.T, showfliers = False,
                positions = r_specs)

ax[0,2].boxplot(ND_saavedra.T, showfliers = False,
                positions = r_specs)
ax[1,2].boxplot(FD_saavedra.T, showfliers = False,
                positions = r_specs)

ax[0,3].boxplot(ND_chesson.T, showfliers = False,
                positions = r_specs)
ax[1,3].boxplot([X[np.isfinite(X)] for X in FD_spaak], showfliers = False,
                positions = r_specs)



ax[0,0].set_ylabel("Niche differences")
ax[1,0].set_ylabel("Fitness differences")
ax[0,0].set_title("Spaak and De Laender")
ax[0,1].set_title("Carroll et al.")
ax[0,2].set_title("Saavedra et al.")
ax[0,3].set_title("Chesson")

ax[1,0].set_xlabel("Niche differences")
ax[1,1].set_xlabel("Niche differences")
ax[1,2].set_xlabel("Niche differences")
ax[1,3].set_xlabel("Niche differences")

fig.tight_layout()

fig.savefig("Figure_ap_all_methods.pdf")