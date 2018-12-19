import numpy as np
import matplotlib.pyplot as plt

from annual_plants_multispecies import NFD_annual_plants

richness = np.arange(2,11)
NO_all, FD_all = np.empty((2,3,len(richness),1000,max(richness)))
for i in richness:
    if i ==3:
        NO_all[:,i-2] = np.nan
        FD_all[:,i-2] = np.nan
        continue
    file = np.load("NFD_values,richness {}.npz".format(i))
    NO_all[:,i-2,:,:i] = file["NO"].copy()
    FD_all[:,i-2,:,:i] = file["FD"].copy()
    print(np.sum(np.isfinite(file["NO"][...,0])),
                             np.sum(np.isfinite(NO_all[:,i-2,:,0])),i)

fs = 14    
fig = plt.figure(figsize = (11,11))

n_models = 3
model_names = ["LV", "holling 1", "holling 2"]
for m in range(n_models):
    
    if m == 0:
        ax_NO = fig.add_subplot(6,n_models,m+1)
        ax_FD = fig.add_subplot(6,n_models,m+n_models+1)
        ax_coex = fig.add_subplot(3,n_models,n_models + m + 1)
    else:
        ax_NO = fig.add_subplot(6,n_models,m+1, sharey = ax_NO)
        ax_FD = fig.add_subplot(6,n_models,m+n_models+1, sharey = ax_FD)
        ax_coex = fig.add_subplot(3,n_models,n_models + m + 1, 
                                  sharey = ax_coex, sharex = ax_coex)
    ax_NO.boxplot([NO[np.isfinite(NO)] for NO in NO_all[m,...,0]],
                  positions = richness,showfliers = False)

    
    ax_FD.boxplot([FD[np.isfinite(FD)] for FD in FD_all[m,...,0]],
                  positions = richness, showfliers = False)
    ax_FD.set_xlabel("number of species")
    ax_NO.set_title(model_names[m], fontsize = fs)
    ax_n_com = ax_NO.twinx()
    ax_n_com.plot(richness,np.sum(np.isfinite(NO_all[m,...,0]),axis = -1)
                ,alpha = 0.5)
    ax_n_com.set_ylim(0,1000)
    
    
    
    x = np.linspace(0,1,1000)
    im = ax_coex.scatter(1-NO_all[m], FD_all[m], s = 5,linewidth = 0)
    #fig.colorbar(im,ax = ax_coex)
    ax_coex.plot(x,-x/(1-x), color = "black")
    ax_coex.set_xlim(np.nanpercentile(1-NO_all,[1,99]))
    ax_coex.set_ylim(np.nanpercentile(FD_all,[1,99]))
    ax_coex.invert_yaxis()
    
    ax_coex.set_xlabel(r"$\mathcal{N}$", fontsize = fs)
    
    if m == 0:
        ax_NO.set_ylabel(r"$\mathcal{NO}$", fontsize = fs)     
        ax_FD.set_ylabel(r"$\mathcal{F}$", fontsize = fs)     
        ax_coex.set_ylabel(r"$-\mathcal{F}$", fontsize = fs)
fig.tight_layout()
fig.savefig("holling types.pdf")