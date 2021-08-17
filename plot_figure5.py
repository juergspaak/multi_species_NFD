import numpy as np
import matplotlib.pyplot as plt
from nfd_definitions.numerical_NFD import NFD_model, InputError
import pandas as pd
from matplotlib.cm import viridis

data = pd.read_csv("evolution_LV.csv")

n_specs = np.arange(2,7)
ND,FD = np.full((2,len(n_specs), n_specs[-1]), np.nan)
A_all = np.full((len(n_specs),n_specs[-1], n_specs[-1]), np.nan)
for i,n in enumerate(n_specs):
    mu = np.empty(n)
    A = np.empty((n,n))
    data_c = data[data.Sinit == n]
    for j in range(n):
        mu[j] = data_c.intr_growth.iloc[j]
        A[j] = [float(data_c.alphas.iloc[j].split("|")[k])
                for k in range(n)]
    A_all[i, :n, :n] = A
    try:
        pars = NFD_model(lambda N: mu - A.dot(N), int(n))
        ND[i,:n] = pars["ND"]
        FD[i,:n] = pars["FD"]
    except InputError:
        pass

fig = plt.figure(figsize = (9,9))
colors = viridis(np.linspace(0,1,n_specs[-1]))[::-1]
offset = np.arange(n_specs[-1]) * np.ones(ND.shape) / (n_specs[:,None]-1)
offset = offset/len(n_specs)-0.1
###############################################################################
# niche differences versus species richness
ax_ND = fig.add_subplot(3,2,6)
for i in np.arange(n_specs[-1]):
    ax_ND.scatter(n_specs + offset[:,i], ND[:,i], s = 15, color = colors[i],
                  label = "species {}".format(i+1))
ax_ND.set_ylabel("$\mathcal{N}$", fontsize = 14)   
ax_ND.set_ylim([0.8,1]) 
ax_ND.set_yticks([0.8,0.9,1])
ax_ND.set_title("H")
ax_ND.set_xlabel("species richness")
###############################################################################
# fitness differences versus species richness
ax_FD = fig.add_subplot(3,2,2)
for i in np.arange(n_specs[-1]):
    ax_FD.scatter(n_specs + offset[:,i], FD[:,i], s = 15, color = colors[i],
                  label = "species {}".format(i+1))
ax_FD.set_ylim([0.1,-4])
ax_FD.set_yticks([0,-2,-4])
ax_FD.legend(fontsize = 10)
ax_FD.set_ylabel("$\mathcal{F}$", fontsize = 14) 
ax_FD.set_xticklabels(5*[""])
ax_FD.set_title("F")

###############################################################################
# interaction strength
offset =  np.arange(n_specs[-1])/(n_specs[:,None]-1)/len(n_specs)
offset = offset-0.1
offset = offset[...,np.newaxis] * np.ones(A_all.shape)
offset *= 1.5
ax_alpha = fig.add_subplot(3,2,4)
for n in range(n_specs[-1]):
    ax_alpha.scatter(n_specs[:,np.newaxis] + offset[:,n],
                     A_all[:,n], s = 15, color = colors[n])
    
    #ax_alpha.plot(n, np.nanmean(A_all[i]), 'ko')
ax_alpha.set_ylim([None,0.4])
ax_alpha.set_yticks([0,0.2,0.4])
ax_alpha.set_title("G")
ax_alpha.set_xticklabels(5*[""])
ax_alpha.set_ylabel("interaction strength")



x = np.linspace(-0.6,0.6,101)
for i,n in enumerate(n_specs):
    ax = fig.add_subplot(len(n_specs),2, 2*(i+1)-1)
    ax.set_title("ABCDEF"[i])
    ax.set_ylim([0,1])
    ax.set_xlim([-0.6,0.6])
    ax.set_xticks([])
    ax.set_yticks([])
    data_c = data[(data.Sinit == n)]
    for j, row in data_c.iterrows():
        ax.fill_between(x,row.intr_growth*np.exp(-(x-row.m)**2/row.w**2),
          alpha = 0.5, color = colors[row.species-1])
        if row.species == n:
            break
ax.set_xlabel("resources")
ax_label = fig.add_subplot(1,2,1, frameon = False)
ax_label.tick_params(labelcolor='none',
                     top=False, bottom=False, left=False, right=False)
ax_label.set_ylabel("resource utilisation", visible = "True")
    
  

fig.tight_layout()
fig.savefig("Figure_5.pdf") 