"""level plots of the different definitions for fitness and niche
differences for the annual plant model"""

import matplotlib.pyplot as plt
import numpy as np

alpha = np.random.uniform(0,1,size = (2,2,100000))
# remove alphas combinations with alpha[i,i]<alpha[i,j]

good = (alpha[0,0]>alpha[0,1]) * (alpha[1,1]>alpha[1,0])
alpha = alpha[...,good]

# define growth rates
f_0 = 1 # = f_i(0,0)
r_i = 1-alpha[0,1]/alpha[1,1] # = f_i(0,N_j^*)
f_N = 1-alpha[0,0]/alpha[1,1] # = f_i(N_j^*,0)
r_i_cut = r_i.copy()
cut = 1
r_i_cut[r_i<np.percentile(r_i,cut)] = np.percentile(r_i,cut)

ND_chesson = 1 - np.sqrt(alpha[0,1]*alpha[1,0]/(alpha[1,1]*alpha[0,0]))
FD_chesson = np.sqrt(alpha[0,1]*alpha[0,0]/(alpha[1,0]*alpha[1,1]))

ND_spaak3 = (r_i - f_N)/(f_0-f_N)
FD_spaak3 = -f_N/(f_0-f_N)*(1-r_i/f_0)

# check equality
if not(r_i-(ND_spaak3-FD_spaak3)<1e-10).all():
    print("Error spaak3")
    
ND_spaak4 = (r_i - f_N)/(f_0-f_N)
FD_spaak4 = -f_N/f_0

# check equality
if not(r_i-(ND_spaak4-FD_spaak4+ND_spaak4*FD_spaak4)<1e-10).all():
    print("Error spaak4")
    
fig, ax = plt.subplots(2,2, figsize = (9,9),sharex = True, sharey = True)

ND = np.linspace(0,1,100)

vmin,vmax = min(r_i), max(r_i)
im = ax[0,0].scatter(ND_chesson, FD_chesson,c = r_i_cut,linewidth = 0)
ax[0,0].plot(ND,1/(1-ND))
ax[1,0].scatter(ND_spaak3, FD_spaak3,c = r_i_cut,linewidth = 0)
ax[1,0].plot(ND,ND)
ax[1,1].scatter(ND_spaak4, FD_spaak4,c = r_i_cut,linewidth = 0)
ax[1,1].plot(ND,ND/(1-ND))
ax[1,1].set_ylim([-1,10])

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)

ax[0,0].set_title("Chesson")
ax[1,0].set_title("Spaak3")
ax[1,1].set_title("Spaak4")