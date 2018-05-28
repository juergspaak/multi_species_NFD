"""
@author: J.W. Spaak
See how parameters of the LV-model would have to change to go towards 
"neutral model". """

import matplotlib.pyplot as plt
import numpy as np

alpha = np.random.uniform(0,1,size = (2,2,10000))
alpha[0,0] = 1 # to simplify matters

# remove alphas combinations with alpha[i,i]<alpha[i,j]
good = (alpha[0,0]>alpha[0,1]) * (alpha[1,1]>alpha[1,0])
alpha = alpha[...,good]

r_i = 1-alpha[0,1]/alpha[1,1] # = f_i(0,N_j^*)


# the new value for alpha_jj and alpha_ij
alpha_p = np.linspace(0,1,1001)[:,np.newaxis]


c_all_rel_org = (alpha_p/alpha[1,1]-1)*(alpha_p/alpha[0,1]-1)
c_tot = np.amin(c_all_rel_org,axis = 0)
c_rel_org = c_tot.copy()
c_rel_org[c_tot<np.percentile(c_tot, 5)] = np.percentile(c_tot,5)

c_all_abs = (alpha_p-alpha[1,1])*(alpha_p-alpha[0,1])
c_tot = np.amin(c_all_abs,axis = 0)
c_abs = c_tot.copy()
c_abs[c_tot<np.percentile(c_tot, 5)] = np.percentile(c_tot,5)

c_all_rel_new = (alpha_p-alpha[1,1])*(alpha_p-alpha[0,1])/(alpha_p**2)
c_tot = np.amin(c_all_rel_new,axis = 0)
c_rel_new = c_tot.copy()
c_rel_new[c_tot<np.percentile(c_tot, 5)] = np.percentile(c_tot,5)

# compute ND and FD
f_0 = 1 # = f_i(0,0)
r_i = 1-alpha[0,1]/alpha[1,1] # = f_i(0,N_j^*)
f_N = 1-alpha[0,0]/alpha[1,1] # = f_i(N_j^*,0)
r_i_cut = r_i.copy()
cut = 1
r_i_cut[r_i<np.percentile(r_i,cut)] = np.percentile(r_i,cut)
    
ND_spaak4 = (r_i - f_N)/(f_0-f_N)
FD_spaak4 = -f_N/f_0

# check equality
if not(r_i-(ND_spaak4-FD_spaak4+ND_spaak4*FD_spaak4)<1e-10).all():
    print("Error spaak4")
    
fig, ax = plt.subplots(2,2, figsize = (9,9),sharex = True, sharey = True)

ND = np.linspace(0,1,100)

# plt with r_i
ax[0,0].scatter(ND_spaak4, FD_spaak4,c = np.abs(r_i_cut),linewidth = 0)
ax[0,0].plot(ND,ND/(1-ND))


# plot with c_tot
im = ax[0,1].scatter(ND_spaak4, FD_spaak4,c = c_abs,linewidth = 0)
ax[0,1].plot(ND,ND/(1-ND))

# plot with c_tot
im = ax[1,0].scatter(ND_spaak4, FD_spaak4,c = c_rel_new,linewidth = 0)
ax[1,0].plot(ND,ND/(1-ND))

# plot with c_tot
im = ax[1,1].scatter(ND_spaak4, FD_spaak4,c = c_rel_org,linewidth = 0)
ax[1,1].plot(ND,ND/(1-ND))

ax[1,1].set_ylim([-1,10])

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)