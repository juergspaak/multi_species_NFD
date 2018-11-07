"""level plots of the different definitions for fitness and niche
differences for the annual plant model"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from scipy.optimize import brentq

rep = 5000

A11 = 1
A22 = 1
A21 = 0.7*np.ones(rep)

lamb1 = 3
lamb2 = 6

A12 = np.linspace(-A22/(lamb2-1)+1e-3,4, rep)

ND = {}
FD = {}
ND_bound = {}
FD_bound = {}

# Definition according to Adler et al. 2007
key = "Adler et al."
ND[key] = np.log(lamb2/(1+A12/A22*(lamb2-1)))
FD[key] = np.log(lamb1/lamb2*np.ones(A12.shape))

# Definition according to Godoy et al.
key = "Godoy et al."
ND[key] = 1-np.sqrt(A12*A21/(A22*A11))
FD[key] = (lamb1-1)/(lamb2-1)*np.sqrt(A21*A22/(A11*A12))

# Definitions by experimental methods
# zero growth rate
f_i_0 = np.log([lamb1,lamb2])

# invasion growth rate
r_i = np.array([np.log(lamb1/(1+A12/A22*(lamb2-1))),
                np.log(lamb2/(1+A21/A11*(lamb1-1)))])

# Definition according to Carroll et al.
S_i = (f_i_0.reshape(-1,1)-r_i)/f_i_0.reshape(-1,1)
key = "Carroll et al."
ND[key] = 1-np.exp(np.mean(np.log(S_i),axis = 0))
FD[key] = np.exp(np.var(np.log(S_i),axis = 0))

# Definition according to Zhao et al.
key = "Zhao et al."
ND[key] = 1+np.sum(r_i, axis = 0)
FD[key] = (np.log((lamb1-1)/A11)-np.log((lamb2-1)/A22))*np.ones(A12.shape)
    
# Definition according to Chesson
key = "Chesson"
phi_i = np.array([A11/lamb1,A22/lamb2])
ND[key] = 0.5*np.sum(r_i/phi_i.reshape(-1,1),axis = 0)
FD[key] = (r_i/phi_i.reshape(-1,1)-ND[key])[0]

# Definition according to Bimler
key = "Bimler et al."

# change to alpha' according to Bimler et al
A11_ = A11/(lamb1-1)
A12_ = A12/(lamb1-1)
A21_ = A21/(lamb2-1)
A22_ = A22/(lamb2-1)

ND[key] = 1 - np.exp((A12_-A11_)+(A21_-A22_))
FD[key] = np.exp(-(A12_+A11_)+(A21_+A22_))

# Definition accoding to Spaak


c = np.empty(rep)
denom = np.array([np.log(1+A12/A22*(lamb2-1)),
                  np.log(1+A21/A11*(lamb1-1))])
fac = np.array([A11/A22*(lamb2-1),A22/A11*(lamb1-1)])

def NO_spaak(c,denom):
    return np.array([denom[0]/np.log(1+c*fac[0]),denom[1]/np.log(1+fac[1]/c)])
def NO_equate(c,denom):
    NO = np.abs(NO_spaak(c,denom))
    return NO[0]-NO[1]

for l in range(rep):
    try:
        c[l] = brentq(NO_equate,0,1e3, args = (denom[:,l]))
    except ValueError:
        c[l] = brentq(NO_equate,0,1e10, args = (denom[:,l]))
key = "Spaak & deLaender"   
ND[key] = 1-NO_spaak(c, denom)[0]
FD[key] = np.log(lamb1/(1+c*A11/A22*(lamb2-1)))/np.log(lamb1)
#FD[key] = np.log(lamb2/(1+1/c*A22/A11*(lamb1-1)))/np.log(lamb2)

fig, ax = plt.subplots(ncols = 2, figsize = (12,9), sharex = True)
keys = ["Spaak & deLaender","Chesson","Carroll et al.", "Zhao et al.","Godoy et al.",
        "Adler et al.",   "Bimler et al."]

colors = {"Chesson":        "green",
          "Carroll et al.": "blue",
          "Zhao et al.":    "purple",
          "Bimler et al.":  "lime",
          "Spaak & deLaender":   "red",
          "Adler et al.":   "cyan",
          "Godoy et al.":   "orange"}
ND_range = [-0.5,1.5]

rect_facilitation = patches.Rectangle([A12[0],1],-A12[0],ND_range[1]-1,
                                      alpha = 0.5, color = "green")
rect_norm = patches.Rectangle([0,0],A11*A22/A21[0],1, alpha = 0.5)
rect_comp = patches.Rectangle([A11*A22/A21[0],0],A12[-1], ND_range[0]
                              , alpha = 0.5, color = "red")
ax[0].add_patch(rect_norm)
ax[0].add_patch(rect_facilitation)
ax[0].add_patch(rect_comp)
          
# plot NFD parameters          
for key in keys:
    ax[0].plot(A12, ND[key], label = key, linewidth = 2, alpha = 1, 
             color = colors[key])
    ax[1].plot(A12, FD[key], label = key, linewidth = 2, alpha = 1, 
             color = colors[key])

# add black dots
ax[0].plot(0,1, 'o', color = "black")
ax[0].plot(A11*A22/A21[0],0, '^', color = "black")


# layout
ax[1].legend(bbox_to_anchor=(1, 0.85))

# axis limits
ax[0].set_xlim(A12[[0,-1]])
ax[0].set_xticks([0,1,2,3,4])
ax[0].set_ylim(*ND_range)
ax[1].set_ylim(-1,2)
ax[1].set_yticks([-1,0,1,2])

# draw axis lines
ax[0].axhline(y=0, color='black', linestyle=':')
ax[0].axvline(x=0, color='black', linestyle=':')
ax[1].axhline(y=0, color='black', linestyle=':')
ax[1].axvline(x=0, color='black', linestyle=':')

# axis labels
ax[0].set_xlabel(r'Interspecific competition ($\alpha_{12}$)', fontsize = 16)
ax[1].set_xlabel(r'Interspecific competition ($\alpha_{12}$)', fontsize = 16)
ax[0].set_ylabel(r'Niche difference $(\mathcal{N})$', fontsize = 16)
ax[1].set_ylabel(r'Fitness difference $(\mathcal{F})$', fontsize = 16)
ax[0].set_title("A")
ax[1].set_title("B")


fig.savefig("NFD in annual plants.pdf")