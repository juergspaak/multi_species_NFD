"""level plots of the different definitions for fitness and niche
differences for the annual plant model"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

rep = 50

A11 = 1.2
A22 = 1
array = np.linspace(1e-2,1,rep)
A12,A21 = np.meshgrid(array,array)

lamb1 = 15
lamb2 = 10

ND = {}
FD = {}
ND_bound = {}
FD_bound = {}

# Definition according to Adler et al. 2007
key = "Adler et al."
ND[key] = lamb2/(1+A12/A22*(lamb2-1))
FD[key] = lamb1/lamb2*np.ones(A12.shape)
ND_bound[key] = "large"
FD_bound[key] = "norm"

# Definition according to Godoy et al.
key = "Godoy et al."
ND[key] = 1-np.sqrt(A12*A21/(A22*A11))
FD[key] = (lamb1-1)/(lamb2-1)*np.sqrt(A21*A22/(A11*A12))
ND_bound[key] = "norm"
FD_bound[key] = "norm"


# Definitions by experimental methods
# zero growth rate
f_i_0 = np.log([lamb1,lamb2])

# invasion growth rate
r_i = np.array([np.log(lamb1/(1+A12/A22*(lamb2-1))),
                np.log(lamb2/(1+A21/A11*(lamb1-1)))])

# Definition according to Carroll et al.
S_i = (f_i_0.reshape(-1,1,1)-r_i)/f_i_0.reshape(-1,1,1)
key = "Carroll et al."
ND[key] = 1-np.exp(np.mean(np.log(S_i),axis = 0))
FD[key] = np.exp(np.var(np.log(S_i),axis = 0))
ND_bound[key] = "norm"
FD_bound[key] = "large"

# Definition according to Zhao et al.
key = "Zhao et al."
ND[key] = np.sum(r_i, axis = 0)
FD[key] = (np.log((lamb1-1)/A11)-np.log((lamb2-1)/A22))*np.ones(A12.shape)
ND_bound[key] = "large"
FD_bound[key] = "norm"

# Definition according to Chesson
key = "Chesson"
phi_i = np.array([A11/lamb1,A22/lamb2])
ND[key] = 0.5*np.sum(r_i/phi_i.reshape(-1,1,1),axis = 0)
FD[key] = (r_i/phi_i.reshape(-1,1,1)-ND[key])[0]
ND_bound[key] = "large"
FD_bound[key] = "large"

# Definition according to Bimler
key = "Bimler et al."

ND[key] = 1 - np.exp((A12-A11)/np.log(lamb1)+(A21-A22)/np.log(lamb2))
FD[key] = np.exp(-(A12+A11)/np.log(lamb1)+(A21+A22)/np.log(lamb2))
ND_bound[key] = "norm"
FD_bound[key] = "norm"

# Definition accoding to Spaak


c = np.empty((rep,rep))
denom = np.array([np.log(1+A12/A22*(lamb2-1)),
                  np.log(1+A21/A11*(lamb1-1))])
fac = np.array([A11/A22*(lamb2-1),A22/A11*(lamb1-1)])

def NO_spaak(c,denom):
    return np.array([denom[0]/np.log(1+c*fac[0]),denom[1]/np.log(1+fac[1]/c)])
def NO_equate(c,denom):
    NO = np.abs(NO_spaak(c,denom))
    return NO[0]-NO[1]

for k in range(rep):
    for l in range(rep):
        try:
            c[l,k] = brentq(NO_equate,0,1e3, args = (denom[:,l,k]))
        except ValueError:
            c[l,k] = brentq(NO_equate,0,1e10, args = (denom[:,l,k]))
key = "Spaak et al."   
ND[key] = 1-NO_spaak(c, denom)[0]
FD[key] = np.log(lamb1/(1+c*A11/A22*(lamb2-1)))/np.log(lamb1)
ND_bound[key] = "norm"
FD_bound[key] = "norm"


fig, ax = plt.subplots(3,2, figsize = (9,9), sharex = True, sharey = True)
keys = ["Chesson","Carroll et al.", "Zhao et al.","Godoy et al.",
        "Adler et al.",   "Bimler et al."]

for k,key in enumerate(keys):
    axc = ax.flatten()[k]
    
    if ND_bound[key]=="norm":
        vmin,vmax,cmap = 0,1,None
    else:
        vmin, vmax , cmap = 0, 10, "cool"

    cl = axc.imshow(ND[key], origin  = "lower", vmin = vmin, vmax = vmax,
                    cmap = cmap, extent = [0,1,0,1], aspect = "auto",
                    interpolation = "bilinear")
    plt.colorbar(cl, ax = axc)
        
    axc.set_title(key)

fs = 14    
ax[0,0].set_ylabel(r'$\alpha_{12}$', fontsize = fs)
ax[1,0].set_ylabel(r'$\alpha_{12}$', fontsize = fs)
ax[2,0].set_ylabel(r'$\alpha_{12}$', fontsize = fs)

ax[2,0].set_xlabel(r'$\alpha_{21}$', fontsize = fs)
ax[2,1].set_xlabel(r'$\alpha_{21}$', fontsize = fs)

fig.savefig("ND,Annualplants_colorplot.pdf")

fig, ax = plt.subplots(3,2, figsize = (9,9), sharex = True, sharey = True)
plt.axis([0,1,0,1])
for k,key in enumerate(keys):
    axc = ax.flatten()[k]
    
    if FD_bound[key]=="norm":
        vmin,vmax,cmap = -0.5,2,None
    else:
        vmin, vmax , cmap = 0, 10, "cool"

    cl = axc.imshow(FD[key], origin  = "lower", vmin = vmin, vmax = vmax,
                    cmap = cmap, extent = [0,1,0,1], aspect = "auto", 
                    interpolation = "bilinear")
    plt.colorbar(cl, ax = axc)
        
    axc.set_title(key)

fs = 14    
ax[0,0].set_ylabel(r'$\alpha_{12}$', fontsize = fs)
ax[1,0].set_ylabel(r'$\alpha_{12}$', fontsize = fs)
ax[2,0].set_ylabel(r'$\alpha_{12}$', fontsize = fs)

ax[2,0].set_xlabel(r'$\alpha_{21}$', fontsize = fs)
ax[2,1].set_xlabel(r'$\alpha_{21}$', fontsize = fs)

fig.savefig("FD,Annualplants_colorplot.pdf")