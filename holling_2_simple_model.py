import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt

from nfd_definitions.numerical_NFD import NFD_model, InputError


n_max = 10 # maximal number of species
# interaction matrix, 1 on the diagonal, 1/n_max else to ensure coexistence
A = np.full((n_max, n_max), 1/n_max)
np.fill_diagonal(A, 1)

# parameters for resource use
H = np.full(n_max, 0.5) # halfsaturation constant for resource uptake
K_R = 2 # carrying capacity of resource
r_R = 0.5 # regeneration speed of resource
util = 1 # conversion of resources to biomass
exp = 2 # exponent of Holling type function

def growth_model(N, A, H, exp, r_R):
    """growth model, combining LV and resource uptake model
    
    The growth of the species is the sum of LV (1-AN) and resource consumption
    
    N: Species density
    A: interaction matrix
    H: Halfsaturation constant for resource consumption
    exp: Exponent in Holling type 3 funciton
    
    returns: f(N) the gorwth of each species"""
    R_star = resource_dens(N, H, exp, r_R) # compute resource density
    # (LV) + (resource model)
    return ( 1 - A.dot(N) ) + ( R_star**exp*util/(R_star**exp + H) )

def resource_dens(N, H, exp, r_R):
    # compute resource density using brentq method
    return brentq(lambda R: r_R*(1-R/K_R) -  
                  np.sum(R**(exp-1)*N/(R**exp + H)), 0, K_R*1.1)
    
res = np.linspace(0, K_R, 101) # resource range

for r_R in [0.5, 6.5]:
    fig, ax = plt.subplots(2,2, figsize = (10,10))
    
    # compute for all species richness the NFD parameters
    for i in range(2,n_max +1):
        pars = NFD_model(growth_model, n_spec = i, 
                         args = (A[:i,:i], H[:i], exp, r_R))
        
        # compute resource density at equilibrium density
        res_dens = resource_dens(pars["N_star"][0], H[:i], exp, r_R)
        
        # plot results
        ax[1,0].plot(np.sum(pars["N_star"][0]), res_dens, 'o')
        ax[0,0].plot(res_dens, res_dens**exp/(H[0] + res_dens**exp), 'o')
        ax[0,1].plot(i, pars["ND"][0], 'o')
        ax[1,1].plot(i, pars["FD"][0], 'o')
        
    Ns = np.linspace(0,ax[1,0].get_xlim()[1]*1.1, 101)    
    ax[0,0].plot(res, res**exp/(H[0] + res**exp)) # resource consumption
    # plot resource vs spec density
    ax[1,0].plot(Ns, [resource_dens(np.full(2, N/2), H[:2], exp, r_R) for N in Ns])
    
    # add axis labels and titles
    ax[0,0].set_xlabel("Resource density")
    ax[0,0].set_ylabel("consumption rate")
    ax[0,0].set_title("Resource consumption rate")
    
    ax[1,0].set_xlabel("Total species abundance")
    ax[1,0].set_ylabel("Resource density")
    ax[1,0].set_title("Resource density at equilibrium")
    
    ax[0,1].set_title("ND vs species richness")
    ax[0,1].set_xlabel("Species richness")
    ax[0,1].set_ylabel("ND")
    
    ax[1,1].set_title("FD vs species richness")
    ax[1,1].set_xlabel("Species richness")
    ax[1,1].set_ylabel("FD")
    fig.tight_layout()
    fig.savefig("NFD_values_in_de_crease_r{}.png".format(r_R))
    