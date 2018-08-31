"""
@author: J.W.Spaak
Numerically compute ND and FD
"""

import numpy as np
from scipy.optimize import brentq, fsolve

def find_NFD(f, n_spec = 2, args = ()):
    """Compute the ND and FD for a differential equation f
    
    Compute the niche difference (ND), niche overlapp (NO), 
    fitnes difference(FD) and conversion factors (c) as defined in Spaak et al.
    
    Parameters
    -----------
    f : callable ``f(N, *args)``
        Percapita growth rate of the species.
        1/N dN/dt = f(N)
        f must take and return an array
    n_spec : int, optional, default = 2
        number of species in the system
    args : tuple, optional
        Any extra arguments to `f
        
    Returns
    -------
    ND : ndarray (shape = n_spec)
        Niche difference of the species to the other species
    NO : ndarray (shape = n_spec)
        Niche overlapp of the species (NO = 1-ND)
    FD : ndarray (shape = n_spec)
        Fitness difference
    c : ndarray (shape = (n_spec,n_spec))
        conversion factors from one species to the oter. 1/c = c.T
        
    Literature:
    The unified Niche and Fitness definition, J.W.Spaak, F. deLaender
    """
    # obtain equilibria densities and invasion growth rates
    pars = preconditioner(f,n_spec, args)
    
    # list of all species
    l_spec = list(range(n_spec))
    
    # compute conversion factors
    c = np.ones((n_spec,n_spec))
    for i in l_spec:
        for j in l_spec:
            if i>=j:
                continue
            c[i,j] = solve_c(pars,[i,j])
            c[j,i] = 1/c[i,j]

    # compute NO and FD
    NO = np.empty(n_spec)
    FD = np.empty(n_spec)
    
    for i in l_spec:
        # creat a list with i at the beginning
        sp = np.array([i]+l_spec[:i]+l_spec[i+1:])
        # compute NO and FD
        NO[i] = NO_fun(pars, c[i, sp[1:]], sp)
        FD[i] = FD_fun(pars, c[i, sp[1:]], sp)
    return 1-NO, NO, FD, c
    
def preconditioner(f, n_spec, args = ()):
    """Returns equilibria densities and invasion growth rates for system `f`
    
    Parameters
    -----------
    f : callable ``f(N, *args)``
        Percapita growth rate of the species.
        1/N dN/dt = f(N)
        f must take and return an array
    n_spec : int, optional, default = 2
        number of species in the system
    args : tuple, optional
        Any extra arguments to `f
            
    Returns
    -------
    pars : dict
        A dictionary with the keys:
        
        ``N_star`` : ndarray (shape = (n_spec, n_spec))
            N_star[i] is the equilibrium density of the system with species 
            i absent. The density of species i is set to 0.
        ``r_i`` : ndarray (shape = n_spec)
            invsaion growth rates of the species
    """
    pars = {}
    # equilibrium densities
    N_star = np.empty((n_spec, n_spec))
    N_star_pre = np.ones(n_spec-1)
    # invasion growth rates
    r_i = np.empty(n_spec)
    for i in range(n_spec):
        # to set species i to 0
        ind = np.arange(n_spec) != i
        # solve for equilibrium, use equilibrium dens. of previous run
        N_star_pre = fsolve(lambda N: f(np.insert(N,i,0))[ind], N_star_pre)
        N_star[i] = np.insert(N_star_pre,i,0)
        # compute invasion growth rates
        r_i[i] = f(N_star[i])[i]
    pars = {"N_star": N_star, "r_i": r_i, "f": f}
    return pars
    
def solve_c(pars, sp = [0,1]):
    """find the conversion factor c for species sp
    
    Parameters
    ----------
    pars : dict
        Containing the N_star and r_i values, see `preconditioner`
    sp: array-like
        The two species to convert into each other
        
    Returns
    -------
    c : float, the conversion factor c_sp[0]^sp[1]
    """
    sp = np.asarray(sp)
    def inter_fun(c):
        # equation to be solves
        NO_ij = np.abs(NO_fun(pars,c, sp))
        NO_ji = np.abs(NO_fun(pars,1/c,sp[::-1]))
        return NO_ij-NO_ji

    # find interval for brentq method
    a = 1
    # find which species has higher NO for c =1
    direction = np.sign(inter_fun(a))
    fac = 2**direction
    b = a*fac
    # change searching range to find c with changed size of NO
    while np.sign(inter_fun(b)) == direction:
        a = b
        b *= fac
    
    # solve equation
    return brentq(inter_fun,a,b)
    
def NO_fun(pars,c, sp):
    # Compute NO for specis sp and conversion factor c
    f0 = pars["f"](switch_niche(pars["N_star"][sp[0]],sp))[sp[0]]
    fc = pars["f"](switch_niche(pars["N_star"][sp[0]],sp,c))[sp[0]]
    
    return (f0-pars["r_i"][sp[0]])/(f0-fc)
    
def FD_fun(pars, c, sp):
    # compute the FD for species sp and conversion factor c
    f0 = pars["f"](switch_niche(pars["N_star"][sp[0]],sp))[sp[0]]
    fc = pars["f"](switch_niche(pars["N_star"][sp[0]],sp,c))[sp[0]]
    return fc/f0
    
def switch_niche(N,sp,c=0):
    # switch the niche of sp[1:] into niche of sp[0]
    N = N.copy()
    N[sp[0]] += np.sum(c*N[sp[1:]])
    N[sp[1:]] = 0
    return N