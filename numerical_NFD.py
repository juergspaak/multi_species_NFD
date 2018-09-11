"""
@author: J.W.Spaak
Numerically compute ND and FD
"""

import numpy as np
from scipy.optimize import brentq, fsolve
from warnings import warn

def find_NFD(f, n_spec = 2, args = (), pars = {}, monotone_f = True):
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
    if (monotone_f is True) or monotone_f is False :
        monotone_f = np.full(n_spec, monotone_f, dtype = "bool" )
    f, pars = __input_check__(f, n_spec, args, pars)
    # obtain equilibria densities and invasion growth rates
    if len(pars.keys()) == 0:
        pars = preconditioner(f,n_spec, args, pars)
    else:
        pars["f"] = f
    
    # list of all species
    l_spec = list(range(n_spec))
    
    # compute conversion factors
    c = np.ones((n_spec,n_spec))
    for i in l_spec:
        for j in l_spec:
            if i>=j:
                continue
            c[i,j] = solve_c(pars,[i,j], monotone_f[i] and monotone_f[j])
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
  
def __input_check__(f, n_spec, args, pars):
    # check input on corectness
    if not isinstance(n_spec, int):
        raise InputError("Number of species (`n_spec`) must be an integer")
    
    try:
        f0 = f(np.zeros(n_spec))
        if not (f0.shape == (n_spec,)):
            raise InputError("`f` must return an array of length `n_spec`")            
    except TypeError:
        raise InputError("`f` must be a callable")
    except AttributeError:
        fold = f
        f = lambda N, *args: fold(N, *args)
        f0 = f(np.zeros(n_spec))
        warn("`f` does not return a proper `np.ndarray`")
    if min(f0)<0 or (not np.all(np.isfinite(f0))):
        raise InputError("All species must have positive monoculture growth"
                    +"i.e. `f(0)>0`. Especially this value must be defined")
    shapes = {"N_star": (n_spec, n_spec), "r_i": (n_spec,)}
    for key in pars.keys():
        if not (pars[key].shape == shapes[key]):
            warn("pars[{}] must have shape {}.".format(key,shapes[key])
                +" The {} values will be computed automatically".format(key))
            return f, {}
    return f, pars
        
class InputError(Exception):
    pass
        
def preconditioner(f, n_spec, args = (), pars = {}):
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
        res_growth = f(N_star[i])
        if np.amax(np.abs(res_growth[ind])/N_star[i,ind])>1e-10:
            raise InputError("Not able to find resident equilibrium density, "
                        + "with species {} absent.".format(i)
                        + " Please provide manually via the `pars` argument")
        r_i[i] = f(N_star[i])[i]
    pars = {"N_star": N_star, "r_i": r_i, "f": f}
    return pars
    
def solve_c(pars, sp = [0,1], monotone_f = True):
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
    
    if not monotone_f:
        c = fsolve(inter_fun,1)[0]
        if np.abs(inter_fun(c))>1e-10:
            raise ValueError("Not able to find c_{}^{}.".format(*sp) +
                "Please pass a better guess for c_i^j via the `pars` argument")
        return c

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
    try:
        return brentq(inter_fun,a,b)
    except ValueError:
        raise ValueError("f does not seem to be monotone. Please run with"
                         +"`monotone_f = False`")
    
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