import numpy as np
from nfd_definitions.numerical_NFD import InputError, NFD_model
import warnings

def find_NFD_computables(A,r = None):
    """find all communities for which NFD can be computed
    
    Computes the feasibility and stability of the subcoms.
    
    Parameters:
    -----------
    A: np.array, shape = n_com, n_spec, n_spec
        The interaction matrices of a n_com communities
        with n_spec species per community
    r: np.array, shape = n_com, n_spec, optional
        Intrinsic growth rate of each species
            
    Returns:
    --------
    computable: boolean array, shape = n_com
        True if NFD can be commputed for said community
    sub_equi: np.array, shape = n_com, n_spec, n_spec
        equilibrium density of each species in each of the
        sub communites
    """
    n_com, n_spec = A.shape[:2]
    
    if r is None:
        r = np.ones((n_com, n_spec))
    
    # compute equilibrium densities where one species is
    # absent, i.e. the subcommunities
    sub_equi = np.zeros((n_com,n_spec,n_spec))
    sub_jacobi = np.empty((n_com,n_spec,n_spec-1),
                          dtype = "complex")
    
    for i in range(n_spec):
        # to remove species i
        inds = np.arange(n_spec)[np.arange(n_spec)!=i]
        # compute subcommunities equilibrium
        sub_equi[:,i,inds] = np.linalg.solve(
                A[:,inds[:,np.newaxis],inds],r[:,inds])
        # compute stability of subcommunities
        diag_sub_equi = np.swapaxes(sub_equi[:,[[i]],inds],1,2)
        sub_jacobi[:,i] = np.linalg.eigvals(-diag_sub_equi*
                  A[:,inds[:,np.newaxis],inds])
    
    # all subcommunities must be stable and feasible
    # sub_equi[:,i,i] = 0, all others must be positive
    should = n_spec*(n_spec-1)
    sub_feasible = np.sum(sub_equi>0, axis = (1,2)) == should
    sub_stable = np.all(np.real(sub_jacobi)<0, axis = (1,2))
    computable = sub_feasible & sub_stable
    
    # check whether results obtained by automatic comp. is equivalent
    i = np.random.randint(n_com)
    try:
        NFD_model(lambda N: r[i] - A[i].dot(N), n_spec = A.shape[1])
        if not computable[i]:
            raise ValueError("Cannot compute NFD values for com. {}".format(i)
                +"with LV-specific method, but with automatic")
    except InputError:
        if computable[i]:
            raise ValueError("Cannot compute NFD values for com. {}".format(i)
               + "with automatic method, but with LV_specific")
    
    return computable, sub_equi
    
def NFD_LV_multispecies(A,sub_equi, r = None, check = True, c_one = False):
    """compute NFD values for communities A
    
    LV model is assume to be of the form 
    1/N dN/dt = r - A.dot(N)
    
    Paramters
    ---------
    A: np.array, shape = n_com, n_spec, n_spec
        The interaction matrices of a n_com communities
        with n_spec species per community
    sub_equi: np.array, shape = n_com, n_spec, n_spec
        The equiblibrium density of the species in the
        subcommunities
    r: np.array, shape = n_com, n_spec, optional
        Intrinsic growth rate of each species
        
    Returns:
    --------
    ND: np.array, shape = n_com, n_spec
        ND values for each community
    FD: np.array, shape = n_com, n_spec
        FD values for each community
    c: np.array, shape = n_com, n_spec, n_spec
        the conversion factors for the species
    ND_ij: np.array, shape = n_com, n_spec, n_spec
        ND values of two species subcommunities
    FD_ij: np.array, shape = n_com, n_spec, n_spec
        FD values of two species subcommunities        
    """
    if r is None:
        r = np.ones(A.shape[:2])
    
    def own_einsum(string,*args):
        return np.sqrt(np.abs(np.einsum(string,*args)))
    
    # compute the two species niche overlap
    # sqrt (a_ij*a_ji/(a_ii*a_jj))
    NO_ij = own_einsum("nij,nji,nii,njj->nij",A,A,1/A,1/A)
    NO_ij *= np.sign(A) 
            
    
    # NO is a weighted average of the two species NO
    # weights = sqrt(a_ij/a_ji*a_jj/a_ii)
    c = own_einsum("nij,nji,njj,nii->nij", A,1/A,A,1/A)
    c[np.isnan(c)] = 0
    if c_one: # set all c to one, assume same resource use of all species
        c = np.ones(c.shape)
        check = False # checking results would always result in false
    try:
        NO = np.average(NO_ij, axis = -1, 
                        weights = c*sub_equi)
    except ZeroDivisionError: # avoid error of zero weights
        weights_sum = np.sum(c*sub_equi, axis = -1)
        NO = np.sum(NO_ij*c*sub_equi,axis = -1)/weights_sum
        NO[np.isnan(NO)] = 0
    
    # compute monoculture equilibrium density
    specs = np.arange(A.shape[-1])
    mono_equi = r/A[:,specs, specs]
    # F_i^j = 1- r_j/r_i*sqrt(a_ij/a_ji*a_ii/a_jj)
    FD_ij = own_einsum("nij,nji,nii,njj->nij",A,1/A,A,1/A)
    FD_ij[np.isnan(FD_ij)] = 0
    FD_ij = 1 - r[:,np.newaxis]/r[...,np.newaxis]*FD_ij
    FD = 1 - np.einsum("nij,nij,nj->ni",
                       1-FD_ij,sub_equi, mono_equi)
    
    # compute invasion growth rates of each species
    r_i = 1 - np.einsum("nji,nki->njk", A, sub_equi)
    # take only invasion growth rates, not resident growth rates
    r_i = r_i[:,np.arange(A.shape[1]), np.arange(A.shape[1])]
    
    # check whether results are correct with a random index
    if check:
        i = np.random.randint(len(FD))
        try:
            with warnings.catch_warnings(record = True):
                pars = {"N_star": sub_equi[i], "c": c[i]}
                pars = NFD_model(lambda N: r[i] - A[i].dot(N), 
                                 n_spec = A.shape[1], pars = pars)
            if not (np.allclose([pars["ND"], pars["FD"]],[1-NO[i], FD[i]],
                                rtol = 1e-5, atol = 1e-5)
                                and np.allclose(r_i[i],pars["r_i"])):
                print("ND automatic:", pars["ND"])
                print("ND LV_specific:", 1-NO[i])
                print("FD automatic:", pars["FD"])
                print("FD LV_specific:", FD[i])
                print(NO_ij[i],"NO_ij\n")
                print(A[i], "A\n")
                print(c[i], "c\n")
                raise ValueError("Computed NFD values are not exact enough"
                                 +" for {}th community".format(i))
        except InputError:
            print(A[i], r[i])
            raise ValueError("Could not compute NFD " +
                             "values for {}th com.".format(i))
    
    return 1-NO, FD, c, NO_ij, FD_ij, r_i

def diag_fill(A, values):
    # fill the diagonal of a multidimensional array `A` with `values`
    n = A.shape[-1]
    A[..., np.diag_indices(n)[0], np.diag_indices(n)[1]] = values
    return

def geo_mean(A,axis = None):
    # compute geometric mean
    return np.exp(np.nanmean(np.log(A), axis = axis))

def NFD_average(A_in):
    A = A_in.reshape(A_in.shape[:-2] + (-1,))
    A_all_inter = A[...,np.newaxis]*A[...,np.newaxis,:]
    diag_fill(A_all_inter, np.nan)
    return np.nanmean(np.sqrt(A_all_inter), axis = (1,2,3))