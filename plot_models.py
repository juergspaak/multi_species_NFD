import numpy as np
import matplotlib.pyplot as plt

from annual_plants_multispecies import NFD_annual_plants

# parameters for running code
n_com = 500 # number of communities at the beginning
max_alpha = 0.2 # maximal interspecific interaction strength
min_alpha = 0 # minimal interspecific interaction strength
min_lamb = 100 # minimal intrinsic growth rate
max_lamb = 200 # maximal intrinsic growth rate
richness = np.arange(2,11) # species richness

diag_one = True # whether to set intraspecific competition to 1
symm = False # whether A should be symmetric

class annual_plant(object):
    def __init__(self, F_fun, lamb_eq = lambda l:np.log(l), N_eq = lambda N:N):
        self.model = lambda N,A,lamb: np.log(lamb*F_fun(N,A))
        self.lamb_eq = lambda lamb: lamb_eq(lamb)
        self.N_eq = lambda N: N_eq(N)
        
standard = annual_plant(lambda N,A: 1/(1+A.dot(N)), lambda lamb: lamb-1)   

exponent = annual_plant(lambda N,A: np.exp(-A.dot(N)))
exp_log = annual_plant(lambda  N,A: np.exp(-A.dot(np.log(N+1))),
                       N_eq = lambda N: np.exp(N)-1)
models = [standard, exponent, exp_log]
model_names = ["standard", "exponent", "exp_log"]
model_equations = [r"$(1+A\cdot N)^{-1}$",
                   r"$\exp(-A\cdot N)$",
                   r"$\exp(-A\cdot \log(N+1))$"] 
n_models = len(models)     

NO_all, FD_all = np.full((2,len(models),len(richness),n_com,richness[-1])
                ,np.nan)

def diag_fill(A, values):
    n = A.shape[-1]
    A[:, np.diag_indices(n)[0], np.diag_indices(n)[1]] = values
    return

for r,n in enumerate(richness):
    
    A = np.random.uniform(min_alpha,max_alpha,size = (n_com,n,n))
    # intraspecific competition is assumed to be 1
    diag_fill(A, np.random.uniform(1,2,(n_com,n)))
    if diag_one:
        diag_fill(A,1)
    if symm:
        A_prime = (A + A.swapaxes(1,2))/2
    
    lamb = np.random.uniform(min_lamb, max_lamb,(n_com,n))
    for m,model in enumerate(models):
        
        NO,FD = NFD_annual_plants(A,lamb,model)
        print(m, "model", n, "richness", len(NO), len(FD), 
              sum(np.isfinite(NO)), sum(np.isfinite(FD)))
        NO_all[m,r,:len(NO),:n] = NO
        FD_all[m,r,:len(FD),:n] = FD
    print("\n\n")
        


fs = 14    
fig = plt.figure(figsize = (11,11))

for m in range(n_models):
    
    if m == 0:
        ax_NO = fig.add_subplot(4,n_models,m+1)
        ax_coex = fig.add_subplot(2,n_models,n_models + m + 1)
        ax_FD = fig.add_subplot(4,n_models,m+n_models+1)
    else:
        ax_NO = fig.add_subplot(4,n_models,m+1, sharey = ax_NO)
        ax_FD = fig.add_subplot(4,n_models,m+n_models+1, sharey = ax_FD)
        ax_coex = fig.add_subplot(2,n_models,n_models + m + 1, 
                                  sharey = ax_coex, sharex = ax_coex)
    ax_NO.boxplot([NO[np.isfinite(NO)] for NO in NO_all[m,...,0]],
                  positions = richness,showfliers = False)

    
    ax_FD.boxplot([FD[np.isfinite(FD)] for FD in FD_all[m,...,0]],
                  positions = richness, showfliers = False)
    ax_FD.set_xlabel("number of species")
    ax_NO.set_title(model_names[m], fontsize = fs)
    ax_FD.set_title(model_equations[m], fontsize = fs)
    ax_n_com = ax_NO.twinx()
    ax_n_com.plot(richness,np.sum(np.isfinite(NO_all[m,...,0]),axis = -1)/n_com
                ,alpha = 0.5)
    ax_n_com.set_ylim(0,1)
    
    
    
    x = np.linspace(0,1,1000)
    im = ax_coex.scatter(1-NO_all[m], FD_all[m], s = 5,linewidth = 0,
                         c = richness.reshape(-1,1,1)*np.ones(NO_all[m].shape))
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
fig.savefig("multispecies_annual plants, low alpha.pdf")