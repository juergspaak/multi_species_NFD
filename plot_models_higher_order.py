import numpy as np
import matplotlib.pyplot as plt

from higher_order_models import NFD_higher_order_LV
# parameters for running code
np.random.seed(28)
n_com = 1000 # number of communities at the beginning
mu = 1
# the prefactor is chosen to resemble the real dataset distributions of alpha
alpha,beta,gamma = 0.26*np.array([0.1,0.01,0.001])
a_ij_min = 0
a_ij_max = 1
n_order = 3

richness = np.arange(2,7)

NO_all, FD_all = np.full((2,n_order,len(richness),
                                    n_com,richness[-1]),np.nan)
c_all = np.full((n_order, len(richness), n_com, richness[-1], richness[-1]),
                 np.nan)

for r,n in enumerate(richness):

    A = np.random.uniform(a_ij_min,a_ij_max,size = (n_com, n,n))
    B = np.random.uniform(a_ij_min,a_ij_max,size = (n_com, n,n,n))
    C = np.random.uniform(a_ij_min,a_ij_max,size = (n_com, n,n,n,n))
    A,B,C = -alpha * A, -beta * B, -gamma * C
    A[:,np.arange(n), np.arange(n)] = -1
    #B[:,np.arange(n), np.arange(n), np.arange(n)] = 0
    #C[:,np.arange(n), np.arange(n), np.arange(n), np.arange(n)] = 0
     
    
    interactions = [A,B,C]
     
    for order in range(n_order):
        
        NO,FD,c = NFD_higher_order_LV(mu,*interactions[:order+1])
        NO_all[order,r,:len(NO),:n] = NO
        FD_all[order,r,:len(FD),:n] = FD
        c_all[order,r,:len(c),:n,:n] = c
        print("richness {}; order {}, communities {}".format(
                r+2, order+2, len(NO)))
    print("\n\n")
        

c_analyse = c_all.copy()
c_analyse[c_analyse <= 1] = np.nan
fs = 14    
fig = plt.figure(figsize = (11,11))

for m in range(n_order):
    
    if m == 0:
        ax_NO = fig.add_subplot(6,n_order,m+1)
        ax_FD = fig.add_subplot(6,n_order,m+n_order+1)
        ax_coex = fig.add_subplot(3,n_order,n_order + m + 1)
        ax_c = fig.add_subplot(3,n_order, 2*n_order + m + 1)
    else:
        ax_NO = fig.add_subplot(6,n_order,m+1, sharey = ax_NO)
        ax_FD = fig.add_subplot(6,n_order,m+n_order+1, sharey = ax_FD)
        ax_coex = fig.add_subplot(3,n_order,n_order + m + 1, 
                                  sharey = ax_coex, sharex = ax_coex)
        ax_c = fig.add_subplot(3,n_order, 2*n_order + m + 1, sharex = ax_c,
                               sharey = ax_c)
    ax_NO.boxplot([1-NO[np.isfinite(NO)] for NO in NO_all[m,...,0]],
                  positions = richness, showfliers = False)

    
    ax_FD.boxplot([FD[np.isfinite(FD)] for FD in FD_all[m,...,0]],
                  positions = richness, showfliers = False)
    ax_FD.set_xlabel("number of species")
    ax_NO.set_title("{}. order interactions".format(m + 1), fontsize = fs)
    ax_n_com = ax_NO.twinx()
    ax_n_com.plot(richness,np.sum(np.isfinite(NO_all[m,...,0]),axis = -1)/n_com
                ,alpha = 0.5)
    ax_n_com.set_ylim(0,1)
    
    
    
    x = np.linspace(0,1,1000)
    color = np.ones(NO_all[m].shape)
    color[:] = np.arange(len(color)).reshape(-1,1,1)
    im = ax_coex.scatter(1-NO_all[m], FD_all[m], s = 5,linewidth = 0,
                         c = color)
    ax_coex.plot(x,-x/(1-x), color = "black")
    ax_coex.set_xlim(np.nanpercentile(1-NO_all,[1,99]))
    ax_coex.set_ylim(np.nanpercentile(FD_all,[1,99]))
    ax_coex.invert_yaxis()
    
    ax_coex.set_xlabel(r"$\mathcal{N}$", fontsize = fs)
    
    ax_c.boxplot([c[np.isfinite(c)] for c in c_analyse[m]],
                    positions = richness, showfliers = False)
    
    if m == 0:
        ax_NO.set_ylabel(r"$\mathcal{ND}$", fontsize = fs)     
        ax_FD.set_ylabel(r"$\mathcal{F}$", fontsize = fs)     
        ax_coex.set_ylabel(r"$-\mathcal{F}$", fontsize = fs)
fig.tight_layout()
fig.savefig("multispecies_higher_order.pdf")

np.savez("multispecies_NFD", A = A, B = B, C = C,
         ND = 1-NO, FD = FD, c = c)


