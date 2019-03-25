import numpy as np
import matplotlib.pyplot as plt
import sys

from higher_order_models import NFD_higher_order_LV

# getting data from jobscript 
try:
    job_id = int(sys.argv[1])
except IndexError:
    job_id = np.random.randint(18)

# determine string and parameter settings for run   
interaction = ["neg, ", "pos, ", "bot, "] # 1. order interaction
ord_2 = ["neg, ", "bot, ", "pos, "] # second order interaction
connectance = ["h", "l"] # connectance
strings = [i+j+k for i in interaction for j in ord_2 for k in connectance]
interaction =  [[-1,0], [0,1], [-1,1]]
ord_2 = [[-1,0], [-1,1], [0,1]]
connectance = [1,0.5]
parameters = [(i,j,k) for i in interaction for j in ord_2 for k in connectance]

string = strings[job_id]
parameters = parameters[job_id]

# parameters for running code
n_com = 100 # number of communities at the beginning
mu = 1
alpha,beta,gamma = 0.1,0.01,0.01

n_order = 3

richness = np.arange(2,7)

NO_all, FD_all = np.full((2,n_order,len(richness),
                                    n_com,richness[-1]),np.nan)
c_all = np.full((n_order, len(richness), n_com, richness[-1], richness[-1]),
                 np.nan)

for r,n in enumerate(richness):

    A = np.random.uniform(*parameters[0],size = (n_com, n,n))
    B = np.random.uniform(*parameters[1],size = (n_com, n,n,n))
    C = parameters[2]*np.random.uniform(*parameters[1],size = (n_com, n,n,n,n))
    A,B,C = -alpha * A, -beta * B, -gamma * C
    
    # set intraspecific effects
    A[:,np.arange(n), np.arange(n)] = -1
    B[:,np.arange(n), np.arange(n), np.arange(n)] = 0
    C[:,np.arange(n), np.arange(n), np.arange(n), np.arange(n)] = 0
     
    
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
fig.savefig("multispecies_{}.pdf".format(string))

np.savez("NFD_values_{}".format(string), FD = FD_all, ND = 1-NO_all, c = c_all)


