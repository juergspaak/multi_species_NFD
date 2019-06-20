import numpy as np

import sys
from higher_order_models import NFD_higher_order_LV

try:
    job_id = int(sys.argv[1])-1
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
print(string)
parameters = parameters[job_id]

# parameters for running code
n_com = 10 # number of communities at the beginning
mu = 1
alpha,beta,gamma = 0.1,0.01,0.001

n_order = 3

richness = np.arange(2,7)

NO_all, FD_all = np.full((2,n_order,len(richness),
                                    n_com,richness[-1]),np.nan)
c_all = np.full((n_order, len(richness), n_com, richness[-1], richness[-1]),
                 np.nan)

for r,n in enumerate(richness):
    print(r,n)
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
    print("\n")


np.savez("NFD_values_{}".format(string), FD = FD_all, ND = 1-NO_all, c = c_all)


