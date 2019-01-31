import numpy as np
import matplotlib.pyplot as plt

from nfd_definitions.numerical_NFD import NFD_model, InputError

n_spec_max = 3
n_res = 2*n_spec_max

overlap = 0.1 # overlap in the speces specific resources
# resources that are species specific, i.e. species i eats res. i
ND_res = overlap*np.ones((n_spec_max, n_spec_max))
np.fill_diagonal(ND_res,1)

# general resources
NO_res = 9*overlap*np.ones((n_spec_max, n_spec_max))
np.fill_diagonal(NO_res,1)

util = np.append(ND_res, NO_res, axis = 1)

K = np.ones(n_res) # carrying capacity of the resources
r = 3*np.ones(n_res) # regeneration speed of the resources

m = np.sum(util*K,axis = 1)/2
             
def res_model(N,util,m,r, ret = False):
    
    # compute density of resources
    R_star = K*(1-np.sum(util*N[:,np.newaxis],axis = 0)/r)
    R_star[R_star<0] = 0
    if not ret:      
        return np.sum(util*R_star,axis = -1) -m
    else:
        return R_star

r_vals = np.linspace(0.1,1,10)
NO_cur = np.full((n_spec_max, n_spec_max), np.nan)

count = 0
count2 = 0

for counter in range(1000):
    r = np.random.uniform(0,3,n_res)
    try:
        pars_01 = NFD_model(res_model, args = (util[:2], m[:2], r))
        pars_02 = NFD_model(res_model, args = (util[[0,2]], m[[0,2]], r))
        pars_12 = NFD_model(res_model, args = (util[1:], m[1:], r))
        pars_3 = NFD_model(res_model, n_spec = 3, args = (util, m, r))
        NO_cur[0,:2] = pars_01["NO"]
        NO_cur[1,[0,2]] = pars_02["NO"]
        NO_cur[2,1:] = pars_12["NO"]
    except InputError:
        continue
    
    if (pars_3["NO"]>np.nanmax(NO_cur, axis = 0)*1.01).all():
        print(r)
        
        print(pars_01["NO"])
        print(pars_02["NO"])
        print(pars_12["NO"])
        print(NO_cur)
        print(pars_3["NO"], "\n")
        count+=1
        break
    elif (pars_3["NO"]>np.nanmax(NO_cur, axis = 0)*1.01).all():
        count2 += 1
        
        
###############################################################################
# compute which resources have been depleeted

resources = np.full((4,3,2,6), np.nan)
specs = [[0,1], [0,2], [1,2], [0,1,2]]
for i,pars in enumerate([pars_01, pars_02, pars_12, pars_3]):
    c, N_star = pars["c"], pars["N_star"]
    for j in range(len(pars["NO"])):
        util_c, m_c = util[specs[i]], m[specs[i]]
        # compute resources at invasion
        resources[i,specs[i][j],0] = res_model(N_star[j], util_c, m_c, r, True)
        test = res_model(N_star[j], util_c, m_c, r)
        #print(test, pars["r_i"][j])
        
        N_c = np.zeros(len(N_star))
        N_c[j] = np.sum(N_star[j]*c[j])
        resources[i,specs[i][j],1] = res_model(N_c, util_c, m_c, r, True)
        test = res_model(N_c, util_c, m_c, r)
        
        #print(pars["NO"][j], (pars["f0"][j]-pars["r_i"][j])/(pars["f0"][j]-test[j]), "NO")
        
    
colors = ["red", "green", "orange", "black"]
spec = 2
for i in range(4):
    plt.plot(resources[i,spec,0].T, color = colors[i])
    plt.plot(resources[i,spec,1].T, marker = 'o', color = colors[i])