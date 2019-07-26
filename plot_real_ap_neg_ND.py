import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import rainbow
from scipy.special import comb

from LV_real_multispec_com import LV_pars, LV_multi_spec, max_spec

# check whether there is a community in which ND<0 and has coexistence
neg_ND = []
neg_FD = []
for i in range(2, 7):
    coex = LV_pars["coex_invasion"][i]
    
    ND, FD = LV_pars["ND"][i][coex], LV_pars["FD"][i][coex]
    if len(ND) == 0:
        continue
    neg_id = np.sum(ND<0, axis = 1)>=1
    neg_ND.append(ND[neg_id])
    neg_FD.append(FD[neg_id])
    
for i in range(len(neg_ND)):
    plt.plot(neg_ND[i].flatten(), neg_FD[i].flatten(), 'o')
    


ND = np.linspace(*plt.gca().get_xlim(),1000)
ND = ND[ND<1]
plt.plot(ND, -ND/(1-ND))
plt.gca().invert_yaxis()
