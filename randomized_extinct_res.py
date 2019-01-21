"""see exting_res.py
Try to find a community structure, which feeds on resources.
Depending on the regeneration speed of the resources (r)
the 3 species community has consistently lower or higher ND values"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import odeint

traits = pd.read_csv("phytoplankton_traits.csv")
traits = traits[(traits.temperature <= 22) & (traits.temperature >=18)]
traits = traits[["mu_amm","k_amm_m", 
                     "mu_nit", "k_nit_m", "mu_p", "k_p_m"]].values.T

species = np.sum(np.isfinite(traits), axis = 0)>=4
traits = traits[:,species]
                
amm_nit = np.sum(np.isfinite(traits[:,:4]), axis = 0) == 4
amm_pho = np.sum(np.isfinite(traits[:,[0,1,4,5]]), axis = 0) == 4
nit_pho = np.sum(np.isfinite(traits[:,2:]), axis = 0) == 4


K = traits[1::2]
mu = traits[::2]

for i in range(3):
    plt.figure()
    nutrients = np.linspace(0,2*np.nanmax(K[:,i]),100)
    plt.plot(nutrients, nutrients[:,None]*mu[i]/(nutrients[:,None] + K[i]))
    
def phyto_growth(N, R_in, K, mu, c = 0.03, l = 0.5):
    R = R_in - np.sum(c*N, axis = 1)
    return N * (np.amin(R*K/(R+K), axis = 0) - l)


    