import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import viridis
import pandas as pd

regress = pd.read_csv("regression_fullfactorial.csv")
regress = regress[regress["ord1_strength"] == "weak, "]
factors = regress.keys()[:7]
s = 10
alpha = 0.5

for factor in factors:
    fig, ax = plt.subplots(2,2, figsize = (9,9))
    fig.suptitle(factor)
    ax[0,0].set_title("$\mathcal{N}$", fontsize = 14)
    ax[0,1].set_title("$\mathcal{F}$", fontsize = 14)
    
    ax_label = fig.add_subplot(111,frameon = False)
    ax_label.tick_params(labelcolor="none", top = False,
                         bottom=False, left=False, right=False)
    
    ax_label.set_ylabel("intercept\n", fontsize = 16)
    
    
    ax[0,0].set_ylabel("mean")
    ax[1,0].set_ylabel("variation")
    ax[1,0].set_xlabel("slope")
    ax[1,1].set_xlabel("slope")
    levels = list(set(regress[factor]))
    colors = viridis(np.linspace(0,1,len(levels)))
    for i,level in enumerate(levels):
        ind = regress[factor] == level
        ax[0,0].scatter(regress["ND_slope"][ind],
                      regress["ND_intercept"][ind],
                      s = s,  c = colors[i], alpha = alpha,
                      label = level)
        
        ax[1,0].scatter(regress["ND_var_slope"][ind],
                      regress["ND_var_intercept"][ind],
                      s = s,  c = colors[i], alpha = alpha,
                      label = None)
        
        ax[0,1].scatter(regress["FD_slope"][ind],
                      regress["ND_intercept"][ind],
                      s = s,  c = colors[i], alpha = alpha,
                      label = None)
        
        ax[1,1].scatter(regress["FD_var_slope"][ind],
                      regress["FD_var_intercept"][ind],
                      s = s,  c = colors[i], alpha = alpha,
                      label = None)
    ax[0,0].legend()