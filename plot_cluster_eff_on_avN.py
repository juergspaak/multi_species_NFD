import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

regress = pd.read_csv("regression_fullfactorial.csv")

fig, ax = plt.subplots(2,2,figsize = (7,7))
ax[0,0].set_title("$\mathcal{N}$", fontsize = 14)
ax[0,1].set_title("$\mathcal{F}$", fontsize = 14)

ax_label = fig.add_subplot(111,frameon = False)
ax_label.tick_params(labelcolor="none", top = False,
                     bottom=False, left=False, right=False)
ax_label.set_ylabel("intercept")

ax[0,0].set_ylabel("mean")
ax[1,0].set_ylabel("variation")
ax[1,0].set_xlabel("slope")
ax[1,1].set_xlabel("slope")
s = 25
alpha = 0.5
marker = ["o",1,"+"]
c = ["blue", "red", "green"]
cases_ord1 = np.array([case[:5] for case in regress.case])
for i,case in enumerate(sorted(set(regress.ord1))):
    ind = regress.ord1 == case
    print(case)
    # plot the mean of ND
    ax[0,0].scatter(regress["ND_slope"][ind],
                  regress["ND_intercept"][ind],
                  s = s,  c = c[i], alpha = alpha, marker = marker[i])
    
    # mean of FD
    ax[0,1].scatter(regress["FD_slope"][ind],
                  regress["FD_intercept"][ind],
                  s = s,  c = c[i], alpha = alpha, marker = marker[i])
    
    # variance of ND
    ax[1,0].scatter(regress["ND_var_slope"][ind],
                  regress["ND_var_intercept"][ind],
                  s = s,  c = c[i], alpha = alpha, marker = marker[i])
    
    # variance of FD
    ax[1,1].scatter(regress["FD_var_slope"][ind],
                  regress["FD_var_intercept"][ind],
                  s = s,  c = c[i], alpha = alpha, marker = marker[i])
    
ax[0,0].set_xlim([-0.003, 0.003])
ax[1,0].set_xlim([-0.01, 0.001])
ax[1,0].set_ylim([0,None])
fig.tight_layout()
fig.savefig("figure_eff_combinations.pdf")