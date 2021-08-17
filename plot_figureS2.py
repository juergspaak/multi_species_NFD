import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

regress = pd.read_csv("regression_fullfactorial.csv")


fig, ax = plt.subplots(2,2,figsize = (8,8))
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
s = 25
alpha = 0.5

marker = ["o",1,"+"]
c = ["blue", "red"]

for i,case1 in enumerate(sorted(set(regress.ord1))):
    ind1 = regress.ord1 == case1
    for j, case2 in enumerate(sorted(set(regress.ord1_strength))):
        ind2 = regress.ord1_strength == case2
        ind = ind1 & ind2
        # plot the mean of ND
        ax[0,0].scatter(regress["ND_slope"][ind],
                      regress["ND_intercept"][ind],
                      s = s,  c = c[j], alpha = alpha, marker = marker[i],
                      label = None)
        
        # mean of FD
        ax[0,1].scatter(regress["FD_slope"][ind],
                      regress["FD_intercept"][ind],
                      s = s,  c = c[j], alpha = alpha, marker = marker[i])
        
        # variance of ND
        ax[1,0].scatter(regress["ND_var_slope"][ind],
                      regress["ND_var_intercept"][ind],
                      s = s,  c = c[j], alpha = alpha, marker = marker[i])
        
        # variance of FD
        ax[1,1].scatter(regress["FD_var_slope"][ind],
                      regress["FD_var_intercept"][ind],
                      s = s,  c = c[j], alpha = alpha, marker = marker[i])

ax[0,0].plot(np.nan, np.nan, 'rs', label = "weak")
ax[0,0].plot(np.nan, np.nan, 'bs', label = "strong")   
ax[0,0].plot(np.nan, np.nan, 'k+', label = "facilitation")
ax[0,0].scatter([np.nan, np.nan],[np.nan, np.nan],
  color = 'k', label = "competition", marker = 1, s = s)   
ax[0,0].plot(np.nan, np.nan, 'ko', label = "both")
ax[0,0].legend(loc = "lower left")

fig.tight_layout()

fig.savefig("Figure_S2.pdf")