import matplotlib.pyplot as plt
import pandas as pd

# load regression data
# get regression data from fullfactorial design
regression = pd.read_csv("regression_fullfactorial.csv")

###############################################################################
# plot results

fig, ax = plt.subplots(2,2,figsize = (7,7))
ax[0,0].set_title("$\mathcal{N}$", fontsize = 14)
ax[0,1].set_title("$\mathcal{F}$", fontsize = 14)

ax_label = fig.add_subplot(111,frameon = False)
ax_label.tick_params(labelcolor="none", top = False,
                     bottom=False, left=False, right=False)
ax_label.set_ylabel("intercept\n\n")

ax[0,0].set_ylabel("mean")
ax[1,0].set_ylabel("variation")
ax[1,0].set_xlabel("slope")
ax[1,1].set_xlabel("slope")
s = 25
alpha = 0.5

variable = ["ND", "FD", "ND_var", "FD_var"]
col_indicator = "ord1"
color = ["red", "green", "blue", "orange"]
marker = [1, "+", "o"]
for i, var in enumerate(variable):
    axc = ax.flatten()[i]
    
    for j,factor in enumerate(set(regression[col_indicator])):
        datac = regression[regression[col_indicator] == factor]
        axc.scatter(datac[var+"_slope"], datac[var+"_intercept"],
                s = s, alpha = alpha, marker = marker[j],
                color = color[j])



ax[0,0].set_xlim([-0.003, 0.003])
ax[1,0].set_xlim([-0.01, 0.001])
ax[1,0].set_ylim([0,None])
fig.savefig("figure_eff_combinations.pdf")