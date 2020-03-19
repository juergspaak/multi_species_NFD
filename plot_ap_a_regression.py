"""
@author: J.W. Spaak
compute NFD values for randomly generated matrices and plot NFD distributions
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# load the regression lines
regress = pd.read_csv("regression_fullfactorial.csv")
n_specs = np.arange(2,7)
fig, ax = plt.subplots(1,1)
color = {"weak, ": "purple", "strong, ": "yellow", "pos, ": "red"}
for i,row in regress.iterrows():
    ax.plot(n_specs, row.a_intercept + row.a_slope*n_specs,
               color = color[row.ord1_strength], alpha = 0.1)



ax.set_ylabel(r"interaction strength $|a|$")
ax.set_xlabel("species richness")
ax.set_xticks(n_specs)

fig.savefig("species_richenss_on_alpha.pdf")