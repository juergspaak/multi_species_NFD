"""
@author: J.W. Spaak
Compute realistic distributions of interaction matrices
generates Interaction_strength.pdf, a figure from the appendix"""

import numpy as np
from LV_real_multispec_com import LV_pars, max_spec 

from scipy.stats import gaussian_kde


data = []

for i in range(2, max_spec + 1):
    if i<=1:
        continue
    A = LV_pars["matrix"][i]
    A = A[LV_pars["NFD_comp"][i]]
    A = A[LV_pars["coex_invasion"][i]]
    A[:, np.arange(A.shape[-1]), np.arange(A.shape[-1])] = np.nan
    data.extend(A.flatten())

# remove outliers
data = np.array(data)
data = data[np.isfinite(data)]
# compute interquartile range
cut = 2.5
quantiles2 = np.percentile(data, [cut, 100-cut])
cut = 25
quantiles = np.percentile(data, [cut, 100-cut])
IQR = quantiles[1]-quantiles[0]

# retain non-outlies, defines by Q1-1.5*IQR < data < Q3 +1.5 IQR
scale = 1.5 # 1.5 corresponds to 99% retained, 0.9625 corresponds to 95%
treshholds = quantiles + IQR * np.array([-scale, scale])

# realistic resampling of the data
data_wide = data[(treshholds[0] < data) &
            (data<treshholds[1])]

# fit kernel density distribution
interaction_wide = gaussian_kde(data_wide)
resample_wide = lambda N: interaction_wide.resample(int(N)).reshape(-1)

# resampling of the data such that all communities coexist
data_short = data[(quantiles[0] < data) &
            (data<quantiles[1])]

# fit kernel density distribution
interaction_short = gaussian_kde(data_short)
resample_short = lambda N: interaction_short.resample(int(N)).reshape(-1)

if __name__ == "__main__":   
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, sharex = True)
    bins = np.linspace(-2,2,101)
    ax[0].hist(data_short, bins = bins, density = True, color = "red")
    x = np.linspace(-2,2, 501)
    ax[0].plot(x,interaction_short(x), 'k', linewidth = 3)
    
    ax[1].hist(data_wide, bins = bins, density = True,
      color = "blue")
    x = np.linspace(-2,2, 501)
    ax[1].plot(x,interaction_wide(x), 'k', linewidth = 3)
    ax[1].set_xlim([-1.1, 1.1])
    ax[1].set_xticks([-1,0,1])
    
    # layout
    ax[0].set_ylabel("density", fontsize = 14)
    ax[1].set_ylabel("density", fontsize = 14)
    ax[1].set_xlabel("interaction strength", fontsize = 14)
    
    ax[0].set_title("A: Weak interactions", loc = "left")
    ax[1].set_title("B: Strong interactions", loc = "left")
    
    fig.savefig("Interaction_strength.pdf")