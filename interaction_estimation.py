"""
@author: J.W. Spaak
Compute realistic distributions of interaction matrices"""

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
    #print(i, np.nanmin(A), np.nanmax(A))
    A[:, np.arange(A.shape[-1]), np.arange(A.shape[-1])] = np.nan
    data.extend(A.flatten())

# remove outliers
data = np.array(data)
data = data[np.isfinite(data)]
# compute interquartile range
cut = 25
quantiles2 = np.percentile(data, [cut, 100-cut])
cut = 25
quantiles = np.percentile(data, [cut, 100-cut])
IQR = quantiles[1]-quantiles[0]

# retain non-outlies, defines by Q1-1.5*IQR < data < Q3 +1.5 IQR
scale = 1.5 # 1.5 corresponds to 99% retained, 0.9625 corresponds to 95%
treshholds = quantiles + IQR * np.array([-scale, scale])
treshholds = quantiles2
data = data[(treshholds[0] < data) &
            (data<treshholds[1])]

# fit kernel density distribution
interaction_kde = gaussian_kde(data)
resample = lambda N: interaction_kde.resample(int(N)).reshape(-1)
#resample = lambda N: np.random.normal(np.median(data), np.var(data)**0.5, int(N))

if __name__ == "__main__":   
    import matplotlib.pyplot as plt
    plt.hist(data, 100, density = True)
    print(np.var(data))
    plt.hist(np.random.normal(np.median(data), np.var(data)**0.5, int(1e5)),
             bins = 100,
             alpha = 0.5,
         density = True)

    x = np.linspace(-1,1, 101)
    plt.plot(x,interaction_kde (x))