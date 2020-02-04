
import numpy as np
import pandas as pd
from scipy.stats import linregress
from itertools import product


n_max = 6
try:
    data
except NameError:
    data = pd.read_csv("fullfactorial_data.csv")
    #data = data[data.con == "h, "]

factors = ["ord1", "ord2", "ord3", "cor", "con", "indirect"]
factors_dict = []
for factor in factors:
    factors_dict.append(sorted(set(data[factor])))
    
case = np.array(list(product(*factors_dict)))
regressions = pd.DataFrame(case, columns = factors)
regressions["case"] = np.sum(regressions, axis = 1)
for fit in product(["ND", "FD"], ["", "_var"], ["_slope", "_intercept"]):
    regressions[fit[0]+fit[1]+fit[2]] = np.nan

n_specs = np.arange(2, n_max +1)

# compute regressions for each data set
for i,row in regressions.iterrows():
    data_c = data[data.case == row.case] # select current case
    NFDs = np.array([data_c[["ND", "FD"]][data_c.richness == n].values
           for n in n_specs])
    
    # outlier save version of variance
    NFDs_var = np.percentile(NFDs, [25,75], axis = 1)
    NFDs_var = NFDs_var[1]-NFDs_var[0]
    
    # how factors affect variance and average of NFD
    n = np.arange(3,n_max +1).reshape(-1,1)
    regressions.at[i,["ND_slope", "ND_intercept"]] = linregress(data_c.richness
                  ,data_c.ND)[:2]
    regressions.at[i,["FD_slope", "FD_intercept"]] = linregress(data_c.richness
                  ,data_c.FD)[:2]
    regressions.at[i,["ND_var_slope", "ND_var_intercept"]] = linregress(n_specs
                  ,NFDs_var[:,0])[:2]
    regressions.at[i,["FD_var_slope", "FD_var_intercept"]] = linregress(n_specs
                  ,NFDs_var[:,1])[:2]
    print(i)
    
regressions.to_csv("regression_fullfactorial.csv", index = False)