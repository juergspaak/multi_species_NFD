
import numpy as np
import pandas as pd
from scipy.stats import linregress
from itertools import product


n_max = 6
try:
    data
except NameError:
    data = pd.read_csv("fullfactorial_data.csv")
<<<<<<< HEAD
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
=======


cases = sorted(set(data.case))
keys = data.keys()[:7]
regressions = pd.DataFrame(index = np.arange(len(cases)), columns = keys)
regressions["case"] = cases
for fit in product(["ND", "FD"], ["", "_var"], ["_slope", "_intercept"]):
    regressions[fit[0]+fit[1]+fit[2]] = np.nan
    
for fit in ["a_slope", "a_intercept"]:
    regressions[fit] = np.nan

n_coms = ["n_com_{}".format(i) for i in range(2, n_max+1)] 
for n_com in n_coms:
    regressions[n_com] = np.nan
>>>>>>> real_interaction_data

n_specs = np.arange(2, n_max +1)

# compute regressions for each data set
for i,row in regressions.iterrows():
    data_c = data[data.case == row.case] # select current case
    NFDs = np.array([data_c[["ND", "FD"]][data_c.richness == n].values
           for n in n_specs])
    
    # outlier save version of variance
<<<<<<< HEAD
    NFDs_var = np.percentile(NFDs, [25,75], axis = 1)
    NFDs_var = NFDs_var[1]-NFDs_var[0]
    
=======
    NFDs_var = np.nanpercentile(NFDs, [25,75], axis = 1)
    NFDs_var = NFDs_var[1]-NFDs_var[0]
    
    data_c = data_c[np.isfinite(data_c.ND*data_c.FD)]
    
>>>>>>> real_interaction_data
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
<<<<<<< HEAD
    print(i)
=======
    
    regressions.at[i, ["a_slope", "a_intercept"]] = linregress(data_c.richness,
                  data_c.a)[:2]
    regressions.at[i, n_coms] = np.sum(np.isfinite(NFDs[...,0]), axis = 1)
    
    for key in keys:
        regressions.at[i, key] = data_c[key].iloc[0]
    print(i, regressions.loc[i, "ord1_strength"],
          np.sum(np.isfinite(NFDs[...,0]), axis = 1))
>>>>>>> real_interaction_data
    
regressions.to_csv("regression_fullfactorial.csv", index = False)