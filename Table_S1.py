import numpy as np
import pandas as pd
from itertools import product

regress = pd.read_csv("regression_fullfactorial.csv")
regress = regress[regress["ord1_strength"] == "weak, "]
regress = regress[np.isfinite(regress.FD_var_slope.values)]
factors = regress.keys()[:7]
factors = factors[factors != "ord1_strength"]

variables = list(product(["ND", "FD"], ["_", "_var_"], ["slope", "intercept"]))
variables = ["".join(var) for var in variables]

# median
percent_1 = pd.DataFrame(columns = variables[::2],
                       index = ["_".join([factor, level]) for factor in factors
                           for level in list(set(regress[factor]))])
percent_2 = pd.DataFrame(columns = variables[::2],
                       index = ["_".join([factor, level]) for factor in factors
                           for level in list(set(regress[factor]))])
ranges = pd.DataFrame(columns = variables[::2],
                       index = ["_".join([factor, level]) for factor in factors
                           for level in list(set(regress[factor]))])
percent =25
string = "[{}; {}]"
for factor in factors:
    for level in list(set(regress[factor])):
        for variable in variables[::2]:
            percents = np.round(np.nanpercentile(regress[regress[factor]
                        == level][variable], [percent, 100-percent]),3)
            
            ranges.loc["_".join([factor, level]), variable] = \
                    string.format(*percents)
                  
ranges.index = ["ord1: " + word for word in 
                  ["negative","unrestricted", "positive"]]\
                + ["ord2: " + word for word in 
                ["negative", "absent", "positive", "unrestricted"]]\
                + ["ord3: " + word for word in ["absent", "present"]] \
                + ["con: " + word for word in ["high", "middle", "low"]] \
                + ["cor: " + word for word in ["negative", "positive", "none"]]\
                + ["indirect: " + word for word in ["absent", "present"]]
ranges.columns = ["ND slope", "ND var slope", "FD slope", "FD var slope"]
                
ranges.to_csv("Table_S1.csv")