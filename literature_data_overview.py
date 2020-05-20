import numpy as np
import pandas as pd
from scipy.special import comb

from LV_real_multispec_com import LV_pars, max_spec, LV_multi_spec

# simple summary for number of communities
n_specs = np.arange(2,7)
com_sum = pd.DataFrame()
# distinct communities from papers
dist_com = [sum(LV_multi_spec.n_spec == i) for i in range(max_spec+1)]
com_sum["dist_com"] = dist_com
# maximal number of communities
com_sum["max_com"] = [int(sum(dist_com * comb(np.arange(0,max_spec +1),i)))
        for i in range(0,max_spec+1)]
# communities for which all parameters exist and are nonzero
com_sum["full_com"] = [len(mat) for mat in LV_pars["matrix"]]
# communities for which we can compute NFD parameters
[sum(comp) for comp in LV_pars["NFD_comp"]]
com_sum["NFD_comp"] = [len(ND) for ND in LV_pars["ND"]]
# communities with stable equilibrium
com_sum["coex"] = [sum(coex) for coex in LV_pars["real_coex"]]
com_sum["no_coex"] = com_sum["full_com"]-com_sum["coex"]



# number of communities, for which invasion is not possible, or does not
# predict coexistnece, but can coexist
coex_real = LV_pars["real_coex"]
NFD_comp = LV_pars["NFD_comp"]
coex_invasion = LV_pars["coex_invasion"]


coex_no_inv = [coex_real[i] & (~NFD_comp[i]) for i in n_specs]
inv_wrong = [coex_real[i][NFD_comp[i]]  != coex_invasion[i] for i in n_specs]
com_sum["no_inv"] = 0
com_sum["no_inv"].iloc[n_specs] = [sum(c) for c in coex_no_inv]
com_sum["inv_wrong"] = 0
com_sum["inv_wrong"].iloc[n_specs] = [sum(c) for c in inv_wrong]
com_sum["NFD_coex"] = com_sum["coex"]-com_sum["no_inv"]
com_sum["NFD_no_coex"] = com_sum["NFD_comp"] -com_sum["NFD_coex"]
com_sum = com_sum.T

com_sum["total"] = np.sum(com_sum.values, axis = 1)
print(com_sum)
com_sum.index = ["Original matrices", "Subcommunities",
                 "Complete\n int. matrix", "NFD computed", "coexistence",
                 "comp. exclusion", "no invasion analysis", "invasion wrong",
                 "NFD coexistence", "NFD comp. excl"]
del(com_sum[0])
del(com_sum[1])
com_sum.to_csv("literature_data_overview.csv", index = True)  
