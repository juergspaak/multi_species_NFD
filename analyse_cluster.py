"""
Combines the files `NFD_val/NFD_values *.npz` into the file
    `fullfactorial_data.csv`
"""

import numpy as np
import pandas as pd
from timeit import default_timer as timer
# determine string and parameter settings for run   
ord1 = ["neg, ", "bot, ", "pos, "] # 1. order interaction
ord2 = ["neg, ", "bot, ", "pos, ", "abs, "] # second order interaction
ord3 = ["pre, ", "abs, "] # presence of third order interaction
correlation = ["pos, ", "neg, ", "nul, "]
connectance = ["h, ", "m, ", "l, "] # connectance
indirect = ["pre, ", "abs, "] # indirect interactiosn between species
ord_1_strength = ["weak, ", "strong, "]

n = 100
richness = np.arange(2,7)
s_richness = np.arange(max(richness))
strings = [i+j+k+l+m+n+o for i in ord1 for j in ord2 for k in ord3
           for l in connectance for m in correlation 
           for n in ord_1_strength for o in indirect]
string_files = [i+j+k+l+m+n for i in ord1 for j in ord2 for k in ord3
           for l in connectance for m in correlation for n in ord_1_strength]
parameters = np.array([[i,j,k,l,m,n,o] for i in ord1
    for j in ord2 for k in ord3 for l in connectance for m in correlation
    for n in ord_1_strength for o in indirect])
parameters = np.repeat(parameters, n*len(richness), 0)
file_str = "NFD_val"
file_str += "/NFD_values {}.npz"

df = pd.DataFrame(parameters)
df.columns = ["ord1", "ord2", "ord3", "con", "cor", 
              "ord1_strength","indirect"]
df["case"] = (df.ord1 + df.ord2 + df.ord3 + df.cor + df.con +
            df.indirect + df.ord1_strength)
df["id"] = np.repeat(strings, n*len(richness))
df["richness"] = np.tile(np.repeat(richness, n), len(strings))
df["ND"] = np.nan
df["FD"] = np.nan
df["a"] = np.nan

time = timer()
for i, string in enumerate(string_files):
    print(i, df.shape, timer()-time)
    file = np.load(file_str.format(string))
    keys = ["ND", "FD"]
    df.loc[df.id == (string+"pre, "), keys] = np.array([
            file["ND"][:,:,0].flatten(),
            file["FD"][:,:,0].flatten()]).T
    df.loc[df.id == (string+"abs, "), keys] = np.array([
            file["ND_no_indir"][:,:,0].flatten(),
            file["FD_no_indir"][:,:,0].flatten()]).T
    A = file["A"]
    A[:,:,s_richness, s_richness] = np.nan
    print(np.sum(np.isfinite(file["ND"][:,:,0].flatten())),
          np.sum(np.isfinite(np.nanmean(np.abs(A),
                  axis = (-2,-1)).flatten())))
    df.loc[df.id == (string+"abs, "), "a"] = np.nanmean(np.abs(A),
                  axis = (-2,-1)).flatten()
    df.loc[df.id == (string+"pre, "), "a"] = np.nanmean(np.abs(A),
                  axis = (-2,-1)).flatten()

df.to_csv("fullfactorial_data.csv", index = False)

print(np.sum(np.isfinite(df.ND)))
print(np.sum(np.isfinite(df.FD)))