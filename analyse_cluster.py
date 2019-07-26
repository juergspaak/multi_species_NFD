
import numpy as np
import pandas as pd

# determine string and parameter settings for run   
ord1 = ["neg, ", "bot, ", "pos, "] # 1. order interaction
ord2 = ["neg, ", "bot, ", "pos, ", "abs, "] # second order interaction
ord3 = ["pre, ", "abs, "] # presence of third order interaction
correlation = ["pos, ", "neg, ", "nul, "]
connectance = ["h, ", "m, ", "l, "] # connectance

n = 100
richness = np.arange(2,7)
strings = [i+j+k+l+m for i in ord1 for j in ord2 for k in ord3
           for l in connectance for m in correlation]
parameters = np.array([[i,j,k,l,m] for i in ord1
    for j in ord2 for k in ord3 for l in connectance for m in correlation])
parameters = np.repeat(parameters, n*len(richness), 0)
file_str = "C:/Users/jspaak/Documents UNamur/NFD_values_multispecies"
file_str = "NFD_val"
file_str += "/NFD_values {}.npz"

df = pd.DataFrame(parameters)
df.columns = ["ord1", "ord2", "ord3", "con", "cor"]
df["id"] = np.repeat(strings, n*len(richness))
df["richness"] = np.tile(np.repeat(richness, n), 216)
df["ND"] = np.nan
df["FD"] = np.nan
df["degA"] = np.nan
df["degB"] = np.nan
df["degC"] = np.nan

for i, string in enumerate(strings):
    print(i, df.shape)
    file = np.load(file_str.format(string))
    df.loc[df.id == string, "ND"] = file["ND"][:,:,0].flatten()
    df.loc[df.id == string, "FD"] = file["FD"][:,:,0].flatten()
    for i, order in enumerate(["A", "B", "C"]):
        link = np.isfinite(file[order]) & (file[order] != 0)
        link = link[:,:,0] # take only first species per community
        df.loc[df.id == string, "deg" + order] = np.sum(link, axis = 
              tuple(np.arange(2, link.ndim))).flatten()
df["degA"] = df.degA -1 # remove self-link
df.to_csv("test2.csv", index = False)

print(np.sum(np.isfinite(df.ND)))
print(np.sum(np.isfinite(df.FD)))