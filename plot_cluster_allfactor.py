
import numpy as np
import matplotlib.pyplot as plt

# determine string and parameter settings for run   
interaction = ["neg, ", "bot, ", "pos, "] # 1. order interaction
ord_2 = ["neg, ", "bot, ", "pos, ", "abs, "] # second order interaction
ord_3 = ["pre, ", "abs, "] # presence of third order interaction
correlation = ["pos, ", "neg, ", "nul, "]
connectance = ["h, ", "m, ", "l, "] # connectance

interaction = [inter + o2 for inter in interaction for o2 in ord_2]
cor_con = [con + cor for cor in correlation for con in connectance]

string = "C:/Users/jspaak/Documents UNamur/NFD_values_multispecies"
string += "/NFD_values_ {}.npz"
add_FD = 9 + 1
richness = np.arange(2, 7)
col = ["green", "blue"]


for inter in interaction:        
    fig = plt.figure(figsize = (18,9))         
    for i, cc in enumerate(cor_con):
        if i == 0:
            ax_ND = fig.add_subplot(6,3,1)
            ax_FD = fig.add_subplot(6,3,add_FD)
            ax_ND.set_title(cc)
        else:
            ax_ND = fig.add_subplot(6,3,i + 1, sharey = ax_ND)
            ax_ND.set_title(cc)
            ax_FD = fig.add_subplot(6,3,i + add_FD, sharey = ax_FD)
        for k, o3 in enumerate(ord_3):
            add = (k-1)/4
            data = np.load(string.format(inter + o3 + cc))
            ax_ND.boxplot([ND[np.isfinite(ND)] for ND in data["ND"]]
                , positions = richness + add,
                           showfliers = False, boxprops = dict(color = col[k]))
            ax_FD.boxplot([FD[np.isfinite(FD)] for FD in data["FD"]]
                , positions = richness + add,
                           showfliers = False, boxprops = dict(color = col[k]))
    fig.tight_layout()
    plt.show()
    fig.savefig("Figure{}.png".format(inter))