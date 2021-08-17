import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import rainbow
from scipy.optimize import curve_fit

from LV_real_multispec_com import LV_pars, matrices


origin = {}
spec_range = np.arange(2,7)

fig = plt.figure(figsize = (7,7))

ND_LV = LV_pars["ND"]
FD_LV = LV_pars["FD"]
ND_LV = [np.array(ND) for ND in ND_LV]
FD_LV = [np.array(FD) for FD in FD_LV]

ax_FD_LV = fig.add_subplot(2,2,3)
ax_ND_LV = fig.add_subplot(2,2,1)
pos = list(range(2,7))

ax_ND_LV.set_ylim([-1,2])
ax_FD_LV.set_ylim([-1,10])

ax_coex_LV = fig.add_subplot(1,2,2)
color = rainbow(np.linspace(0,1,len(spec_range)))
for i in spec_range:
    ax_coex_LV.scatter(ND_LV[i], FD_LV[i], s = (i+2)**2, alpha = 0.5
                       ,label = "{} species".format(i), c = color[i-2])
    ax_ND_LV.scatter(np.full(ND_LV[i].shape, i), ND_LV[i], s = 4, alpha = 0.2,
                  color = "grey")
    ax_FD_LV.scatter(np.full(FD_LV[i].shape, i), FD_LV[i], s = 4, alpha = 0.2,
                  color = "grey")
    
###############################################################################    
# find a community with negative ND, but still coexistes
special_ND = [[],[]]
special_FD = [[],[]]
for i in spec_range:
    # species that coexist for which ND can be computed
    coex = LV_pars["real_coex"][i][LV_pars["NFD_comp"][i]]
    neg_ND = np.amin(LV_pars["ND"][i], axis = -1)<0
    special_ND.append(LV_pars["ND"][i][neg_ND & coex])
    special_FD.append(LV_pars["FD"][i][neg_ND & coex])

i = 3
j = 10

special_matrix = LV_pars["matrix"][i][LV_pars["NFD_comp"][i]][j]

ax_coex_LV.plot(special_ND[i][j], special_FD[i][j], 'dimgrey',
                label = "example\ncommunity", linewidth = 3, zorder = 5)
lab = "cba"
s = [100,200,100]
for ind in range(len(lab)):
    ax_coex_LV.scatter(special_ND[i][j][ind], special_FD[i][j][ind],
                       marker = r"${}$".format(lab[ind]), c = "k",zorder = 10,
                       s = s[ind])

###############################################################################
# add layout



fs = 18
fs_label = fs-2
fs_axis = fs-6

# add layout
ax_ND_LV.set_title("A", fontsize = fs)
ax_FD_LV.set_title("B", fontsize = fs)
ax_coex_LV.set_title("C", fontsize = fs)

ax_coex_LV.legend(fontsize = fs_axis-2)

ax_FD_LV.set_xlabel("species richness",fontsize = fs_label)
ax_FD_LV.set_ylabel(r"$\mathcal{F}$",fontsize = fs_label)
ax_ND_LV.set_ylabel(r"$\mathcal{N}$",fontsize = fs_label)

ax_coex_LV.set_ylabel(r"$\mathcal{F}$",fontsize = fs_label)
ax_coex_LV.set_xlabel(r"$\mathcal{N}$",fontsize = fs_label)

# add ticks
ND_ticks, FD_ticks = [-1,0,1,2], np.array([-10,-5,0,1])
ax_ND_LV.set_yticks(ND_ticks)
ax_ND_LV.set_xticks(spec_range)
ax_ND_LV.set_xticklabels(len(spec_range)*[""])
ax_FD_LV.set_yticks(FD_ticks)
ax_FD_LV.set_xticks(spec_range)
ax_FD_LV.set_ylim([-10,1])
ax_FD_LV.invert_yaxis()
ax_coex_LV.set_xticks(ND_ticks)
ax_coex_LV.set_yticks(FD_ticks)
ax_coex_LV.set_ylim([-10,1])
ax_coex_LV.invert_yaxis()
ax_ND_LV.tick_params(axis='both', which='major', labelsize=fs_label)
ax_FD_LV.tick_params(axis='both', which='major', labelsize=fs_label)
ax_coex_LV.tick_params(axis='both', which='major', labelsize=fs_label)


ND_bound = np.linspace(-2,2,101)
ax_coex_LV.plot(ND_bound, -ND_bound/(1-ND_bound), "black")
ax_coex_LV.axhline(0, color = "grey", linestyle = "--")
ax_coex_LV.axvline(0, color = "grey", linestyle = "--")
ax_coex_LV.set_xlim(ax_ND_LV.get_ylim())

ND_reg_all = []
FD_reg_all = []
s_richness_all = []


def linear_theil_senn(y, random = False):
    
    y = [np.array(yi) for yi in y]
    
    slopes = []
    # estimates the slope via median, robust to outliers
    for i in range(len(y)):
        for j in range(len(y)):
            if j <= i:
                continue
            slopes.append((y[i].flatten()-
                           y[j].flatten()[:, np.newaxis]).flatten()/(j-i))
            
    gcm = np.lcm.reduce([len(s) for s in slopes])
    if random:
        slopes = np.array([np.random.choice(s, int(1e5)) for s in slopes])
    else:
        slopes = np.array([np.repeat(s, gcm//len(s)) for s in slopes])
    slope = np.nanmedian(slopes)
    
    intercept = [y[i] - slope*(i+2) for i in range(len(y))]
    intercept = np.nanmedian([item for sublist in intercept
                              for item in sublist.flatten()])
    
    return slope, intercept
 
def saturation_theil_senn(y, random = False):
    # fit a saturating function through the data, y = r*x/(x+h)
    y = [np.array(yi).flatten() for yi in y]
    
    h_all = []
    r_all = []
    # estimates the slope via median, robust to outliers
    for i in range(len(y)):
        for j in range(len(y)):
            if j <= i:
                continue
            det = y[i]*i - y[j][:,np.newaxis]*j
            h = 1/det*(y[i]*i*j-y[j][:,np.newaxis]*j*i)
            h_all.append(h.flatten())
            r = 1/det*(y[i]*y[j][:np.newaxis]*(j-i))
            r_all.append(r.flatten())
            
    gcm = np.lcm.reduce([len(r) for r in r_all])
    if random:
        r_all = np.array([np.random.choice(r, int(1e5)) for r in r_all])
        h_all = np.array([np.random.choice(h, int(1e5)) for h in h_all])
    else:
        r_all = np.array([np.repeat(r, gcm//len(r)) for r in r_all])
        h_all = np.array([np.repeat(h, gcm//len(h)) for r in h_all])
    h = np.nanmedian(r_all)
    r = np.nanmedian(h_all)
    
    return r,h


def sat_fit(y):
    y = [yi.flatten() for yi in y]
    if len(y) == 2:
        min_h = 1e5 # fit a line, not a saturating function
    else:
        min_h = 0
    x = [np.full(y[i].shape, i) for i in range(len(y))]
    
    y = np.array([i for yi in y for i in yi])
    x = np.array([i for xi in x for i in xi])
    x = x[np.isfinite(y)]
    y = y[np.isfinite(y)]
    
    return curve_fit(lambda x,r,h: r*(x)/(h+x), x, y, [1,min_h + 1],
                     bounds = ([-np.inf, min_h],np.inf))[0]               


def shapes(X):
    return np.array([len(x) for x in X])

slopes = []
rs = []
for key in matrices.keys():
    index = [np.array(LV_pars["origin"][i]) == key for i in spec_range]
    for i in spec_range:
        index[i-2][np.array(LV_pars["NFD_comp"][i])]
    index2 = [index[i-2][np.array(LV_pars["NFD_comp"][i])] for i in spec_range]
    
    ND_org = [LV_pars["ND"][i][index2[i-2]] for i in spec_range]
    FD_org = [LV_pars["FD"][i][index2[i-2]] for i in spec_range]
    
    ND_org = [n for n in ND_org if len(n)!= 0]
    FD_org = [n for n in FD_org if len(n)!= 0]
    
    if len(ND_org) != 1:
        y = FD_org
        r,h = sat_fit(FD_org)
        rs.append(r)
        richness = np.linspace(2, len(FD_org)+1, 51)
        FD_fit = r*(richness-2)/(richness-2+h)
        ax_FD_LV.plot(richness, FD_fit, color = "grey")
        
        slope, intercept = linear_theil_senn(ND_org)
        slopes.append(slope)
        key_range = np.array([2, len(ND_org) + 1])
        ax_ND_LV.plot(key_range, intercept + slope*key_range, color = "grey")


# fit data through all points
slope, intercept = linear_theil_senn([n for n in ND_LV if len(n) != 0], True)
print(slope, intercept)
ax_ND_LV.plot(spec_range, intercept + slope*spec_range, color = "black",
              linewidth = 2)

r,h = sat_fit([f for f in FD_LV if len(f) != 0])
print(r,h)
richness = np.linspace(2, 6, 51)
FD_fit = r*(richness-2)/(richness-2+h)
ax_FD_LV.plot(richness, FD_fit, color = "black", linewidth = 2)
 
fig.tight_layout()

fig.savefig("Figure_NFD_in_LV_real_per_origin.pdf")