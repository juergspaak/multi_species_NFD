import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

from nfd_definitions.numerical_NFD import NFD_model, InputError
from plot_intuitive_explanation import mu, var_cons, mean_cons, n_specs
import plot_intuitive_explanation as pi

###############################################################################
#
res_ext = np.linspace(-10,10, 1001)
res_ext = (mu*np.exp(-np.abs((mean_cons-res_ext[:,np.newaxis])/var_cons)**2)).T
A = np.einsum("ir,jr->ij",res_ext,res_ext) # interaction matrix
A_scal = A/np.diag(A)[:,np.newaxis]
A_scal = A_scal**3
mu_grow = np.sum(res_ext, axis = -1)

i = 0
ND,FD = [],[]
A_av = []
for rich in n_specs:
    ND_rich, FD_rich, A_rich = [],[], []
    for comb in combinations(np.arange(n_specs[-1]), rich):
        if 0 not in comb:
            continue
        comb = np.array(comb)
        A_current = A_scal[comb, comb[:,np.newaxis]]
        pars = {}
        pars = NFD_model(lambda N: np.ones(rich) - A_current.dot(N),
                         int(rich), pars = pars)
        ND_rich.append(pars["ND"][0])
        FD_rich.append(pars["FD"][0])
        np.fill_diagonal(A_current, np.nan)
        A_rich.append(np.nanmean(A_current))
    ND.append(ND_rich)
    FD.append(FD_rich)
    A_av.append(A_rich)
    
# specific combinations for increase/deacrease ND
ind = np.argsort(ND[0])
ND_up = np.array([0] + list(ind+1))
ND_down =  np.array([0] + list(ind[::-1]+1))

ND_change = np.empty((2,len(n_specs)))
FD_change = np.empty((2,len(n_specs)))
A_change = np.empty((2,len(n_specs)))
for i,rich in enumerate(pi.n_specs):
    A_current = A_scal[ND_up[:rich], ND_up[:rich,np.newaxis]]
    pars = {}
    pars = NFD_model(lambda N: np.ones(rich) - A_current.dot(N),
                     int(rich), pars = pars)
    ND_change[0,i] = pars["ND"][0]
    FD_change[0,i] = pars["FD"][0]
    np.fill_diagonal(A_current, np.nan)
    A_change[0,i] = np.nanmean(A_current)
    A_current = A_scal[ND_down[:rich], ND_down[:rich,np.newaxis]]
    pars = {}
    pars = NFD_model(lambda N: np.ones(rich) - A_current.dot(N),
                     int(rich), pars = pars)
    ND_change[1,i] = pars["ND"][0]
    FD_change[1,i] = pars["FD"][0]
    np.fill_diagonal(A_current, np.nan)
    A_change[1,i] = np.nanmean(A_current)
###############################################################################
#plot results   
fig = plt.figure(figsize = (7,7))

ax_ND = fig.add_subplot(221)
ax_ND.set_xlabel("Species richness")
ax_ND.set_ylabel("$\mathcal{N}_1$")
ax_ND.set_title("A")
ax_ND.set_yticks([0.7,0.8,0.9,1])
ax_FD = fig.add_subplot(222)
ax_FD.set_xlabel("Species richness")
ax_FD.set_ylabel("$\mathcal{F}_1$")
ax_FD.set_title("B")
ax_FD.invert_yaxis()
ax_res = fig.add_subplot(224)
ax_res.set_ylabel("resource utilisation\n$A_j$")
ax_res.set_xlabel("resources")
ax_res.set_title("D")
ax_alpha = fig.add_subplot(223)
ax_alpha.set_ylabel(r"$\overline{\alpha}$")
ax_alpha.set_xlabel("Species richness")
ax_alpha.set_title("C")
labels = ["species {}".format(i+1) for i in range(6)]
labels[0] = "Focal species"
handles = []

for i in range(len(ND)):
    ax_ND.plot(n_specs[i]*np.ones(len(ND[i])), ND[i], 'ko')
    ax_FD.plot(n_specs[i]*np.ones(len(FD[i])), FD[i], 'ko')
    ax_alpha.plot(n_specs[i]*np.ones(len(A_av[i])), A_av[i], 'ko')


ax_ND.plot(n_specs, [np.average(N) for N in ND], 'sg', markersize = 10, 
               label = "$\overline{\mathcal{N}_i}$", alpha = 0.5)
ax_FD.plot(n_specs, [np.average(F) for F in FD], 'sg', markersize = 10, 
              alpha = 0.5)
ax_alpha.plot(n_specs, [np.average(a) for a in A_av], 'sg', markersize = 10, 
              alpha = 0.5)
# plot special cases of increasing
ax_ND.plot(n_specs, ND_change[0], 'ro',
               label = "Neg. cor")
ax_ND.plot(n_specs, ND_change[1], "bo",
               label = "Pos. cor")

# plot special cases of increasing
ax_FD.plot(n_specs, FD_change[0], 'ro',
               label = "Increasing $\mathcal{N}_1$")
ax_FD.plot(n_specs, FD_change[1], "bo",
               label = "Decreasing $\mathcal{N}_1$")

# average interation strength
ax_alpha.plot(n_specs, A_change[0], 'ro',
               label = "Neg. cor")
ax_alpha.plot(n_specs, A_change[1], "bo",
               label = "Pos. cor")

ax_ND.legend()

alphas = 6*[0.5]
alphas[0] = 1
for i in range(len(n_specs)):
    ax_res.set_ylim([0,1.5])
    ax_res.set_xlim(pi.resources[[0,-1]])
    ax_res.set_xticks([])
    ax_res.set_yticks([])
    handles = []
    for j in range(n_specs[i]):
        handles.append(ax_res.fill_between(pi.resources, pi.res_use[j],
                            alpha = alphas[j], color = pi.colors[j]))
    ax_res.fill_between(pi.resources, pi.res_use[0],
                        hatch = "/", edgecolor = pi.colors[0],
                            facecolor = [1,1,1,0],
                            linewidth = 0)
ax_res.legend(handles, labels, loc = 1)
fig.tight_layout()

fig.savefig("Figure_ap_species_packing.pdf")