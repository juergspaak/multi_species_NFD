import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.cm import viridis
from string import ascii_uppercase as ABC

plt.rc('axes', labelsize=16)
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)
plt.rc('axes', titlesize=16)
mpl.rcParams['hatch.linewidth'] = 7.0
mpl.rcParams['hatch.color'] = "r"

# species richness
n_specs = np.arange(2,7)

# parameters for resource utilization function
mu = np.array([1,1,0.5, 1.3, 0.5,1.4]) # peak of gaussian
mean_cons = np.array([0.42, 0.58, 0.5, 0.2, 0.85, 0.35]) # mean of gaussian
var_cons = np.array([0.1,0.1,0.4,0.05,0.5,0.05]) # variance of gaussian

resources = np.linspace(0,1.1,1000)
# compute resource utilization
res_use = (mu*np.exp(-np.abs((mean_cons-resources[:,np.newaxis])/var_cons)**2)).T    

# compute intrinsic growth rates and species interactions
mu_grow = np.sum(res_use, axis = 1)
A = np.sum(res_use*res_use[:,np.newaxis], axis = -1)
colors = viridis(np.linspace(0,1,n_specs[-1]))[::-1] # colors of species
###############################################################################
# start plotting
if __name__ == "__main__":
    fig = plt.figure(figsize = (10,10))
   
    alpha = 0.7 # transparancy
    
    
    
    ###########################################################################
    # resource consumption graph
    labels = ["species {}".format(i+1) for i in range(6)]
    labels[0] = "Focal species"
    handles = []
    alphas = 6*[alpha]
    alphas[0] = 1
    for i in range(len(n_specs)):
        # plot the resoruce use of the species in all the communities
        ax_res = fig.add_subplot(len(n_specs),2,2*i+1)
        ax_res.set_title(ABC[i])
        ax_res.set_ylim([0,1.5])
        ax_res.set_xlim(resources[[0,-1]])
        ax_res.set_xticks([])
        ax_res.set_yticks([])
        handles = []
        for j in range(n_specs[i]):
            handles.append(ax_res.fill_between(resources, res_use[j],
                                alpha = alphas[j], color = colors[j]))
        ax_res.fill_between(resources, res_use[0],
                            hatch = "/", edgecolor = colors[0],
                                facecolor = [1,1,1,0],
                                linewidth = 0)
    
        if i == 0:
            ax_foc = ax_res
    # add legend in first plot
    leg1 = ax_foc.legend([handles[0]], ["Focal species"], loc = 2)
    ax_foc.legend(handles[1:], ["species {}".format(i)
                    for i in range(2,n_specs[-1]+1)], loc = 1)
    ax_foc.add_artist(leg1)
    
    # add axis labels
    ax_label = fig.add_subplot(1,3,1, frameon = False)
    ax_label.tick_params(labelcolor='none',
                         top=False, bottom=False, left=False, right=False)
    ax_label.set_ylabel("resource utilisation\n$A_j$", visible = "True")
    ax_label.set_xlabel("resources", visible = True)
    
    ###########################################################################
    # plot panel F
    ax_tot = fig.add_subplot(3,2,2)
    bar_mu =  np.tril(mu_grow*np.ones((n_specs[-1], n_specs[-1]))).T
    bar_mu[0] = 0
    bar_mu = bar_mu[:,1:]
    for j in range(len(n_specs)):
        ax_tot.bar(n_specs, bar_mu[j+1],
                bottom = np.sum(bar_mu[:n_specs[j]-1], axis = 0),
                color = colors[j+1], alpha = alpha,
                label = "species {}".format(n_specs[j]))
    ax_tot.set_xticks([])
    ax_tot.set_ylabel("Total utilisation\n$\sum\Vert A_j\Vert$")
    ax_tot.set_title("F")
    ax_tot.set_yticks([])
        
    
    # plot panel G
    ax_scal = fig.add_subplot(3,2,4)
    bar_ov =  np.tril(A[0]*np.ones((n_specs[-1], n_specs[-1]))).T
    bar_ov[0] = 0
    bar_ov = bar_ov[:,1:]
    for j in range(len(n_specs)):
        ax_scal.bar(n_specs, bar_ov[j+1],
                bottom = np.sum(bar_ov[:n_specs[j]-1], axis = 0),
                color = colors[j+1], alpha = alpha,
                label = "species {}".format(n_specs[j]))

    ax_scal.set_xticks([])
    ax_scal.set_ylabel("Utilisation overlap\n$\sum\Vert A_j\cap A_1\Vert$")
    ax_scal.set_title("G")
    ax_scal.set_yticks([])
        
    # plot panel H
    ax_ND = fig.add_subplot(3,2,6)
    bar_ND = bar_ov/np.sum(bar_mu, axis = 0, keepdims = True)
    for j in range(len(n_specs)):
        ax_ND.bar(n_specs, bar_ND[j+1],
                bottom = np.sum(bar_ND[:n_specs[j]-1], axis = 0),
                color = colors[j+1], alpha = alpha)
    ax_ND.set_xticks([])
    ax_ND.set_ylabel("Relative overlap\n$\sum\Vert$"
                     "$A_1\cap A_j\Vert/\sum\Vert A_j\Vert$")
    ax_ND.set_title("H")
    ax_ND.set_xticks(n_specs)
    ax_ND.set_yticks([])
    ax_ND.set_xlabel("species richness")
    
    fig.savefig("Figure_1.pdf")