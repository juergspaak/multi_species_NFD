"""
@author: J.W.Spaak
Fit different annual plant models through the same data and compare the
computed ND and FD values
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import curve_fit

from NFD_code.numerical_NFD import NFD_model
#from NFD_code.NFD_for_experiments import NFD_experiment

###############################################################################
# create the differential equation system
n_spec = 2 # number of species in the system

# Lotka-Volterra model
A = np.random.uniform(0.5,1,(n_spec,n_spec)) # interaction matrix
np.fill_diagonal(A,1) # to ensure coexistence
mu = np.random.uniform(0.5,1,n_spec) # intrinsic growth rate
def test_f(N,t = 0):
    return mu - np.dot(A,N)

# create dataset for monoculture
time = time_exp1 = time_exp2 = np.linspace(0,20,20)
dt = time[1]-time[0]
dens_exp1, dens_exp2 = np.empty((2,2,len(time_exp1)))

# densities during the first experiment
dens_exp1[0] = odeint(lambda N,t: N*test_f(N),[1e-3,0],time_exp1)[:,0]
dens_exp1[1] = odeint(lambda N,t: N*test_f(N),[0,1e-3],time_exp1)[:,1]

# densities during the second experiment
dens_exp2[0] = odeint(lambda N,t: N*test_f(N),[2*dens_exp1[0,-1],0]
                            ,time_exp1)[:,0]
dens_exp2[1] = odeint(lambda N,t: N*test_f(N),[0,2*dens_exp1[1,-1]]
                            ,time_exp1)[:,1]

# invasion growth rate (measured)
N_star = mu/np.diag(A)
r_i = mu-N_star[::-1]*A[[0,1],[1,0]]
r_i_discrete = np.exp(r_i)
###############################################################################


def fitting(model,exp1, exp2, time, p_start = [2,1]):
    # fit the annual plant parameters through the model data
    
    def help_fun(time, lamb,A, exp_init):
        A = np.array([A])
        exp1_est = exp_init*np.ones(len(time))
        for i in range(len(time)-1):
            exp1_est[i+1] = exp1_est[i]*model(exp1_est[i],lamb,A)
        return exp1_est
    
    # variable to fit
    A = np.empty((2,2))
    lamb = np.empty(2)
    # fit the first species
    [lamb[0], A[0,0]], pcov = curve_fit(lambda time, lamb,A: 
                help_fun(time, lamb, A, exp1[0,0]), time, exp1[0], p_start)
    # fit the second species
    [lamb[1], A[1,1]], pcov = curve_fit(lambda time, lamb,A: 
                help_fun(time, lamb, A, exp1[1,0]), time, exp1[1], p_start)
    # the interspecific interaction strength is not fitted yet!
    return lamb,A

def solve_difference_equation(model, start,time, par):
    # solve the difference equation, equal to ode, but time discrete
    sol = start*np.ones((len(time),len(start)))
    for i in range(len(time)-1): # iterative for next time step
        sol[i+1] = sol[i]*model(sol[i], *par)
    return sol

# the three different annual plant models according to Levine 2009
model_stand = lambda N,lamb,A: lamb/(1+A.dot(N))
model_exp = lambda N,lamb,A: lamb*np.exp(-A.dot(N))
model_log_exp = lambda N,lamb,A: lamb*np.exp(-A.dot(np.log(1+N)))

# A_ij fits the interspecific interaction for the different models
def A_ij_stand(lamb,A,r_i):
    return (lamb/r_i_discrete-1)/((lamb-1)/A.diagonal())[::-1]

def A_ij_exp(lamb,A,r_i):
    return (np.log(lamb)-r_i)/(np.log(lamb)/A.diagonal())[::-1]

def A_ij_log_exp(lamb,A,r_i):
    return (np.log(lamb)-r_i)/((np.log(lamb)/A.diagonal())[::-1])

# start the actual fitting, save variables:
# all models
models = {"stand": model_stand, "exp": model_exp, "log_exp": model_log_exp}
# how to fit the interspecific interaction
A_ij = {"stand": A_ij_stand, "exp": A_ij_exp, "log_exp": A_ij_log_exp}
args = {} # the parameters for the models
exp1_sp1 = {} # fitted densities for exp1, sp1
exp1_sp2 = {} # fitted densities for exp1, sp2
comp = {} # predicted densities during competition
# color for plotting
color = {"stand": "red", "exp": "blue", "log_exp": "green", "original": "black"
         , "experiment": "orange"}
# start density and time for competition experiment
start_comp = np.array([1e-3,1e-3])
time_comp = np.linspace(0,20,20)

# fit each model and compute densities during exp2, exp1 and comp
for key in models.keys():
    args[key] = fitting(models[key], dens_exp1, dens_exp2, time)
    args[key][1][[0,1],[1,0]] = A_ij[key](*args[key],r_i)
    exp1_sp1[key] = solve_difference_equation(models[key],
        np.array([dens_exp1[0,0],0]), time, args[key])
    exp1_sp2[key] = solve_difference_equation(models[key], 
        np.array([0,dens_exp1[1,0]]), time, args[key])
    comp[key] = solve_difference_equation(models[key], 
        start_comp, time_comp, args[key])

###############################################################################
# compute the NFD parameters
pars = {}
for key in models.keys(): 
    pars[key] = NFD_model(lambda N: np.log(models[key](N,*args[key])))

# add correct and fitted through experimental data
pars["original"] = NFD_model(test_f)
pars["experiment"] = NFD_experiment(N_star, time_exp1, dens_exp1,
                    time_exp2, dens_exp2, r_i,  visualize = False)
###############################################################################
# plot the results

# plot experimental densities and fitted densities through experimental data
fig = plt.figure(figsize = (9,9))
ax = 3*[None]
ax[0] = fig.add_subplot(2,2,1) # exp1, sp1
ax[1] = fig.add_subplot(2,2,2) # exp1, sp2
ax[2] = fig.add_subplot(2,1,2) # predicted competition experiment
ax[0].plot(time, dens_exp1[0], 'o', label = "experiment", 
  color = color["experiment"])
ax[1].plot(time, dens_exp1[1], 'o', color = color["experiment"])
ax[2].plot(time_comp, odeint(lambda N,t: N*test_f(N),start_comp,time_comp), 
      "black", linewidth = 3, label = "original", color = color["original"])

for key in models.keys():
    ax[0].plot(time, exp1_sp1[key][:,0], label = key, color = color[key])
    ax[1].plot(time, exp1_sp2[key][:,1], label = key, color = color[key])
    ax[2].plot(time_comp, comp[key], label = key, color = color[key])
    
ax[0].legend(loc = "best")
ax[0].set_title("Exp1, species 1")
ax[1].set_title("Exp1, species 2")
ax[2].set_title("Competition")

# plot the resulting ND and FD as well as the per capita growth rate
fig = plt.figure(figsize = (9,9))

ax_coex = fig.add_subplot(2,1,1) # plot the values for ND and FD
for key in pars.keys():
    ax_coex.plot(pars[key]["ND"], pars[key]["FD"], 'o', color = color[key],
                 label = key)
ax_coex.legend()
ax_coex.set_xlim([0,1])
ax_coex.set_ylim([-2,1])
ax_coex.set_xlabel(r"$\mathcal{N}$", fontsize = 14)
ax_coex.set_ylabel(r"$\mathcal{F}$", fontsize = 14)

ax_f1 = fig.add_subplot(2,2,3) # per capita growth rate of species 1
ax_f2 = fig.add_subplot(2,2,4) # per capita growth rate of species 2

N1 = np.linspace(dens_exp1[0,0], dens_exp2[0,0],100)
N2 = np.linspace(dens_exp1[1,0], dens_exp2[1,0],100)

alpha = 0.7
for key in pars.keys():
    par_c = pars[key]
    ax_f1.plot(N1,[par_c["f"](np.array([N,0]))[0] for N in N1], 
               color = color[key], alpha = alpha)
    c_N_star = (par_c["c"]*par_c["N_star"])[[0,1],[1,0]]
    ax_f1.plot(c_N_star[0], par_c["f"](np.array([c_N_star[0],0]))[0], 'o',
               color = color[key], alpha = alpha)
    
    ax_f2.plot(N2,[par_c["f"](np.array([0,N]))[1] for N in N2], 
               color = color[key], alpha = alpha)
    ax_f2.plot(c_N_star[1], par_c["f"](np.array([0,c_N_star[1]]))[1], 'o',
               color = color[key], alpha = alpha)
    
# axis labeling
ax_f1.set_title("Species 1, per capita growth")
ax_f2.set_title("Species 2, per capita growth")

ax_f1.set_ylabel(r"Percapita growth rate $f_i(N_i,0)$")
ax_f1.set_xlabel(r" Density $N_1$")
ax_f2.set_xlabel(r" Density $N_2$")
        
ax_f1.axhline(0,linestyle = "dotted", color = "black")
ax_f1.axvline(N_star[0],linestyle = "dotted", color = "black")

ax_f2.axhline(0,linestyle = "dotted", color = "black")
ax_f2.axvline(N_star[1],linestyle = "dotted", color = "black")