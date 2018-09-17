"""Compute the ND over time for a multiple specie LV system"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from numerical_NFD import find_NFD

# here we choose the number of species
dim = 3
# creation of interaction matrix
A = np.random.uniform(0.5,1,size = (dim,dim))
np.fill_diagonal(A, np.random.uniform(dim,dim+1,dim))
A = 0.5*(A+A.T)

# differential equation
dNdt = lambda N,t = 0: N*(1-np.einsum('ij,...j->...i',A,N))
# per capita growth rate
f = lambda N,t = 0: 1-np.einsum('ij,...j->...i',A,N)

time = np.linspace(0,100,50)
ind_t = range(0,len(time),5*len(time)//int(time[-1]))
# solve the differential equation
N_t = odeint(dNdt,np.full(dim,0.01),time)


###############################################################################
# plot the results obtained manually

# plot densities over time
fig, ax = plt.subplots(3,2, figsize = (8,8), sharex = True)
ax[0,0].set_title("Manually computed")
ax[0,0].plot(time, N_t)
ax[0,0].plot(time[ind_t], N_t[ind_t], '.', color = "black")
ax[0,0].set_xlabel("Time")
ax[0,0].set_ylabel("Densities")

# compute NO over time
j,i = np.meshgrid(np.arange(dim), np.arange(dim))
NO_ij = np.sqrt(A[i,j]*A[j,i]/(A[i,i]*A[j,j]))
np.fill_diagonal(NO_ij,0)

c_ij = np.sqrt(A[j,j]*A[i,j]/(A[i,i]*A[j,i]))
np.fill_diagonal(c_ij,0)

A_prime = A.copy()
np.fill_diagonal(A_prime, 0)      
NO_t = np.dot(A_prime,N_t.T).T/(np.diagonal(A)*np.dot(c_ij,N_t.T).T)

ax[2,0].plot(time, NO_t)
ax[2,0].plot(time[ind_t], NO_t[ind_t], '.', color = "black")
ax[2,0].set_xlabel("Time")
ax[2,0].set_ylabel("NO")

np.fill_diagonal(c_ij,1)
FD_t = (1 - np.diagonal(A)*np.dot(c_ij,N_t.T).T)/(1-np.diagonal(A)*N_t)
ax[1,0].plot(time, FD_t)
ax[1,0].plot(time[ind_t], FD_t[ind_t], '.', color = "black")
ax[1,0].set_xlabel("Time")
ax[1,0].set_ylabel("FD")

t_end = 50
ax[0,0].set_xlim(0,t_end)
ax[0,1].set_xlim(0,t_end)


###############################################################################
# plot the results obtained by numerical solution

# plot the densities for comparison
ax[0,1].set_title("Software computed")
ax[0,1].plot(time, N_t)
ax[0,1].plot(time[ind_t], N_t[ind_t], '.', color = "black")
ax[0,1].set_xlabel("Time")
ax[0,1].set_ylabel("Densities")

NO_num = np.empty((len(time),dim))
FD_num = np.empty((len(time),dim))

for i in range(len(time)):
    pars = find_NFD(f, n_spec = dim,force = True,
                   pars = {"N_star": N_t[i]*np.ones((dim,dim))})
    NO_num[i] = pars["NO"]
    FD_num[i] = pars["FD"]

ax[1,1].plot(time, FD_num)
ax[1,1].plot(time[ind_t], FD_num[ind_t], '.', color = "black")

ax[2,1].plot(time, NO_num)
ax[2,1].plot(time[ind_t], NO_num[ind_t], '.', color = "black")   