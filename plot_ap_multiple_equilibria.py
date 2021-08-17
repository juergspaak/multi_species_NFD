import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp

#from higher_order_models import LV_model
from interaction_estimation import resample_short, resample_wide

def interaction(T,N,d):
    return np.tensordot(T,tensor(N,d-1),axes = d-1)

def tensor(N,d):
    return np.prod([N.reshape((-1,)+i*(1,))*np.ones(d*(len(N),)) 
                    for i in range(d)], axis = 0)

def LV_model(N, mu, A , B):
    return mu - interaction(A,N,2) - interaction(B,N,3)

def find_feasible_stable_equi(N_start, mu, A, B):
    x, info,a,b = fsolve(LV_model, N_start, args = (mu, A, B),
                     full_output=True)
    if np.any(x<0):
        print("neg")
        return np.nan
    
    # is it stable?
    n_spec = len(N_start)
    r = np.zeros((n_spec, n_spec))
    r[np.triu_indices(n_spec)] = info["r"].copy()
    jac = np.diag(x).dot(info["fjac"].T).dot(r)
    
    # not a stable equilibrium
    if np.amax(np.real(np.linalg.eigvals(jac)))>0:
        print("eigval")
        return np.nan
    
    return x

itera = 1000
HOI_equis = np.full((itera,4, 6), np.nan)
ode_equi = np.full((itera, 6), np.nan)
LV_equi = np.full((itera, 6), np.nan)
for i in range(itera):
    print(i)
    # create a community model
    n = np.random.randint(2,7)
    aij = resample_short(n*n)
    A = aij.reshape((n,n))
    A[np.arange(n), np.arange(n)] = 1
    B = np.random.uniform(-0.05, 0.05, (n,n,n))
    mu = np.ones(n)
    
    B = A[...,np.newaxis]*B
    
    
    LV_equi[i,:n] = np.linalg.solve(A, np.ones(n))
    HOI_equis[i,0,:n] = find_feasible_stable_equi(LV_equi[i,:n], mu, A, B)
    HOI_equis[i,1,:n] = find_feasible_stable_equi(np.random.uniform(0,5,n), mu, A, B)
    HOI_equis[i,2,:n] = find_feasible_stable_equi(np.random.uniform(10,20,n), mu, A, B)
    
    
    time = [0,1000]
    N_start = np.full(n, 0.1)
    sol_ode = solve_ivp(lambda t, N: N*LV_model(N, mu, A, B), time, 
                    N_start)
    N_start = sol_ode.y[:,-1]
    
    HOI_equis[i,3,:n] = fsolve(LV_model, N_start, args = (mu, A, B))
    
    ode_equi[i,:n] = N_start
    
bins = np.linspace(-0.5,0.5, 50)
fig = plt.figure()
plt.hist(((LV_equi - HOI_equis[:,0])/LV_equi).flatten(), bins = bins, density = True,
         label = "LV equilibrium")
plt.hist(((HOI_equis[:,1] - HOI_equis[:,0])/HOI_equis[:,0]).flatten(), bins = bins,
         density = True, label = "Uniform [0,5]")
plt.hist(((HOI_equis[:,2] - HOI_equis[:,0])/HOI_equis[:,0]).flatten(), bins = bins,
         density = True, label = "Uniform [10,50]")
plt.hist(((HOI_equis[:,3] - HOI_equis[:,0])/HOI_equis[:,0]).flatten(), bins = bins,
         density = True, label = "Odeint solution")

plt.legend()

plt.xlabel("Relative distance of equilibria")
plt.ylabel("Probability")

fig.savefig("Figure_ap_multiple_equilibria.pdf")