
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from NFD_code.numerical_NFD import NFD_model, InputError

n = 3
n_com = 100

mu = 1
alpha,beta,gamma = 0.1,0.1,0.1
A = alpha * np.random.normal(size = (n_com, n,n))
B = beta * np.random.normal(size = (n_com, n,n,n))
C = gamma * np.random.normal(size = (n_com, n,n,n,n))

alpha,beta,gamma = 0.01,0.01,0.01
A = -alpha * np.random.uniform(0,1,size = (n_com, n,n))
B = -beta * np.random.uniform(0,1,size = (n_com, n,n,n))
C = -gamma * np.random.uniform(0,1,size = (n_com, n,n,n,n))
A[:,np.diag_indices(n)[0], np.diag_indices(n)[1]] = -1

A_def = np.zeros((n,n))
B_def = np.zeros((n,n,n))
C_def = np.zeros((n,n,n,n))

def interaction(T,N,d):
    return np.tensordot(T,tensor(N,d-1),axes = d-1)

def tensor(N,d):
    return np.prod([N.reshape((-1,)+i*(1,))*np.ones(d*(len(N),)) 
                    for i in range(d)], axis = 0)

def LV_model(N, A = A_def, B = B_def, C = C_def):
    return mu + interaction(A,N,2) + interaction(B,N,3) + interaction(C,N,4)


time = np.linspace(0,50,100)
index = np.full(n_com,False, dtype = "bool")

NO_norm, NO_ord3,NO_ord4 = np.empty((3,n_com,n))
FD_norm, FD_ord3,FD_ord4 = np.empty((3,n_com,n))
for i in range(n_com):
    equi_start = np.linalg.solve(A[i],np.ones(n)/n)
    try:
        pars_norm = NFD_model(LV_model,n,args = (A[i],))
        pars_ord3 = NFD_model(LV_model,n,args = (A[i],B[i]))
        pars_ord4 = NFD_model(LV_model,n,args = (A[i],B[i],C[i]))
        index[i] = True
    except InputError:
        continue
    print(i)
    NO_norm[i] = pars_norm["NO"]
    NO_ord3[i] = pars_ord3["NO"]
    NO_ord4[i] = pars_ord4["NO"]
    
    FD_norm[i] = pars_norm["FD"]
    FD_ord3[i] = pars_ord3["FD"]
    FD_ord4[i] = pars_ord4["FD"]
    
NO_norm, NO_ord3,NO_ord4 = NO_norm[index], NO_ord3[index], NO_ord4[index]
FD_norm, FD_ord3,FD_ord4 = FD_norm[index], FD_ord3[index], FD_ord4[index]

plt.boxplot([NO_norm[:,0], NO_ord3[:,0], NO_ord4[:,0]], showfliers = False)

plt.boxplot([FD_norm[:,0], FD_ord3[:,0], FD_ord4[:,0]], showfliers = False)
