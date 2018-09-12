"""
@author: J.W.Spaak
In this example we show all possible pitfalls while using the NFD computation

The example serves as a mathematical illustration, the model is not
necessarily biologically meaningful. 
"""

import numpy as np
from numerical_NFD import find_NFD
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# to run without error change this variable to True
no_error = True

###############################################################################
# we first define the percapita growth rates
# the (monoculture) percapita growth rates are polynomials of degree 3
# figure 1 shows the per capita growth rate for the species in monoculutre
x = np.array([[0,3,6,9],
              [0,1,2,3]])
y = np.array([[2,-0.1,1,0],
              [2,1,1.5,0]])

coef = np.array([np.polyfit(x[0],y[0],3), np.polyfit(x[1],y[1],3)]).T

# interspecific competition is linear and chosen s.t. invasion growth rate 1.5
roots = np.array([np.real(np.roots(coef[:,0])[-1]),x[1,3]])
r_i = 1.5
comp = (coef[-1]-r_i)/roots[::-1]

# the percapita growth rate of the species
def full_example(N):
    return coef[0]*N**3 +coef[1]*N**2 +coef[2]*N +coef[3] -comp*N[...,::-1]

time = np.linspace(0,50,1001)
sol = odeint(lambda N,t: fun(N)*N,[1,1],time)

# plot the funciton in monoculture
steps = 1000
end = 10
N_mono = np.zeros((steps,2))
N_mono[:,0] = np.linspace(0,end,steps)

fun = full_example
percapita_mono = np.array([full_example(N_mono)[:,0], 
                           full_example(N_mono[:,::-1])[:,1]])

# plot the per capita growth rate in monoculture
plt.figure()
plt.plot(N_mono[:,0],percapita_mono.T, label = "species")
plt.grid()
plt.axis([0,end,-2,3])
plt.legend(["species 1", "species 2"])
plt.xlabel("Density (N)", fontsize = 14)
plt.ylabel(r"per capita growth rate $f_i(N_i,0)$", fontsize = 14)
plt.title("Per capita growth rate in monoculture")
plt.show()
print("Note that species 1 has two stable monoculture equilibria.")
print("Species 2 has a local minima (not equilibrium) at 1.",
      "The algorithm can't find equilibrium density in this case.",
      "\nWe have to provide a beter starting estimate for",
      "the equilibrium density (default is 1).")
print("Finally the f are not monotone, which has to be specified by:")
print("monotone_f = False\n\n\n")


if not no_error:
    print('\033[31m'+
        "Error created for illustration, set no_error to True\n\n\n")
    # will result in InputError, as fsolve can't find equilibrium density
    find_NFD(full_example)
       
# Pass estimates of the equilibrium densities via the pars argument
# These only have to be estimates, not the actual equilibrium values

# Note: N_star_approx[0] is the equilibrium density of species 1, as species 0
# is absent
N_star_approx = np.array([[0,roots[1]],
                          [roots[0],0]], dtype = float)
# compute ND etc.
pars = find_NFD(full_example, pars = {"N_star": N_star_approx.copy()},
                monotone_f = False)


###############################################################################
# show potential other solutions to ND1 = ND2

c = 5**np.linspace(-3,1,101)
Nc = np.zeros((len(c),2))
Nc[:,0] = c*pars["N_star"][0,1]
NO_0 = (pars["f0"][0]-pars["r_i"][0])/(pars["f0"][0]-full_example(Nc)[:,0])


Nc = np.zeros((len(c),2))
Nc[:,1] = 1/c*pars["N_star"][1,0]
NO_1 = (pars["f0"][1]-pars["r_i"][1])/(pars["f0"][1]-full_example(Nc)[:,1])

# plot ND_1 = ND_2
plt.figure()
plt.title(r"Solve $ND_1 = ND_2$")
plt.xlabel(r"$c_1$", fontsize = 14)
plt.ylabel(r"$ND_i(c_1)$", fontsize = 14)

plt.plot(c,1-NO_0, label = r"$ND_1$")
plt.plot(c,1-NO_1, label = r"$ND_2$")
plt.grid()
plt.axis([0,4,-0.5,1.5])
plt.legend(fontsize = 14)
plt.show()

print("To find ND we have to solve ND1 = ND2.")
print("This equation has several solutions (as f is not monoton).")
print("Default will assume c=1 and solve the equation.")
print("To find other solutions we can pass the key 'c' to pars.\n\n\n")

###############################################################################
# Now we want to find all possible solutions

