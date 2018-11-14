"""
@author: J.W.Spaak
Create the plots used in the manuscript
"""

import matplotlib.pyplot as plt
import numpy as np

###############################################################################
# Example of Mac Arthur resource model with NO
fig = plt.figure()
bars = np.array([0.1,0.2,0.3,0.4,0.5,0.4,0.3,0.2,0.1])
x = np.arange(len(bars))

bar_edges = np.array([[0,0,0.8,0.8],[0,1,1,0]])

# draw lines of to high total resource use
#for i in range(len(bars)):
#    plt.plot(x[i]+bar_edges[0], bars[i]+0.5*bars[i]*bar_edges[1], "r--")

# barplots of dependence on limiting factor
plt.bar(x,bars, color = "red", alpha = 0.5)
plt.bar(x+4,bars, color = "blue", alpha = 0.5)

fs = 14
#ax[0].text(6.4,0.04,"Niche\nOverlap", ha = "center",fontsize = fs-2,
#        bbox = {"facecolor": "blueviolet","boxstyle": "round"})

plt.xticks([])
plt.yticks([])

plt.xlabel("Limiting factor", fontsize = fs)
plt.ylabel("Dependence on limiting factor", fontsize = fs)

plt.axis([-0.3,13.1,None,0.6])

fig.savefig("Figure, Limiting factors.pdf")

###############################################################################
# Extended regions
fig = plt.figure(figsize = (7,7))

x = np.linspace(-1,3,101)
plt.plot(x,x/(1-x), "black")
plt.axis([x[0],2,-1,4])
plt.ylabel(r"$-\mathcal{F}$", fontsize = fs)
plt.xlabel(r"$\mathcal{N}$", fontsize = fs)

plt.axhline(y=0, color = "black", linestyle = ":")
plt.axhline(y=-1, color = "black", linestyle = ":")
plt.axvline(x=0, color = "black", linestyle = ":")
plt.axvline(x=1, color = "black", linestyle = ":")

plt.xticks([0,1])
plt.yticks([0,-1])

# plot the varius definitions
plt.plot([-0.5,-0.5], [-0.1,0.1], 'o', 
         color = "black", label = "priority effects")

plt.plot([0,0], [0,0], '^', 
         color = "black", label = "neutrality")

plt.plot([0.2,0.2], [-0.3,3], 'v', color = "black", label = "comp. exclusion")

plt.plot([0.6,0.6], [-0.5,1.2], '*', color = "black", label = "stable coex.")

plt.plot([1.2,0.8], [2,-0.8], '+', markersize = 15,
         color = "black", label = "parasitism")

plt.plot([1.4,1.4], [-0.9,1.7], 's', 
         color = "black", label = "mutualism")





coex_x = np.append(np.linspace(x[0],1-1e-3),x[[-1,-1,0]])
coex_y = coex_x/(1-coex_x)
coex_y[-1] = -1
coex_y[np.isinf(coex_y)] = 10**10
plt.fill(coex_x,coex_y, color = "grey",alpha = 0.5)

plt.legend(numpoints = 1)

fig.savefig("Extended Coexistence region.pdf")
