"""
@author: J.W.Spaak
Create the plots used in the manuscript
"""

import matplotlib.pyplot as plt
import numpy as np

###############################################################################
# Example of Mac Arthur resource model with NO
"""
r = np.linspace(0,10,11)
peaks = np.array([2.3,3.7,6.7]) # peaks of the utilization functions
g = np.exp(-(r-peaks[:,np.newaxis])**2)

fig, ax = plt.subplots(1,2, figsize = (8,3), sharex = False, sharey = True)
ax[0].plot(r,g[0], "red", r,g[1], "blue", linewidth = 2)
ax[0].plot(r,1.3*g[0], "r:")
#ax[0].plot(r, 0.5*g[0], ":r")
ax[0].fill(r,np.amin(g[:2],axis = 0), color = "purple",alpha = 0.5)
ax[0].axis([0,6,0,1.5])
ax[0].text(3,0.1,"Niche\nOverlap", ha = "center")

ax[1].plot(r, g[0], "red", r,g[1], "blue", r,g[2], "green", linewidth = 2)

ax[1].fill(r,np.amin(g[:2],axis = 0), color = "purple",alpha = 0.5)
ax[1].fill(r,np.amin(g[1:],axis = 0), color = "cyan",alpha = 0.5)

ax[0].set_xticks([])
ax[1].set_xticks([])
ax[0].set_yticks([])
ax[0].set_xlabel("Resource")
ax[1].set_xlabel("Resource")
ax[0].set_ylabel("Utilization")
ax[0].set_title("A")
ax[1].set_title("B")


fig.savefig("Figure, Utilization functions.pdf")
"""

fig ,ax = plt.subplots(1,2,figsize = (11,4), sharey = True)
bars = np.array([0.1,0.2,0.3,0.4,0.5,0.4,0.3,0.2,0.1])
x = np.arange(len(bars))

bar_edges = np.array([[0,0,0.8,0.8],[0,1,1,0]])

# draw lines of to high total resource use
for i in range(len(bars)):
    ax[0].plot(x[i]+bar_edges[0], bars[i]+0.5*bars[i]*bar_edges[1], "r--")

# barplots of dependence on limiting factor
ax[0].bar(x,bars, color = "red", alpha = 0.5)
ax[0].bar(x+4,bars, color = "blue", alpha = 0.5)

# barplots on multispecies
ax[1].bar(x,bars, color = "red", alpha = 0.5)
ax[1].bar(x+4,bars, color = "blue", alpha = 0.5)
ax[1].bar(x+10,bars, color = "green", alpha = 0.5)

fs = 14
#ax[0].text(6.4,0.04,"Niche\nOverlap", ha = "center",fontsize = fs-2,
#        bbox = {"facecolor": "blueviolet","boxstyle": "round"})

ax[0].set_xticks([])
ax[1].set_xticks([])
ax[0].set_yticks([])

ax[0].set_xlabel("Limiting factor", fontsize = fs)
ax[1].set_xlabel("Limiting factor", fontsize = fs)
ax[0].set_ylabel("Dependence on limiting factor", fontsize = fs)
ax[1].set_ylabel("Dependence on limiting factor", fontsize = fs)
ax[0].set_title("A")
ax[1].set_title("B")

ax[0].axis([-0.3,13.1,None,None])
ax[1].axis([-0.3,19.1,None,None])

fig.savefig("Figure, Limiting factors.pdf")
###############################################################################
# Linear interpolation plot

fig = plt.figure(figsize = (7,7))
plt.axes(frame_on = False)
offset = 0.2
plt.plot([0,1],[0,1],'r-o')
plt.axis([-offset,1+offset, -offset, 1+offset])

ri = 0.7
plt.plot([ri,ri,-offset],[-offset,ri,ri], '--')

fs = 16
plt.xticks([0,ri,1], [r"$f_i(c_jN_j^*,0)$",r"$f_i(0,N_j^*)$",r"$f_i(0,0)$"],
           fontsize = fs)
plt.yticks([0,ri,1], [0,r"$\mathcal{N}_i$",1],fontsize = fs)
plt.tick_params(length = 20,top = False, right = False)
arrow = 0.035
plt.arrow(-offset+arrow,-offset+arrow, 1+offset+arrow,0)
plt.arrow(-offset+arrow,-offset+arrow, 0, 1+offset+arrow)

fig.savefig("Linear interpolation.pdf")

###############################################################################
# Coexistence region

fig = plt.figure(figsize = (7,7))

x = np.linspace(0,1-1e-2,101)
plt.plot(x,x/(1-x), "black")
plt.axis([0,1,-1,5])
plt.ylabel(r"$-\mathcal{F}$", fontsize = fs)
plt.xlabel(r"$\mathcal{N}$", fontsize = fs)

no_coex_x = np.append(x,0)
no_coex_y = np.append(x/(1-x),5)
plt.fill(no_coex_x,no_coex_y, color = "red",alpha = 0.5)

coex_x = np.append(x,[1,1,0])
coex_y = np.append(x/(1-x),[6,-1,-1])
plt.fill(coex_x,coex_y, color = "green",alpha = 0.5)

plt.text(0.5,-0.2,"Coexistence", ha = "center", fontsize = fs)
plt.text(0.5,2,"Competitive\nExclusion", ha = "center", fontsize = fs)

fig.savefig("Coexistence region.pdf")