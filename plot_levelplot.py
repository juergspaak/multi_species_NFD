"""level plots of the different definitions for fitness and niche
differences"""

import matplotlib.pyplot as plt
import numpy as np


FDm = 10

ND = np.linspace(0.0,1,100)
FD = np.linspace(0.0,FDm,100)[:,np.newaxis]

r_i_chesson = 1 - FD + FD*ND
r_i_adler = FD*ND
r_i_spaak3 = ND - FD
r_i_spaak4 = ND - FD + FD*ND
r_i_adler[:] = 0
r_is = np.array([r_i_chesson, r_i_adler, r_i_spaak3, r_i_spaak4])


vmin = np.nanmin(r_is)
vmax = np.nanmax(r_is)

fig, ax = plt.subplots(2,2,sharex = True, sharey =True, figsize = (9,9))

im = ax[0,0].imshow(r_i_chesson,aspect = "auto",
        origin = "lower", extent = [0,1,0,FDm], vmin = vmin, vmax = vmax)
ax[0,0].plot(ND,1/(1-ND))
ax[0,1].imshow(r_i_adler, aspect = "auto", 
        origin = "lower", extent = [0,1,0,FDm], vmin = vmin, vmax = vmax)
ax[1,0].imshow(r_i_spaak3, aspect = "auto", 
        origin = "lower", extent = [0,1,0,FDm], vmin = vmin, vmax = vmax)
ax[1,0].plot(ND,ND)
ax[1,1].plot(ND, ND/(1-ND))
ax[1,1].imshow(r_i_spaak4, aspect = "auto", 
        origin = "lower", extent = [0,1,0,FDm], vmin = vmin, vmax = vmax)

ax[1,1].set_xlim([0,1])

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)

ax[0,0].set_title("Chesson")
ax[1,0].set_title("Spaak3")
ax[1,1].set_title("Spaak4")