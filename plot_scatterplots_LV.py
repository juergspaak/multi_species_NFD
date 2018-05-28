"""scatter plot of the different defintions for ND and FD for the annual
plant model"""

import matplotlib.pyplot as plt
import numpy as np

eps = 0.05
n = 1000000
alpha = np.random.uniform(eps,1-eps,size = (2,2,n))
# remove alphas combinations with alpha[i,i]<alpha[i,j]

good = (alpha[0,0]>alpha[0,1]) * (alpha[1,1]>alpha[1,0])
alpha = alpha[...,good]
n = alpha.shape[-1]
lamb = np.random.uniform(2,10,(2,n))

i,r = np.array([0,1]), np.array([1,0])
r_i = np.log(lamb[i]/(1+alpha[i,r]/alpha[r,r]*(lamb[r]-1)))
col = r_i/np.log(lamb)
vmin, vmax = np.percentile(col,[1,99])
vmax = np.max(np.abs([vmin,vmax]))
vmin = -vmax
germain = np.array([1-np.sqrt(alpha[i,r]*alpha[r,i]/(alpha[i,i]*alpha[r,r])),
                    (lamb[r]-1)/(lamb[i]-1)*np.sqrt(alpha[i,r]*alpha[i,i]/
                    (alpha[r,r]*alpha[r,i]))])

adler = np.array([lamb[r]/(1+alpha[i,r]/alpha[r,r]*(lamb[r]-1)),
                  lamb[i]/lamb[r]])

#Compute Caroll's definition
g_i_0 = np.log(lamb)
S_i = (g_i_0-r_i)/g_i_0
# geometric mean and geometric variance
caroll = np.exp([np.mean(np.log(S_i),axis = 0),
                 np.sqrt(np.var(np.log(S_i),axis = 0)),])
caroll[0] = 1-caroll[0]


# Compute own definition
f_N = np.log(lamb[i]/(1+alpha[i,i]/alpha[r,r]*(lamb[r]-1)))
f_0 = np.log(lamb)

spaak = np.array([(r_i-f_N)/(f_0-f_N),
                  -f_N/f_0,])

fig, ax = plt.subplots(2,2,figsize = (9,9))

def scatter(data, axis, color,FD_fun):
    im = axis.scatter(*data,c = color, s = 4, linewidths=0, vmin = vmin,
                      vmax = vmax, cmap = "RdBu")
    ND = np.linspace(*np.percentile(data[0],[0,100]),100)
    FD = FD_fun(ND)
    axis.plot(ND, FD)
    axis.patch.set_facecolor("lightgrey")
    return im
# scatter plots of ND, FD and r_i
scatter(germain[:,0],ax[0,0],col[0],lambda x: 1/(1-x))
scatter(adler[:,0],ax[0,1],col[0],lambda x: 1/x)
scatter(caroll,ax[1,0],np.amin(col,axis = 0),lambda x: 1/(1-x))
im = scatter(spaak[:,0],ax[1,1],col[0],lambda x: x/(1-x))

# add colorbar
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)

# add titles
ax[0,0].set_title("Germain (=Chesson)")
ax[0,1].set_title("Adler")
ax[1,0].set_title("Caroll")
ax[1,1].set_title("Spaak")

ax[1,1].set_xlabel("Niche difference")
ax[0,0].set_ylabel("Fitness diffrence")

ax[0,0].set_ylim([0,10])
ax[0,1].set_ylim([0,5])
ax[0,1].set_xlim([0,7])
ax[1,0].set_ylim([0,5])
ax[1,1].set_xlim([0,1])
ax[1,1].set_ylim([-1,5])

fig.savefig("Different definitions of ND and FD.png")


# add a plot to compare Germain and Caroll
fig2,ax2 = plt.subplots(2,2, figsize = (9,9))
ax2[0,0].scatter(germain[0,0],caroll[0],c = col[0], s = 4, linewidths = 0,
                vmin = vmin, vmax = vmax)
ax2[0,1].scatter(np.amin(germain[0],axis=0),caroll[0],c = np.amin(col,axis = 0)
                , s = 4, linewidths = 0,vmin = vmin, vmax = vmax)
ax2[1,0].scatter(germain[1,0],caroll[1],c = col[0], s = 4, linewidths = 0,
                vmin = vmin, vmax = vmax)
im = ax2[1,1].scatter(np.amin(germain[1],axis=0),caroll[1],c = np.amin(col,axis = 0)
                , s = 4, linewidths = 0,vmin = vmin, vmax = vmax)

# add colorbar
fig2.subplots_adjust(right=0.8)
cbar_ax = fig2.add_axes([0.85, 0.15, 0.05, 0.7])
fig2.colorbar(im, cax=cbar_ax)

ax2[0,0].set_title("random species comparison")
ax2[0,1].set_title("inferior species")

ax2[0,0].set_ylabel("ND caroll")
ax2[1,0].set_ylabel("FD caroll")

ax2[1,0].set_xlabel("FD germain")
ax2[0,0].set_xlabel("ND germain")
fig2.savefig("Compare germain and Caroll.png")