import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

A = np.array([[1,0.1,0.1],
              [0.1,1,0.9],
              [0.1,0.9,1]])

def plot_mu(mus,*args, triag = False, **kwargs):
    mus = mus/np.sum(mus, axis = 0)
    x = mus[1] + mus[-1]/2
    y = mus[-1]
    if triag:
        x = np.append(x, x[0])
        y = np.append(y, y[0])
    plt.plot(x,y, *args, **kwargs)
    return x,y

fig = plt.figure()
ax = plt.gca()
ax.set_xticks([])
ax.set_yticks([])

ax.set_frame_on(False)
plot_mu(np.eye(3), 'k', triag = True)
x,y = plot_mu(A, 'ko')

coex = Polygon(np.array([x,y]).T, closed = True,
               alpha = 0.5, color = "g")
ax.add_patch(coex)

rc = 1/len(A)*(np.sum(A/np.linalg.norm(A, axis = 1), axis = 0))
plot_mu(rc, 'ro')

itera = 5000
mu1 = np.linspace(0,1,itera)
mu2 = np.linspace(0,1-mu1,itera)
mu1 = np.repeat([mu1], itera, axis = 0)
mu3 = 1 - mu1 - mu2
mu = np.array([mu1, mu2, mu3])
good = np.abs(np.einsum("i, inm->nm", rc, mu)/np.linalg.norm(mu, axis = 0)-0.75)<1e-4

plot_mu(mu[:,good], ',b')
mu_theta = mu[:,good]
pos_equi = np.all(np.linalg.solve(A, mu_theta)>0, axis = 0)
plot_mu(mu[:,good][:,pos_equi], ',', color = "orange")

ax.text(0,-0.05,"$\mu_1$", ha = "center")
ax.text(1,-0.05,"$\mu_2$", ha = "center")
ax.text(0.5,1+0.05,"$\mu_3$", ha = "center")

fig.savefig("Figure_S7.pdf")