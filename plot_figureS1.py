import numpy as np
import matplotlib.pyplot as plt

from interaction_estimation import data_short, data_wide, interaction_short, interaction_wide

fig, ax = plt.subplots(2, sharex = True)
bins = np.linspace(-2,2,101)
ax[0].hist(data_short, bins = bins, density = True, color = "red")
x = np.linspace(-2,2, 501)
ax[0].plot(x,interaction_short(x), 'k', linewidth = 3)

ax[1].hist(data_wide, bins = bins, density = True,
  color = "blue")
x = np.linspace(-2,2, 501)
ax[1].plot(x,interaction_wide(x), 'k', linewidth = 3)
ax[1].set_xlim([-1.1, 1.1])
ax[1].set_xticks([-1,0,1])

# layout
ax[0].set_ylabel("density", fontsize = 14)
ax[1].set_ylabel("density", fontsize = 14)
ax[1].set_xlabel("interaction strength", fontsize = 14)

ax[0].set_title("A: Weak interactions", loc = "left")
ax[1].set_title("B: Strong interactions", loc = "left")

fig.savefig("Figure_S1.pdf")