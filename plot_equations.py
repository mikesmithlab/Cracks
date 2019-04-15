import numpy as np
from plotting import Plotter

a=11E-9
L = 1000 # Length of wet region
P_cap = 2*72E-3/a

x = np.linspace(0,L,100)
print(x)

sigma_0 = P_cap*(x/L)

r_tip =

plt = Plotter(subplot=(2, 1),sharey = True)
plt.add_plot(x, sigma_0, marker='r-')
plt.show_figure()