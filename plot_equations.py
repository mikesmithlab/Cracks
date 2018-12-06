import numpy as np
import plotting

phi_0 = 0.68
v = 1
E = 1
h=1E-5
A = 1
B = 1
zeta = 1


x = np.linspace(0,250,0.1)


P_x = -mu_over_k*((v*x*(phi - phi_0)/phi_0) - (E*x**2)/(2*h))

r_p = (np.pi/8)*(K/sigma_ys)

sigma_ys = (phi**2/a)*(A-B*zeta)


plt = plotting.Plotter(x,P_x)