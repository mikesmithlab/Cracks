import numpy as np
import matplotlib.pyplot as plt


def calc_stress_tip(curvature, stress):
    return stress*(1+2*curvature)

def calc_residual_stress(x,stress_0):
    pass

def calc_plastic_zone_size(x,sigma_y, K):
    r_plastic = (1/2*np.pi)*(K/sigma_y)**2

def move_compaction(x,dx=0.0001):
    x+=dx

if __name__ == '__main__':
    x = np.linspace(0,1000,1000)*1E-6
    a = 10E-9
    L=1E-3
    stress_coeff = 72e-3/a
    stress0 = stress_coeff*(x/L)
    phi0=0.64
    d=100E-6
    phi_liquid = 0.3
    phi=(phi0-phi_liquid)*(1/(1+np.exp(-x/d)))+phi_liquid
    yield_const = 10*72E-3/a
    sigma_y = yield_const*phi**2

    #stress_tip = stress0*(1+)



    plt.figure()
    plt.plot(x,phi,'g.')
    #plt.plot(x,stress0,'r-')
    #plt.plot(x, sigma_y,'b-')
    plt.show()

