import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve


if __name__ == '__main__':


    def func(theta):
        "Crack tip radius"
        a = 1E-6
        poisson = 0.3
        r = np.arange(0.0001, 2, 0.0001)

        sigma_Y = 10E6
        stress_far = 5.3 * 72E-3 / a
        K = 1.12 * stress_far * np.sqrt(np.pi * a)

        prefactor = (K/np.sqrt(2*np.pi*r))*np.cos((theta/2)*(1+np.sin(theta/2)))
        sigma_1 = prefactor * np.cos((theta/2)*(1 + np.sin(theta/2)))
        sigma_2 = prefactor * np.cos((theta/2)*(1 - np.sin(theta/2)))
        #print(sigma_1)
        #print(sigma_2)
        "For plane stress"
        sigma_3 = 0

        "For plane strain"
        #sigma_3 = poisson * (sigma_1 + sigma_2)

        rp = K ** 2 / (2 * np.pi * sigma_Y ** 2)
        ans= np.sqrt((sigma_1 - sigma_2)**2 + (sigma_1 - sigma_3)**2 + (sigma_2-sigma_3)**2)/2
        print(ans)
        return ans


    #theta = np.arange(0, 2 * np.pi, np.pi / 180)
    initial_guess = np.array([5E-6])
    r_soln = fsolve(func, initial_guess)
    print(r_soln)


