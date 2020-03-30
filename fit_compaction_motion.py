from Generic.fitting import Fit
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    #filename= '/media/ppzmis/data/Cracks/2020_03_17/25mMnacl5fps_2020-03-17-110622-0000'+ 'pos_compactionb.txt'

    filename = '/media/ppzmis/data/Cracks/2020_02_14/2020_02_14Newno_salt_20uL_A_5fps_2020-02-14-115652-0000' + 'pos_compactionb.txt'
    data=np.loadtxt(filename)
    t = data[:,0]#[:-10,0]
    #4 comes from the 25% resize of frame used in measure_compaction
    x = 4*data[:,1]

    f = Fit('linear',x=t, y=x)
    f.add_params()
    f.fit()
    f.plot_fit()
    print(f.fit_params)


    fit_vals_2020_03_17_25mM_pixels = [10.19369458, 1133.46470945]
    fit_vals_2020_02_14_0mM_pixels = [10.15294118, 1319.38117646]


