from Generic.fitting import Fit
from Generic.filedialogs import load_filename
import numpy as np

if __name__ == '__main__':

    filename = load_filename(directory='/media/ppzmis/SAMSUNG/crackbranching/CrackHoppingVids', file_filter='*.txt')
    time = np.loadtxt(filename, usecols=0)
    width = np.loadtxt(filename, usecols=1)

    shift_time = 133
    shift_width = 40
    time=time-shift_time
    width = width - shift_width
    indices = width > 0
    indices2 = time > 0
    indices = indices * indices2
    time = time[indices]
    width = width[indices]
    print(time)
    print(width)

    #f1 = Fit('linear', x=np.log(time), y=np.log(width))
    f1 = Fit('flipped_exponential', x=time, y=width)
    f2 = Fit('double_flipped_exponential', x=time, y=width)

    #logic1 = time > 200
    logic2 = time > 200

    #f1.plot_data()

    #Fit single exponential to long time data.
    f1.add_filter(logic2)
    f1.add_params(guess=[250, 0.01, 0, 0], lower=[0, 0, None, None], upper=[5000, 0.05, None, None])
    f1.fit()
    f1.plot_fit()





    #Fit double exponential constraining one exponential to be same as long time data

