import numpy as np
import Generic.filedialogs as fd
import matplotlib.pyplot as plt
import Generic.plotting as p



if __name__ == '__main__':
    scale=3.215E-1 #microm/pixel
    fps=2.0
    filename = fd.load_filename(directory='/media/ppzmis/data/Cracks/2020_02_14',file_filter='*.txt')
    crack_width = np.loadtxt(filename)

    #Calculate crack tip trajectory
    tip_pos=np.zeros(np.shape(crack_width[:,1]))
    rad_tip=np.zeros(np.shape(tip_pos))
    for i in range(np.shape(crack_width)[0]-1):
        index = np.max(np.where(crack_width[i,:] > 1))
        tip_pos[i]=index
        rad_tip[i]=np.max(np.sum(crack_width[:,index-50:index]))



    time = (1/fps)*np.arange(0,np.size(tip_pos))
    tip_pos = scale*tip_pos
    rad_tip = scale * rad_tip/2

    np.savetxt(filename[:-4] + '_tiptraj.txt',np.c_[time,tip_pos])



    p2 = p.Plotter(subplot=(3,1))
    #1st subplot is tip trajectory
    p2.add_plot(time, tip_pos, marker='r-',subplot=0)
    p2.configure_xaxis(xlabel='time (s)',subplot=0)
    p2.configure_yaxis(ylabel='xpos (microns)', subplot=0,fontsize=14)

    #2nd subplot is the final width profile of crack indicating where width measurements are being taken
    p2.add_plot(range(0,np.shape(crack_width)[1]),crack_width[np.shape(crack_width)[0]-10,:],marker='b-',subplot=1)
    #Show position of width measurement on this profile
    p2.configure_yaxis(ylabel='width',subplot=1,fontsize=14)
    p2.configure_yaxis(ylabel='Crack width (microns)', subplot=2,fontsize=14)
    p2.configure_xaxis(xlim=(tip_pos[0],None),subplot=1)
    add_position(p2, filename, crack_width, time, 929, 'g')
    add_position(p2, filename, crack_width, time, 1073, 'y')
    add_position(p2, filename, crack_width, time, 1241, 'k')
    add_position(p2, filename, crack_width, time, 1400, 'r')
    #plot width fn of time for particular position in lower subplot.
    #Final subplot is width at position against time.
    p2.show_figure()