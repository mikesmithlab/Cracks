

#Real imports
import numpy as np
from numpy.linalg import eig, inv
from numpy import shape, sin, cos, pi
import cv2
import matplotlib.pyplot as plt



class EllipticalFitter():
    def __init__(self, xdata=None, ydata=None):
        if xdata is not None:
            self.add_fit_data(xdata, ydata)

    def add_fit_data(self,xdata, ydata):
        self.xdata = xdata
        self.ydata = ydata

    def fit(self):
        x = self.xdata[:,np.newaxis]
        y = self.ydata[:,np.newaxis]
        D =  np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
        S = np.dot(D.T,D)
        C = np.zeros([6,6])
        C[0,2] = C[2,0] = 2; C[1,1] = -1
        E, V =  eig(np.dot(inv(S), C))
        n = np.argmax(np.abs(E))
        self.a = V[:,n]
        self.find_center(self.a)
        self.find_axis_length(self.a)
        self.find_angle_of_rotation(self.a)

        print('Fit Params: Center, (major,minor), angle')
        print(self.centre, self.majorax, self.minorax, self.angle)

    def find_center(self,a):
        b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
        num = b*b-a*c
        x0=(c*d-b*f)/num
        y0=(a*f-b*d)/num
        self.centre = (x0,y0)

    def find_axis_length(self, a ):
        b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
        up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
        down1=(b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
        down2=(b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
        self.res1=np.sqrt(up/down1)
        self.res2=np.sqrt(up.real/down2.real)

        if self.res1 >= self.res2:
            self.majorax = self.res1
            self.minorax = self.res2
        else:
            self.majorax = self.res2
            self.minorax = self.res1

    def find_angle_of_rotation(self, a):
        b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
        if b == 0:
            if a > c:
                self.rotation = 0
            else:
                self.rotation = np.pi/2
        else:
            self.rotation = np.arctan(2*b/(a-c))/2
            if self.res1 == self.majorax:
                self.angle = (self.rotation + np.pi/2)
            else:
                self.angle = (self.rotation)

    def generate_fitpts(self):
        u = self.centre[0]  # x-position of the center
        v = self.centre[1]  # y-position of the center
        b = self.majorax  # radius on the x-axis
        a = self.minorax  # radius on the y-axis
        rot = self.angle
        t = np.linspace(0, 2 * pi, 100)
        Ell = np.array([a * np.cos(t), b * np.sin(t)])
        # u,v removed to keep the same center location
        R_rot = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
        # 2-D rotation matrix

        Ell_rot = np.zeros((2, Ell.shape[1]))
        for i in range(Ell.shape[1]):
            Ell_rot[:, i] = np.dot(R_rot, Ell[:, i])

        self.fitx = u + Ell_rot[0,:]
        self.fity = v + Ell_rot[1,:]

    def plot_fit(self):
        plt.figure(1)
        plt.plot(self.xdata,self.ydata,'rx')
        try:
            plt.plot(self.fitx, self.fity, 'b-')
        except:
            self.generate_fitpts()
            plt.plot(self.fitx, self.fity, 'b-')
        plt.show()

    def superimpose_img(self, img, show=False):
        sz = np.shape(img)
        img = cv2.ellipse(img, (int(self.centre[0]),int(self.centre[1])), (int(self.majorax),int(self.minorax)), self.angle, 0,360, color=(0,0,255),thickness=3)
        if show:
            show_frame(resize_frame(img))
        return img


class CircleFitter(EllipticalFitter):
    def __init__(self, xdata=None, ydata=None):
        EllipticalFitter.__init__(self, xdata=None, ydata=None)

    def fit(self):
        pass


def resize_frame(frame, percent=25):
    width = np.shape(frame)[0]
    height = np.shape(frame)[1]

    dim = (int(height * percent / 100), int(width * percent / 100))

    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

def show_frame(frame):
    cv2.imshow('img', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    a=1
    b=5
    t=np.linspace(0,np.pi, 180)
    x=a*np.cos(t) + 0.3*np.random.uniform(-1,1,np.size(t))
    y=b*np.sin(t)+ 0.3*np.random.uniform(-1,1,np.size(t))



    ellipse = EllipticalFitter()
    ellipse.add_fit_data(x, y)
    ellipse.fit()
    ellipse.generate_fitpts()
    ellipse.plot_fit()



