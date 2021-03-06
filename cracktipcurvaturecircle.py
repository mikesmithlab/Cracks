from Generic.video import ReadVideo, WriteVideo
import cv2
import numpy as np
from scipy.interpolate import splprep, splev

from scipy import optimize



def resize_frame(frame,percent=25):
    percent = 25
    width = np.shape(frame)[0]
    height = np.shape(frame)[1]

    dim = (int(height * percent / 100), int(width * percent / 100))

    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

def show_frame(frame):
    cv2.imshow('img', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def threshold(frame, thresh=50):
    ret, bin_img = cv2.threshold(frame, thresh, 255, cv2.THRESH_BINARY_INV)
    return bin_img

def rotate(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

def extract_nth_biggest_object(frame, n=0):
    'stats[0,:] is the entire frame'
    output = cv2.connectedComponentsWithStats(frame, 4, cv2.CV_32S)
    labels=output[1]-1
    stats=output[2]
    stats= stats[1:][:]
    indices = np.argsort(stats[:, cv2.CC_STAT_AREA])
    indices = indices[::-1]
    im = np.zeros(np.shape(frame))
    im[labels==indices[n]] = 255
    cX, _ = find_com(im)
    if cX > 2500:
        im=extract_nth_biggest_object(frame,n=1)
        return im
    else:
        return im

def find_cracktip(crack_frame,frame, minx=400, threshold=10, behind_tip=400):
    #collapse binary image crack along y to get crack width
    sum_alongy = np.sum(crack_frame,axis=0)
    indice = np.argwhere(sum_alongy[minx:] < threshold)[0]
    #in pixels
    cracktip_x = indice + minx

    pt1 = np.mean(np.argwhere(crack_frame[:,cracktip_x-1])[:,0])
    pt2 = np.mean(np.argwhere(crack_frame[:,cracktip_x-400])[:,0])

    #Fitting crack to work out if rotation of image is corect
    grad = (pt1-pt2)/((cracktip_x - 1)-(cracktip_x-behind_tip))
    intercept = pt1 - grad*(cracktip_x-1)
    theta = np.arctan2(pt2-pt1,((cracktip_x - 1)-(cracktip_x-behind_tip)))


    sz = np.shape(crack_frame)
    #temp = cv2.line(frame.copy(), (cracktip_x, 1),(cracktip_x, sz[1]-1), (255,0,0), thickness=5)
    #temp = cv2.line(temp, (1, int(grad+intercept)),
    #                    (sz[1]-1, int((sz[1]-1)*grad+intercept)), (255,0,0), thickness=3)
    #show_frame(resize_frame(temp))
    #frame = temp

    return int(cracktip_x), int(pt1), frame


def find_cracktip_radius(crack_frame, rot_frame, cracktip_x, min_trail=10, max_trail=100):
    width = []
    trails = list(range(min_trail, max_trail,1))

    contours, hierarchy = cv2.findContours(crack_frame.astype(np.uint8),
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    cntrs = contours[0]
    sz = np.shape(cntrs)
    cntrs = cntrs.reshape(sz[0],2)
    tck, u = splprep(cntrs.T, u=None, per=1)# s=0.0,
    u_new = np.linspace(u.min(), u.max(), 1000)
    cntr_x, cntr_y = splev(u_new, tck, der=0)
    contour = np.column_stack((cntr_x.astype(int), cntr_y.astype(int)))
    sz = np.shape(contour)
    contour = contour.reshape(sz[0],1,2)


    rot_frame = cv2.drawContours(rot_frame, contour, -1,
                                           (0, 255, 0), 5)
    #show_frame(resize_frame(rot_frame))

    centres = []
    rads=[]
    errs=[]
    for trail in trails:
        a,b,c = fit_crack_contour(cntr_x, cntr_y, cracktip_x, trail)
        centres.append(a)
        rads.append(b)
        errs.append(c)




    trailmin = np.argmin(rads)
    R = rads[trailmin]
    centre = (int(centres[trailmin][0]),int(centres[trailmin][1]))

    '''
    plt.figure(1)
    plt.plot(trails, rads, 'x')
    plt.plot([trails[trailmin], trails[trailmin]], [np.min(rads), np.max(rads)], 'r')



    plt.show()
    '''


    rot_frame = cv2.drawContours(rot_frame, contour, -1,
                                           (0, 255, 0), 5)
    rot_frame = cv2.circle(rot_frame,centre, int(R), (0,0,255), thickness=3)
    #show_frame(rot_frame[cracktip_y-200:cracktip_y+200,cracktip_x - 200:cracktip_x+200,:])


    return rot_frame, R


def fit_crack_contour(cntr_x, cntr_y, cracktip_x, trailing = None, show=False):
    indices = np.argwhere((cracktip_x - cntr_x) < trailing)
    tip_x = cntr_x[indices].flatten()
    tip_y = cntr_y[indices].flatten()

    def calc_R(xc, yc):
        return np.sqrt((tip_x - xc) ** 2 + (tip_y - yc) ** 2)


    def f_2(c):
        Ri = calc_R(*c)

        return Ri - Ri.mean()

    #f = Fit('quadratic')
    #    f.add_fit_data(tip_y, tip_x)
    #    f.add_params(guess=[-1,1000,np.max(cntr_x)], lower=[None,None,'Fixed'],upper=[None,None,'Fixed'])
    #    f.fit()
    #    f.plot_fit()


    center_estimate = (cracktip_x-50), np.mean(tip_y)
    center_2, ier = optimize.leastsq(f_2, center_estimate)

    x0 = [-1, -3]

    xc_2, yc_2 = center_2
    Ri_2 = calc_R(*center_2)
    R_2 = Ri_2.mean()
    residu_2 = sum((Ri_2 - R_2) ** 2)





    return center_2, R_2, residu_2

def find_com(frame):
    '''
    send binary image with 1 object
    get com coordinates back
    '''
    # calculate moments of binary image
    M = cv2.moments(frame)

    # calculate x,y coordinate of center
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return cX,cY

def imfill(frame):
    #frame should be a thresholded image
    # Copy the thresholded image.
    im_floodfill = frame.copy()
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = frame.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255);

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    im_out = frame | im_floodfill_inv

    return im_out

def width_vals(frame):
    return np.sum(frame,axis=0)

def find_crackwidth(crack_frame, cracktip_x, distancebehind, scale=1):
    sum_alongy = scale*np.sum(crack_frame, axis=0)/255.0
    return sum_alongy[int(cracktip_x-distancebehind)][0]

def find_crackwidth_point(crack_frame, frame, points, scale=1, show=False):

    sum_alongy = scale*np.sum(frame, axis=0) / 255.0
    widths = []
    if show:
        sz = np.shape(frame)
        frametemp = frame

    for point in points:
        widths.append(sum_alongy[int(point)])
        if show:
            frametemp = cv2.line(frametemp, (int(point),1), (int(point),sz[1]), (0,255,0), thickness=3)

        #show_frame(resize_frame(frametemp))
    return widths, frametemp

compaction_front = []


def click(event, x,y,dummy,dummy2):
    global compaction_front

    if event == cv2.EVENT_LBUTTONDOWN:
        compaction_front.append(x)



def processframe(frame, angle, cutpos = 100):
    rot_frame = rotate(frame, angle)
    gray_frame = cv2.cvtColor(rot_frame, cv2.COLOR_BGR2GRAY)

    # BW image with the crack extracted from background
    crack_frame = threshold(frame[:, :, 1], thresh=crack_threshval)
    crack_frame = rotate(crack_frame, angle)

    sz = np.shape(crack_frame)
    #Set 10 columns of threshold white. allows imfill to fill whole crack
    crack_frame[5:sz[0]-5, cutpos:cutpos+5] = 1
    crack_frame = imfill(crack_frame)
    #Set same cols black. now largest opject is crack
    crack_frame[5:sz[0] - 5:, cutpos: cutpos+5] = 0
    crack_frame = extract_nth_biggest_object(crack_frame, n=0)
    return crack_frame, rot_frame, gray_frame

    #a=2.0331085, b=1313.76234616
    #Divide by 5 since per index not per second.
    #0mM - a=10.152941/5, b=1319.38
    # 25mM - 10.19369458, 1133.46470945
def find_compaction(frame, frame_index, a=10.19369/5, b=1133.464):
    compaction_x = int(a*frame_index+b)
    sz = np.shape(frame)
    frame = cv2.line(frame, (compaction_x, 1),(compaction_x, sz[1]), (255,0,0), thickness=3)
    return frame, compaction_x

if __name__ == '__main__':

    filename = '/media/ppzmis/data/Cracks/2020_03_17/25mMnacl5fps_2020-03-17-110622-0000.avi'
    #filename = '/media/ppzmis/data/Cracks/2020_02_14/2020_02_14Newno_salt_20uL_A_5fps_2020-02-14-115652-0000.avi'
    vidObj = ReadVideo(filename=filename)
    angle = -5+ 0.1503646#175
    crack_threshval = 110
    frame_index=1
    scale = 300/1803.91 # scale in um/pixel taken from

    #Different versions of code
    frame = vidObj.find_frame(frame_index)
    crack_frame, rot_frame, gray_frame = processframe(frame, angle, cutpos=100)
    #Find location of crack tip
    cracktip_x, cracktip_y, rot_frame = find_cracktip(crack_frame, rot_frame)
    rot_frame,  R =find_cracktip_radius(crack_frame,rot_frame, cracktip_x)
    rot_frame, compaction_x = find_compaction(rot_frame, frame_index)
    x_cropmin = cracktip_x-400
    x_cropmax = cracktip_x + 2000
    y_cropmin = cracktip_y -200
    y_cropmax = cracktip_y + 200

    show_frame(rot_frame[y_cropmin:y_cropmax, x_cropmin:x_cropmax, :])

    writevid = WriteVideo(filename=filename[:-4] + 'annotation.mp4',frame_size=np.shape(rot_frame[y_cropmin:y_cropmax,x_cropmin:x_cropmax, :]))

    output = [[0,cracktip_x*scale, compaction_x*scale, R*scale]]

    for index in range(700):
        frame=vidObj.find_frame(index)
        crack_frame, rot_frame, gray_frame = processframe(frame, angle, cutpos=100)
        cracktip_x, cracktip_y, rot_frame = find_cracktip(crack_frame,
                                                          rot_frame)
        rot_frame, rad = find_cracktip_radius(crack_frame, rot_frame, cracktip_x)
        rot_frame, compaction_x = find_compaction(rot_frame, index)
        #show_frame(rot_frame[y_cropmin:y_cropmax,x_cropmin:x_cropmax, :])
        writevid.add_frame(rot_frame[y_cropmin:y_cropmax,x_cropmin:x_cropmax, :])
        #frame(rot_frame)
        print(compaction_x)
        output.append([index*0.2,cracktip_x*scale, compaction_x*scale, rad*scale])
    writevid.close()

    np.savetxt(filename[:-4] + 'crackcurvaturecircle.txt',np.array(output))


    

