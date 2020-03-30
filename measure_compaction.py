from Generic.video import ReadVideo, WriteVideo
import cv2
import numpy as np



def resize_frame(frame,percent=25):
    percent = 25
    width = np.shape(frame)[0]
    height = np.shape(frame)[1]

    dim = (int(height * percent / 100), int(width * percent / 100))

    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

def show_frame(frame):
    global compact_x
    cv2.namedWindow('img')
    cv2.setMouseCallback('img', click)
    cv2.imshow('img', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return compact_x

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




def find_compaction(frame):
    return show_frame(resize_frame(frame))



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

global compact_x
def click(event, x,y,dummy,dummy2):
    global compact_x

    if event == cv2.EVENT_LBUTTONDOWN:
        compact_x = x
        print(x)
        print(y)




def processframe(frame, angle):
    rot_frame = rotate(frame, angle)
    gray_frame = cv2.cvtColor(rot_frame, cv2.COLOR_BGR2GRAY)

    # BW image with the crack extracted from background
    crack_frame = frame[:,:,1]#threshold(frame[:, :, 1], thresh=crack_threshval)
    crack_frame = rotate(crack_frame, angle)
    crack_frame = imfill(crack_frame)
    #crack_frame = extract_nth_biggest_object(crack_frame, n=0)
    return crack_frame, rot_frame, gray_frame

if __name__ == '__main__':

    #filename = '/media/ppzmis/data/Cracks/2020_03_17/25mMnacl5fps_2020-03-17-110622-0000.avi'
    filename = '/media/ppzmis/data/Cracks/2020_02_14/2020_02_14Newno_salt_20uL_A_5fps_2020-02-14-115652-0000.avi'
    vidObj = ReadVideo(filename=filename)
    angle = angle = 175#-5+ 0.1503646
    fps=5


    output=[]
    for index in range(0,vidObj.num_frames-1,50):
        vidObj.set_frame(index)
        frame = vidObj.read_next_frame()
        crack_frame, rot_frame, gray_frame = processframe(frame, angle)
        compact_pos_pixels = find_compaction(rot_frame)

        time = (index+1)/fps
        output.append([time, compact_pos_pixels])
    print('final')
    print(output)
    np.savetxt(filename[:-4] + 'pos_compactionb.txt',np.array(output))

    

