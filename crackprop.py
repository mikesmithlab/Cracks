from Generic.video import ReadVideo, WriteVideo
import cv2
import numpy as np
from Generic.filedialogs import get_files_directory
import matplotlib.pyplot as plt



def resize_frame(frame,percent=25):
    percent = 25
    width = np.shape(frame)[0]
    height = np.shape(frame)[1]

    dim = (int(height * percent / 100), int(width * percent / 100))

    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

def show_frame(frame):
    cv2.imshow('detected circles', frame)
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

def set_edge_white(frame,column=200):
    frame[:,column]=255
    frame[:, column+1] = 255
    frame[:,column-1]=1
    frame[:, column - 2] = 0
    frame[:, column - 3] = 0
    frame[:, column - 4] = 0
    frame[:, column - 5] = 0
    frame[:, column - 6] = 0
    frame[:, column - 7] = 0
    frame[:, column - 8] = 0
    frame[:, column - 9] = 0
    mask = np.zeros(np.shape(frame))
    mask[:,column]=255
    mask[:,column]=255
    mask[:, column+1] = 255
    mask[:,column+2] = 255
    mask[:, column -1] = 255
    return frame, mask



def mask_right(frame,column=-100):
    rh_edge = np.shape(frame)[1]
    frame[:,rh_edge + column:-1] = 0
    return frame

def mask_top(frame,row=0):
    frame[0:row,:]=0
    return frame

def find_edges(new_frame,threshold=50, minArea=50000):
    canny_output = cv2.Canny(new_frame, threshold, 250)
    #show_frame(resize_frame(canny_output,percent=25))
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    #remove small contours
    _, contours, _ = cv2.findContours(canny_output, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    hull_list = []
    for i in range(len(contours)):
        # check for connections

        hull = cv2.convexHull(contours[i])
        if cv2.contourArea(hull) > minArea:
            hull_list.append(i)

    contours2 = contours[hull_list[0]]

    for i in range(len(contours2)):
        color=(255,0,0)
        cv2.drawContours(drawing, contours2, i, color,4)

    return drawing

def width_vals(frame):
    return np.sum(frame,axis=0)

if __name__ == '__main__':
    filter = '/media/ppzmis/SAMSUNG/crackbranching/CrackHoppingVids/Sample3_RHS_B4_2fps*.avi'
    filename = get_files_directory(filter)[0]
    vidObj = ReadVideo(filename=filename)


    vidObj.get_vid_props()



    frame = vidObj.find_frame(1000 - 1)

    threshval=100
    new_frame = threshold(frame[:,:,1],thresh=threshval)

    col_val=800
    new_frame,mask = set_edge_white(new_frame,column=col_val)

    angle= -40
    new_frame = rotate(new_frame,angle)

    new_mask = rotate (mask,angle)
    #show_frame(resize_frame(new_frame))
    new_frame = imfill(new_frame)

    new_frame = cv2.subtract(new_frame.astype(np.uint8),new_mask.astype(np.uint8))

    #mask_edge = 2000
    #new_frame = mask_right(new_frame,column=mask_edge)
    #new_frame = mask_top(new_frame,row=mask_edge)

    new_frame=extract_nth_biggest_object(new_frame,n=0)
    width = np.zeros((int(vidObj.num_frames), int(np.shape(new_frame)[1])))

    show_frame(resize_frame(new_frame))

    vidObj.find_frame(0)
    sz=np.shape(resize_frame(new_frame))
    writevid = WriteVideo(filename=filename[:-4]+'_bw.mp4',frame_size=sz,write_frame=False)

    for index in range(vidObj.num_frames-1):
        frame=vidObj.read_next_frame()
        new_frame = threshold(frame[:, :, 1], thresh=threshval)
        new_frame, _ = set_edge_white(new_frame, column=col_val)
        new_frame = rotate(new_frame, angle)
        new_mask = rotate(mask, angle)
        new_frame = imfill(new_frame)
        new_frame = cv2.subtract(new_frame.astype(np.uint8), new_mask.astype(np.uint8))
        #new_frame = mask_right(new_frame, column=mask_edge)
        #new_frame = mask_top(new_frame, row=mask_edge)
        new_frame = extract_nth_biggest_object(new_frame,n=0)

        width[index, :] = width_vals(new_frame)/255
        new_frame = resize_frame(new_frame, percent=25)
        writevid.add_frame(new_frame)
        #show_frame(new_frame)

        print(vidObj.num_frames - index)

    writevid.close()
    np.savetxt(filename[:-4] + '.txt',width)



