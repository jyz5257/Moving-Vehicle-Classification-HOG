import cv2
import numpy as np
from skimage import color
from skimage.feature import hog
from sklearn.externals import joblib
import imutils
import urllib

#BG Subtraction, convert to grayscale, and threshold image
def bgsubtract(bg,car):
    imgAbsdiff = cv2.absdiff(bg, car)
    imgGray = cv2.cvtColor(imgAbsdiff, cv2.COLOR_BGR2GRAY)
    ret1, thres = cv2.threshold(imgGray, 20, 255, cv2.THRESH_BINARY)
    return thres

# count the neighborhood foreground and background pixel
def check_pix(img,p):
    im = img[p[0]-1:p[0]+2, p[1]-1:p[1]+2]
    QF = 0
    QB = 0
    for i in range(0,3):
        for j in range(0,3):
            if im[i,j] == 255:
                QF = QF +1
            if im[i,j] == 0:
                QB = QB +1
    if im[1,1] == 255:
        QF = QF - 1
    if im[1,1] == 0:
        QB = QB - 1
    return QF,QB

# foraground adaptive bg subtraction
def fgbs(img):
    s = img.shape
    for i in range(1,s[0]-1):
        for j in range(1,s[1]-1):
            p = (i,j)
            QF,QB = check_pix(img,p)
            gamma = 1
            theta = 1.2
            v = theta * np.exp((QF-QB)/gamma)
            if v > 1:
                img[i,j] = 255
            if v < 1:
                img[i,j] = 0
    return img

#make bounding boxes
def box(img):
    image, cnts, _ = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    x = [0]*(len(cnts)-1)
    y = [0]*(len(cnts)-1)
    w = [0]*(len(cnts)-1)
    h = [0]*(len(cnts)-1)
    for i in range(0,(len(cnts)-1)):
        x[i],y[i],w[i],h[i] = cv2.boundingRect(cnts[i+1])
    return x,y,w,h

#import video stream
video = 'video/output2.avi'
c = cv2.VideoCapture(video)
_,f = c.read()

#HOG Parameters
orientations = 9
pixels_per_cell = [4, 4]
cells_per_block = [2, 2]
visualize = False
normalize = True

# load svm model
clf = joblib.load('svm_linearmodel.pkl')

# open a video writer using opencv
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
out = cv2.VideoWriter('output.avi', fourcc, 30.0, (320,240))

while True:
    _,f = c.read()
    crop_f = f[44:224, 0:320]

    # read the background image and remove the camera frame
    imgbg = cv2.imread('background.png')
    crop_bg = imgbg[44:224, 0:320]
    thres = bgsubtract(crop_bg,crop_f)
    
    (x,y,w,h) = box(thres)
    
    for i in range(0,len(x)):
        if w[i]>20 and h[i]>15:
            window = f[(y[i]+44):(y[i]+44+h[i]), x[i]:(x[i]+w[i])]
            window = color.rgb2gray(window)
            img1 = cv2.resize(window,(64,48))
            # examine the hog fature
            fd = hog(img1, orientations, pixels_per_cell, cells_per_block, visualize, normalize)
            fd = fd.reshape((1,-1))
            # predict the feature label
            pred = clf.predict(fd)
            if pred == 0:
                if w[i]<180:
                    cv2.rectangle(f,(x[i],y[i]+44),(x[i]+w[i],y[i]+44+h[i]),(255,255,0),1)
                if w[i] >180:
                    cv2.rectangle(f,(x[i],y[i]+44),(x[i]+w[i],y[i]+44+h[i]),(0,0,255),1)
            if pred == 1:
                cv2.rectangle(f,(x[i],y[i]+44),(x[i]+w[i],y[i]+44+h[i]),(0,0,255),1)

    # write the video       
    out.write(f)
    cv2.imshow('img',f)
    
    k = cv2.waitKey(1)
    if k == 27:
        break
    
cv2.destroyAllWindows()
c.release()
