import numpy as np
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
import cv2
import glob
import os

def readimg(img):
    image = cv2.imread(img)
    cropim = image[44:224,0:320]
    return cropim

def bgsubtract(bg,car):
    imgAbsdiff = cv2.absdiff(imgbg, imgcar)
    imgGray = cv2.cvtColor(imgAbsdiff, cv2.COLOR_BGR2GRAY)
    ret1, thres = cv2.threshold(imgGray, 20, 255, cv2.THRESH_BINARY)
    res = np.zeros(thres.shape)
    for i in range(0,thres.shape[0]):
        for j in range(0,thres.shape[1]):
            if thres[i][j] == 0:
                res[i][j] = 255
            if thres[i][j] == 1:
                res[i][j] = 0
    return res

def check_pix(img,p):
    im = img[p[0]-1:p[0]+2, p[1]-1:p[1]+2]
    QF = 0
    QB = 0
    for i in range(0,3):
        for j in range(0,3):
            if im[i,j] == 0:
                QF = QF +1
            if im[i,j] == 255:
                QB = QB +1
    if im[1,1] == 0:
        QF = QF - 1
    if im[1,1] == 255:
        QB = QB - 1
    #if black!=0:
     #   print black, white
    return QF,QB

def fg_adapt(img):
    s = img.shape
    for i in range(1,s[0]-1):
        for j in range(1,s[1]-1):
            p = (i,j)
            QF,QB = check_pix(img,p)
            if QF > 5:
                img[i,j] = 0
            if QB > 5:
                img[i,j] = 255
    return img

def clear(img):
    s = img.shape
    for i in range(0,s[0]):
        img[i,0] = 255
        img[i,s[1]-1] = 255
    for j in range(0,s[1]):
        img[0,j] = 255
        img[s[0]-1,j] = 255
    return img

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
                img[i,j] = 0
            if v < 1:
                img[i,j] = 255
    return img

imgbg = readimg('./bg/back.jpg')

for im_path in glob.glob(os.path.join('./images', "*")):
    if 'car' in im_path or 'truck' in im_path:
        imgcar = readimg(im_path)
        bgresult = bgsubtract(imgbg,imgcar)
        answer = fgbs(bgresult)
        answer = clear(answer)
        filename = 'bgsub1' + os.path.split(im_path)[1].split(".")[0] + '.jpg'
        filepath = os.path.join('./bgsub1', filename)
        cv2.imwrite(filepath, answer)
        print filename
#imgcar = readimg('./datapictures/truck4.jpg')

#bgresult = bgsubtract(imgbg,imgcar)
#answer = fgbs(bgresult)
#answer = clear(answer)

#image, cnts, _ = cv2.findContours(answer.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

#print cnts
#print bgresult[177:179,200:202]

#cv2.namedWindow('result', cv2.WINDOW_NORMAL)
#cv2.imshow('result', answer)
#cv2.imshow('result', bgresult)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
