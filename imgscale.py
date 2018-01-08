import numpy as np
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
import cv2
import glob
import os

for im_path in glob.glob(os.path.join('cropimage', "*")):
    img = cv2.imread(im_path)
    img1 = cv2.resize(img,(64,48))
    if 'car' in im_path:
        filename = os.path.split(im_path)[1].split(".")[0] + '.jpg'
        filepath = os.path.join('dataset/car', filename)
        cv2.imwrite(filepath, img1)
    if 'truck' in im_path:
        filename = os.path.split(im_path)[1].split(".")[0] + '.jpg'
        filepath = os.path.join('dataset/truck', filename)
        cv2.imwrite(filepath, img1)

