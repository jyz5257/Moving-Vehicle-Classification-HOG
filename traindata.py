import numpy as np
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
import cv2
import glob
import os

def train():
    car_path = 'features/car'
    truck_path = 'features/truck'

    data = []
    label = []

    for feat_path in glob.glob(os.path.join(car_path,"*.feat")):
        fd = joblib.load(feat_path)
        data.append(fd)
        label.append(0)

    for feat_path in glob.glob(os.path.join(truck_path,"*.feat")):
        fd = joblib.load(feat_path)
        data.append(fd)
        label.append(1)

    clf = LinearSVC()
    clf.fit(data,label)
    joblib.dump(clf,'svm_linearmodel.pkl')

if __name__=='__main__':
    train()
