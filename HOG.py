from skimage.feature import hog
import cv2
from sklearn.externals import joblib
import glob
import os

def extract_features():
    
    orientations = 9
    pixels_per_cell = [4, 4]
    cells_per_block = [2, 2]
    visualize = False
    normalize = True

    for im_path in glob.glob(os.path.join('dataset/car', "*")):
        
        img = cv2.imread(im_path,0)
        fd = hog(img, orientations, pixels_per_cell, cells_per_block, visualize, normalize)
        
        fd_name = os.path.split(im_path)[1].split(".")[0] + ".feat"
        fd_path = os.path.join('features/car', fd_name)
        joblib.dump(fd, fd_path)

    for im_path in glob.glob(os.path.join('dataset/truck', "*")):
        
        img = cv2.imread(im_path,0)
        fd = hog(img, orientations, pixels_per_cell, cells_per_block, visualize, normalize)
        
        fd_name = os.path.split(im_path)[1].split(".")[0] + ".feat"
        fd_path = os.path.join('features/truck', fd_name)
        joblib.dump(fd, fd_path)

if __name__=='__main__':
    extract_features()

#img = cv2.imread(im_path,0)

#cv2.namedWindow('result', cv2.WINDOW_NORMAL)
#cv2.imshow('result', img)

#cv2.waitKey(0)
#cv2.destroyAllWindows()
