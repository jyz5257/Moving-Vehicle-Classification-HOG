# Moving-Vehicle-Classification-HOG
## Project Objectives
The goal of the project is detecting the moving vehicle on the highway from a web camera and classify the vehicle into passenger cars and large trucks. In this project, I'm using Histogram of Oriented Gradients (HOG) descriptor to decribe the data features and SVM classification to classify them.
## Building Environement
* Python 2.x.x
* Opencv 
* Scikit-learn python package
## Project Description
### Collecting the data
In this project, I collected my dataset from a highway web camera. I didn't build a very large dataset, as there are only two types of objects to distinguish. I uploaded a zip file for the images I collect for training in this project. Basically, every image includes at least a large car.
### Backgrond Subtraction
The easiest way to do background subtraction is the pixel subtraction between an object image and a background image. However, using this way may not able to get a concrete object contour because there might be some pixels on the car very similiar to the background. Therefore, I used a method called adaptive foreground background subtraction to solve that, which is developed in this paper: "Foreground-Adaptive Background Subtraction". In this method, I build a foreground Markov model based on the small spatial neighborhood to improve discrimination sensitivity. Using this method, I'm able to get a better object shape after background subtraction and the noise pixels are also removed a lot. The code for this part is "bgsub_FA.py".
### Featrue Extraction with HOG 
To get the features using HOG, I need to collect the vehicle information first. As we already got the images with bounding boxes, we can crop the vehicle image in the bounding boxes. Then we need to scale all the images to the same size, which I used 64*48. Then, we can use HOG descriptor to extract features for each vehicle image. The HOG parameters we use are [2, 2] cells per black and [4, 4] pixels per cell. Finally, we can get a bunch of histogram data and save these features as the feature training dataset. These data are separated into small car feature data as negative training dataset and truck feature data as positive data training dataset.
### Feature Training with SVM
### Test the result
