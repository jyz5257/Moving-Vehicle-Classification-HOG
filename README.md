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

### Feature Training with SVM
### Test the result
