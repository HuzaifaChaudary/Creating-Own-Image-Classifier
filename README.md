# Creating-Own-Image-Classifier
Creating-Own-Image-Classifier using Machine Learning
This project is part of the Udacity AI Programming with Python Nanodegree program. The goal of this project is to create an image classifier that can classify images into different categories. The project is divided into two parts:

Developing an image classifier using a pre-trained model.
Building a command line application that can train a new classifier on a dataset of images and use the trained classifier to predict new images.
Pre-requisites
Before running this project, you need to have the following installed on your machine:

Python 3.x
Jupyter Notebook
PyTorch
Numpy
Pandas
Matplotlib
Dataset
For this project, we will be using the Oxford Flowers 102 dataset which contains 102 flower categories. The dataset is already split into a training set, a validation set, and a testing set. The training set contains 1020 images, the validation set contains 614 images, and the testing set contains 1020 images.

Steps to run the project
Clone this repository on your local machine.
Navigate to the project directory using the command line.
Open the Jupyter Notebook by running jupyter notebook in the command line.
Run the first two cells of the Jupyter Notebook to import the necessary libraries and load the data.
Run the remaining cells to train the model and save it to a checkpoint file.
To test the model, run the command line application using the following command:
python predict.py path/to/image checkpoint

path/to/image is the path to the image you want to classify.
checkpoint is the path to the saved checkpoint file.
Conclusion
In this project, we have built an image classifier using a pre-trained model and trained it on a dataset of images. We have also built a command line application to predict the category of new images. This project serves as an introduction to building image classifiers using machine learning techniques.



