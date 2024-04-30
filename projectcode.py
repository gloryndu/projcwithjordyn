from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.processing import image
import tensorflow  as tf 
import matplotlib.pyplot as plt 
import cv2 
import os
import numpy as np 

img = image.load_img("")
plt.imshow(img)
cv2.imread("").shape

train = ImageDataGenerator(rescale= 1/255)
vaildation = ImageDataGenerator(rescale= 1/255)