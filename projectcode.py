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

train_dataset = train.flow_from_directory("", 
                                          target_size=(200,200),
                                          batch_size = 3,
                                          class_mode="binary")
vaildation_dataset = train.flow_from_directory("", 
                                          target_size=(200,200),
                                          batch_size = 3,
                                          class_mode="binary")

model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(16,(3,3), activation = 'relu',input_shape=(200,200,3)),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    #
                                    tf.keras.layers.Conv2D(32(3,3), activation = 'relu'),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    #
                                    tf.keras.layers.Conv2D(64(3,3), activation = 'relu'),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    ##
                                    tf.keras.layers.Flatten(),
                                    ##
                                    tf.keras.layers.Dense(512,activation= 'relu'),
                                    ##
                                    tf.keras.layers.Dense(1,activation='sigmoid')
                                    ])

model.compile(loss = "binary_crossentrophy",
              optimizer = RMSprop(lr=0.001),
              metrics= ['accuracy'])

model_fit = model.fit(train_dataset,
                      step_per_epoch = 3,
                      epochs = 30,
                      validation_data = vaildation_dataset)

dir_path = ""

for i in os.listdir(dir_path):
    img =image.load_img(dir_path+"//"+i,target_size=(200,200))
    plt.imshow(img)
    plt.show()

    X = image.img_to_array(img)
    X = np.expand_dims(X, axis=0)
    images = np.vstack([X])
    val = model.predict(images)
    if val == 0:
        print('not AI')
    else:
        print("AI")
