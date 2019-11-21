# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 23:27:55 2019

@author: Sridhar Korada
"""
#**********************MNIST database of handwritten digits
# Dataset of 60,000 28x28 grayscale images of the 10 digits, along with a test set of 10,000 images.

#loading data
import tensorflow as tf
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

#Looking into train and test data
x_train.shape
x_test.shape

import matplotlib.pyplot as plt
plt.imshow(x_train[0]) 
y_train[0]
plt.imshow(x_train[1]) 
y_train[1]
plt.imshow(x_test[0]) 
y_test[0]

for i in range(10):
    print(y_test[i])

print(x_train[0])

#normalizing the pixels in the images to be values between 0 to 1
x_train, x_test = x_train / 255.0, x_test / 255.0

#Build the model
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
    
# compile the model    
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10)

#Model to evaluate with test data (loss, accuracy)
model.evaluate(x_test, y_test) #Accuracy = 98.12%

#Model to predict the results
pred_y=model.predict(x_test)
pred_y.shape
pred_y[1]   #before argmax

y_pred=[]
import numpy as np
for i in range(len(pred_y)):
    y_pred.append(np.argmax(pred_y[i]))
    
y_pred[1]  #after argmax

y_pred[1], y_test[1]

for i in range(20):
    print(y_pred[i], y_test[i])