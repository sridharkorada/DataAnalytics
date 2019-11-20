# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 10:07:03 2019

@author: Sridhar Korada
"""

#*************** this program classifies images******************

#loading data
from keras.models import Sequential
from keras.layers import Dense, Activation

from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data() #it executes based on net speed


print(type(x_train))   #numpy.ndarray

#shape of each dataset
x_train.shape       #50000, 32, 32, 3 - 50000 images of 32x32 pixel image of depth 3
y_train.shape      #50000, 1
x_test.shape   #10000, 32, 32, 3
y_test.shape   #10000, 1

#take a look to at the first image (at index=0) in the training dataset in (matrix form)
x_train[0]

#showing image as picture
import matplotlib.pyplot as plt
img = plt.imshow(x_train[0])  #shows frog image

#printing the label of the image
print('the label is:', y_train[0])  #prints 6 corresponding to frog
# 0-aeroplane, 1-automobile, 2-bird, 3-cat, 4-deer, 5-dog , 6-frog, 7-horse, 8-ship, 9-truck

#one-hot encoding: convert the lables into a set of 10 numbers to input into the neural network (converting to categorical)
from keras.utils import to_categorical
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

#print the new labels in the training dataset
print(y_train_one_hot)

#print an example of the new labels
print('the one hot label is:', y_train_one_hot[0])  #6 converted to [0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]

#normalizing the pixels in the images to be values between 0 to 1
x_train = x_train/255
x_test = x_test/255

#Build the CNN
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

#Create the architecture
model = Sequential()

#Convolution Layer
model.add(Conv2D(32, (5,5), activation='relu', input_shape=(32,32,3)))
#Maxpooling layer
model.add(MaxPooling2D(pool_size=(2,2)))  #image will become 16x16

#Convolution Layer
model.add(Conv2D(32, (5,5), activation='relu'))
#Maxpooling layer
model.add(MaxPooling2D(pool_size=(2,2))) #image will become 8x8

#Flatten layer
model.add(Flatten())

model.add(Dense(1000, activation='relu'))  #relu - REctified Linear Unit
model.add(Dense(10, activation='softmax'))  #10 - because out is 10 diff types

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#Train the model
hist = model.fit(x_train, y_train_one_hot, batch_size=256, epochs=10, validation_split=0.3)

#Get the model accuracy
model.evaluate(x_test, y_test_one_hot)[1]  #0.6783 that means 67.83% accuracy

#Visualize the models accuracy
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()


#Visualize the models loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

#Load the data
from google.colab import files  #not working
uploaded = files.upload()       #not working
my_image=plt.imread('cat.4041.jpg') #not working hence loading x_train[1] below
my_image=x_train[1]

#show the uploaded image
img = plt.imshow(my_image)

#Resize the image
from skimage.transform import resize #resize not needed, not working
my_image_resized = resize(my_image, (32,32,3))
img = plt.imshow(my_image_resized)

# get the probabilities for each class
import numpy as np
probabilities = model.predict(np.array([my_image,]))

#print the probabilities
probabilities

number_to_class = ['aeroplane', 'automobile', 'bird', 'cat', 'deer', 'dog' ,'frog', 'horse', 'ship', 'truck']
index = np.argsort(probabilities[0,:])
print('Most likely class:', number_to_class[index[9]], '--probability:', probabilities[0, index[9]])
print('Second likely class:', number_to_class[index[8]], '--probability:', probabilities[0, index[8]])
print('Third likely class:', number_to_class[index[7]], '--probability:', probabilities[0, index[7]])
print('Fourth likely class:', number_to_class[index[6]], '--probability:', probabilities[0, index[6]])
print('Fifth likely class:', number_to_class[index[5]], '--probability:', probabilities[0, index[5]])


#save the model
model.save('my_model.h5')

#load the model
from keras.models import load_model
model=load_model('my_model.h5')

#**************************************validating all images in test data************************ 
# 0-aeroplane, 1-automobile, 2-bird, 3-cat, 4-deer, 5-dog , 6-frog, 7-horse, 8-ship, 9-truck

print(y_test[0]) #3
plt.imshow(x_test[0])   #3-cat
probabilities = model.predict(np.array([x_test[0],]))
index = np.argsort(probabilities[0,:]) 
print('Most likely class:', number_to_class[index[9]], '--probability:', probabilities[0, index[9]])

print(y_test[1])
plt.imshow(x_test[1])  #8-ship
probabilities = model.predict(np.array([x_test[1],]))
index = np.argsort(probabilities[0,:]) 
print('Most likely class:', number_to_class[index[9]], '--probability:', probabilities[0, index[9]])

print(y_test[3])  #0
plt.imshow(x_test[3])  #0-aeroplane
probabilities = model.predict(np.array([x_test[3],]))
index = np.argsort(probabilities[0,:]) 
print('Most likely class:', number_to_class[index[9]], '--probability:', probabilities[0, index[9]])

print(y_test[4])  #6
plt.imshow(x_test[4])  #frog
probabilities = model.predict(np.array([x_test[4],]))
index = np.argsort(probabilities[0,:]) 
print('Most likely class:', number_to_class[index[9]], '--probability:', probabilities[0, index[9]])

print(y_test[6])  #1
plt.imshow(x_test[6])  #automobile
probabilities = model.predict(np.array([x_test[6],]))
index = np.argsort(probabilities[0,:]) 
print('Most likely class:', number_to_class[index[9]], '--probability:', probabilities[0, index[9]])

print(y_test[11])  #9
plt.imshow(x_test[11])  #truck
probabilities = model.predict(np.array([x_test[11],]))
index = np.argsort(probabilities[0,:]) 
print('Most likely class:', number_to_class[index[9]], '--probability:', probabilities[0, index[9]])

print(y_test[12])  #5
plt.imshow(x_test[12])  #dog
probabilities = model.predict(np.array([x_test[12],]))
index = np.argsort(probabilities[0,:]) 
print('Most likely class:', number_to_class[index[9]], '--probability:', probabilities[0, index[9]])

print(y_test[13])  #7
plt.imshow(x_test[13])  #horse
probabilities = model.predict(np.array([x_test[13],]))
index = np.argsort(probabilities[0,:]) 
print('Most likely class:', number_to_class[index[9]], '--probability:', probabilities[0, index[9]])

print(y_test[22])  #4
plt.imshow(x_test[22])  #deer
probabilities = model.predict(np.array([x_test[22],]))
index = np.argsort(probabilities[0,:]) 
print('Most likely class:', number_to_class[index[9]], '--probability:', probabilities[0, index[9]])

print(y_test[25])  #2
plt.imshow(x_test[25])  #bird
probabilities = model.predict(np.array([x_test[25],]))
index = np.argsort(probabilities[0,:]) 
print('Most likely class:', number_to_class[index[9]], '--probability:', probabilities[0, index[9]])

#*****************************************test check in loop************************
# 0-aeroplane, 1-automobile, 2-bird, 3-cat, 4-deer, 5-dog , 6-frog, 7-horse, 8-ship, 9-truck
number_of_iterations=40

for i in range(number_of_iterations):
    probabilities = model.predict(np.array([x_test[i],]))
    index = np.argsort(probabilities[0,:])
    print('Most likely class:', number_to_class[index[9]], '--probability:', probabilities[0, index[9]])
    print(y_test[i])
    print('***************[',i,']*************')


