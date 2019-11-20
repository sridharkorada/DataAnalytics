# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 07:21:11 2019

@author: Sridhar Korada
"""
#**************Fashion-MNIST database of fashion articles****************************
from keras.datasets import fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# 0-T-shirt/top, 1-Trouser, 2-Pullover, 3-Dress, 4-Coat, 5-Sandal, 6-Shirt, 7-Sneaker, 8-Bag, 9-Ankle boot