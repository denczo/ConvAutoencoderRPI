from ConvAutoencoder.ConvLayerLight import ConvLayerLight
from ConvAutoencoder.Visualizer import Viz

import numpy as np
import matplotlib.image as mpimg
import math
import matplotlib.pyplot as plt

training_data_file = open("C:/Users/Dennis/Documents/studium/mnist_train.csv",'r')
data = training_data_file.readlines()
training_data_file.close()

dataLength = len(data)
amountData = 10001
dataBatch = np.zeros((amountData,784))

for i in range(amountData):
    oneDigit = data[i % dataLength].split(',')
    #temp = (np.asfarray(oneDigit[1:]) / 255.0 * 0.99) + 0.01
    temp = (np.asfarray(oneDigit[1:]) / 255.0 * 0.99)
    dataBatch[i] = temp

äimg=mpimg.imread('images/awing32x32.png')
img=mpimg.imread('images/stardestroyer128x128.png')

img=np.append(img[:,:,:1].flatten(),np.append(img[:,:,1:2],img[:,:,2:3]))
#CAEL = ConvLayerLight(img[:,:,:1].flatten(),9, 5, 1, 0.001)

#input, channels, filterSize, filterAmount, stride, learnRate
channels = 3
filterAmount = 9
#biasesFM = [0.25,0.5,0.25,0.2,0.25,0.1,0.1,0.1,0.1]
biasesFM = [0.0]*9
CAEL = ConvLayerLight(img,channels, 9, filterAmount, 2, 0.01)
#CAEL = ConvLayerLight(dataBatch[0],channels, 9, filterAmount, 1, 0.0001)
CAEL.setBiasVisible(0)
CAEL.setBiasesFMs(biasesFM)
fmSize = CAEL.fStepsOneAxis ** 2
imSizeX = int(math.sqrt(CAEL.inputSize))

VIS = Viz(5,9,filterAmount,fmSize,3,CAEL.fStepsOneAxis)
for i in range(1000):
    #CAEL.updateInput(dataBatch[i])
    CAEL.slide()

    if i%25 == 0:
        print(i)
        VIS.setInputs(CAEL.input,CAEL.filter,CAEL.featureMaps.flatten(),None,channels)
        VIS.visualizeNet()

CAEL2 = ConvLayerLight(CAEL.featureMaps.flatten(),CAEL.filterAmount, 9, 9, 1, 0.01)
CAEL2.setBiasVisible(0.2)
CAEL2.setBiasesFMs([0.25,0.5,0.25,0.2,0.25,0.1,0.3,0.4,0.1])
fmSize = CAEL2.fStepsOneAxis ** 2
imSizeX = int(math.sqrt(CAEL2.inputSize))
VIS2 = Viz(10,18,9,fmSize,3,CAEL2.fStepsOneAxis)

for i in range(100):
    #CAEL.updateInput(dataBatch[i])
    CAEL2.slide()

    if i%25 == 0:
        print(i)
        VIS2.setInputs(CAEL2.input,CAEL2.filter,CAEL2.featureMaps.flatten(),None,CAEL.filterAmount)
        VIS2.visualizeNet()
