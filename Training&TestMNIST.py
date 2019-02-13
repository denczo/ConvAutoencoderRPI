from ConvAutoencoder.ConvLayerLight import ConvLayerLight
from ConvAutoencoder.Visualizer import Viz

import numpy as np
import matplotlib.image as mpimg
import math

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

img=mpimg.imread('images/awing32x32.png')
img=np.append(img[:,:,:1].flatten(),np.append(img[:,:,1:2],img[:,:,2:3]))
#CAEL = ConvLayerLight(img[:,:,:1].flatten(),9, 5, 1, 0.001)

#input, channels, filterSize, filterAmount, stride, learnRate
channels = 3
filterAmount = 8
CAEL = ConvLayerLight(img,channels, 9, filterAmount, 1, 0.01)
#CAEL = ConvLayerLight(dataBatch[0],channels, 9, filterAmount, 1, 0.001)
fmSize = CAEL.fStepsOneAxis ** 2

imSizeX = int(math.sqrt(CAEL.inputSize))
VIS = Viz(5,9,filterAmount,fmSize,3,CAEL.fStepsOneAxis)
CAEL.setBiasVisible(1)
CAEL.setBiasesFMs([0.25,0.5,0.25,0.2,0.25,0.1,0.3,0.4])

for i in range(10000):
    #CAEL.updateInput(dataBatch[i])
    CAEL.slide()

    if i%500 == 0:
        CAEL.createConvMatrix()
        print(i)
        VIS.setInputs(CAEL.input,CAEL.filter,CAEL.featureMaps,CAEL.recon,channels)
        VIS.visualizeNet()

