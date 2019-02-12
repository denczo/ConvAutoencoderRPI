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

img=mpimg.imread('images/stardestroyer32x32.png')
#img=np.append(img[:,:,:1].flatten(),np.append(img[:,:,1:2],img[:,:,2:3]))

#filterSize, filterAmount, stride, learnRate
#CAE = ConvLayer(9, 5, 3, 0.0001)
#CAE.setInput(img,3)
#CAE.setInput(dataBatch[0],1)
#print(img[:,:,:1].shape)
#CAEL = ConvLayerLight(img[:,:,:1].flatten(),9, 5, 1, 0.001)
CAEL = ConvLayerLight(dataBatch[0],9, 9, 1, 0.01)
fmSize = CAEL.fStepsOneAxis ** 2

imSizeX = int(math.sqrt(CAEL.inputSize))
VIS = Viz(5,9,9,fmSize,3,CAEL.fStepsOneAxis)

for i in range(10000):
    CAEL.updateInput(dataBatch[i])
    CAEL.slide()

    if i%50 == 0:
        CAEL.createConvMatrix()
        print(i)
        VIS.setInputs(CAEL.input,CAEL.filter,CAEL.featureMaps,CAEL.recon,1)
        VIS.visualizeNet()

