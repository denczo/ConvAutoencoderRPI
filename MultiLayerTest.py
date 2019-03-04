from ConvAutoencoder.ConvLayerLight import ConvLayerLight
from ConvAutoencoder.Visualizer import Viz
import numpy as np
import math
import matplotlib.image as mpimg

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

def vizLayer(cl,height,width,name):
    fSizeAxis = int(math.sqrt(cl.filterSize))
    fmSize = cl.fStepsOneAxis ** 2
    viz = Viz(height,width,cl.filterAmount,fmSize,fSizeAxis,cl.fStepsOneAxis,name)
    return viz

#img=mpimg.imread('images/XWing32x32.png')

img=mpimg.imread('images/trooper32.png')
img=np.append(img[:,:,:1].flatten(),np.append(img[:,:,1:2],img[:,:,2:3]))

channels = 1
filterAmount = 9
layer = []
#input, channels, filterSize, filterAmount, stride, learnRate
#CL1 = ConvLayerLight(img,channels, 9, 8, 3, 0.05)
CL1 = ConvLayerLight(dataBatch[0],channels, 9, 8, 2, 0.05)
CL2 = ConvLayerLight(CL1.featureMaps.flatten(),CL1.filterAmount, 9, 16, 2, 0.01)
CL3 = ConvLayerLight(CL2.featureMaps.flatten(),CL2.filterAmount, 9, 24, 1, 0.01)

def trainConvLayer(prevLayer,currLayer,epochs):

    Viz = vizLayer(currLayer, 6, 14, len(prevLayer))
    filterAmount = 0
    prevData = 0
    for i in range(epochs):
        if len(prevLayer) <= 0:
            filterAmount = currLayer.filterAmount
            prevOut = dataBatch[i]
        else:
            filterAmount = prevLayer[-1].filterAmount
            prevLayer[0].updateInput(dataBatch[i])
            prevLayer[0].slide(False)
            prevOut = prevLayer[-1].featureMaps.flatten()

        for j in range(len(prevLayer)-1):
            prevData = prevLayer[j].featureMaps.flatten()
            prevLayer[j+1].updateInput(prevData)
            prevLayer[j+1].slide(False)

        currLayer.updateInput(prevOut)

        if i < 100000:
            currLayer.slide(True)
        else:
            currLayer.slide(False)

        if i%25 == 0:
            print(i)
            Viz.setInputs(currLayer.input,currLayer.filter,currLayer.featureMaps.flatten(),currLayer.reconstrInput,currLayer.channels)
            Viz.visualizeNet(False)

    Viz.endViz()
    layer.append(currLayer)

CL1.setBiasVisible(0.2)
CL1.setBiasesFMs(np.full(CL1.filterAmount,0.1))
trainConvLayer(layer,CL1,150)
#CL2.setBiasVisible(0.2)
#CL2.setBiasesFMs(np.full(CL2.filterAmount,0.2))
#trainConvLayer(layer,CL2,200)
#CL3.setBiasVisible(0.3)
#CL3.setBiasesFMs(np.full(CL3.filterAmount,0.3))
#trainConvLayer(layer,CL3,150)

#CL2.guidedBackwardsActivation(CL3.reconstrInput.reshape(CL2.filterAmount,CL2.fStepsOneAxis**2),[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
CL1.guidedBackwardsActivation(CL1.featureMaps.reshape(CL1.filterAmount,CL1.fStepsOneAxis**2),[5,6])
Viz = vizLayer(CL1, 5, 9, 0)
Viz.setInputs(CL1.input,CL1.filter,CL1.featureMaps.flatten(),CL1.reconstrInput,CL1.channels)
Viz.visualizeFeatures(CL1.reconstrInput,True)
