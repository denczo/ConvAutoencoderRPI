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

img=mpimg.imread('images/stardestroyer64x64.png')
img=np.append(img[:,:,:1].flatten(),np.append(img[:,:,1:2],img[:,:,2:3]))

channels = 1
filterAmount = 7
layer = []
#input, channels, filterSize, filterAmount, stride, learnRate
#CL1 = ConvLayerLight(img,channels, 9, filterAmount, 1, 0.01)
CL1 = ConvLayerLight(dataBatch[0],channels, 9, filterAmount, 1, 0.01)
CL2 = ConvLayerLight(CL1.featureMaps.flatten(),CL1.filterAmount, 9, 16, 1, 0.01)
CL3 = ConvLayerLight(CL2.featureMaps.flatten(),CL2.filterAmount, 9, 16, 1, 0.01)

def trainConvLayer(prevLayer,currLayer,observe):

    Viz = vizLayer(currLayer, 6, 14, len(prevLayer))
    filterAmount = 0
    prevData = 0
    for i in range(105):
        if len(prevLayer) <= 0:
            filterAmount = currLayer.filterAmount
            prevOut = dataBatch[i]
        else:
            filterAmount = prevLayer[-1].filterAmount
            prevLayer[0].updateInput(dataBatch[i])
            prevLayer[0].slide(False,False)
            prevOut = prevLayer[-1].featureMaps.flatten()

        for j in range(len(prevLayer)-1):
            prevData = prevLayer[j].featureMaps.flatten()
            prevLayer[j+1].updateInput(prevData)
            prevLayer[j+1].slide(False,False)

        currLayer.updateInput(prevOut)

        if i < 500:
            currLayer.slide(True,observe)
        else:
            currLayer.slide(False,observe)

        if i%25 == 0:
            print(i)
            Viz.setInputs(currLayer.input,currLayer.filter,currLayer.featureMaps.flatten(),currLayer.reconstrInput,currLayer.channels)
            Viz.visualizeNet(False)

    Viz.endViz()
    layer.append(currLayer)

trainConvLayer(layer,CL1,False)
trainConvLayer(layer,CL2,False)
trainConvLayer(layer,CL3,False)

#CL2.foo(CL3.reconstrInput)
CL1.foo(CL2.reconstrInput.reshape(CL1.filterAmount,CL1.fStepsOneAxis**2))
Viz = vizLayer(CL1, 5, 9, 0)
Viz.setInputs(CL1.input,CL1.filter,CL1.featureMaps.flatten(),CL1.reconstrInput,CL1.channels)
Viz.visualizeFeatures(CL1.reconstrInput,True)
