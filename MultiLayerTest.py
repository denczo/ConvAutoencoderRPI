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
    viz = Viz(height,width,filterAmount,fmSize,fSizeAxis,cl.fStepsOneAxis,name)
    return viz

#img=mpimg.imread('images/awing32x32.png')
#img=np.append(img[:,:,:1].flatten(),np.append(img[:,:,1:2],img[:,:,2:3]))

channels = 1
filterAmount = 5
layer = []

#input, channels, filterSize, filterAmount, stride, learnRate
#CL1 = ConvLayerLight(img,channels, 9, filterAmount, 2, 0.01)
CL1 = ConvLayerLight(dataBatch[0],channels, 9, filterAmount, 1, 0.01)
#Viz1 = vizLayer(CL1,5,9,"1ST")
Viz1 = 0


for i in range(0):
    CL1.updateInput(dataBatch[i])
    if i < 100:
        CL1.slide(True,True)
    else:
        CL1.slide(False,True)

    if i%25 == 0:

        print(i)
        Viz1.setInputs(CL1.input,CL1.filter,CL1.featureMaps.flatten(),CL1.reconstrInput,channels)
        Viz1.visualizeNet()

#Viz1.endViz()

CL2 = ConvLayerLight(CL1.featureMaps.flatten(),CL1.filterAmount, 9, 9, 2, 0.001)
#Viz2 = vizLayer(CL2,5,9,"2ND")
Viz2 = 0

for i in range(0):
    CL1.updateInput(dataBatch[i])
    CL1.slide(False,False)

    CL2.updateInput(CL1.featureMaps.flatten())
    if i < 100:
        CL2.slide(True,True)
    else:
        CL2.slide(False,True)

    if i%25 == 0:
        print(i)
        Viz2.setInputs(CL2.input,CL2.filter,CL2.featureMaps.flatten(),CL2.reconstrInput,CL1.filterAmount)
        Viz2.visualizeNet()



CL3 = ConvLayerLight(CL2.featureMaps.flatten(),CL2.filterAmount, 9, 9, 2, 0.001)
#Viz3 = vizLayer(CL3,5,9,"3RD")
Viz3 = 0
for i in range(0):
    CL1.updateInput(dataBatch[i])
    CL1.slide(False)
    CL2.updateInput(CL1.featureMaps.flatten())
    CL2.slide(False)
    CL3.updateInput(CL2.featureMaps.flatten())

    if i < 500:
        CL3.slide(True)
    else:
        CL3.slide(False)

    if i%25 == 0:
        print(i)
        Viz3.setInputs(CL3.input,CL3.filter,CL3.featureMaps.flatten(),None,CL2.filterAmount)
        Viz3.visualizeNet()

def trainConvLayer(prevLayer,currLayer):

    Viz = vizLayer(currLayer, 5, 9, len(prevLayer))
    filterAmount = 0
    prevData = 0
    for i in range(100):
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
            currLayer.slide(True,False)
        else:
            currLayer.slide(False,False)

        if i%25 == 0:
            print(i)
            Viz.setInputs(currLayer.input,currLayer.filter,currLayer.featureMaps.flatten(),None,currLayer.channels)
            Viz.visualizeNet()

    Viz.endViz()
    layer.append(currLayer)

trainConvLayer(layer,CL1)
trainConvLayer(layer,CL2)
trainConvLayer(layer,CL3)
