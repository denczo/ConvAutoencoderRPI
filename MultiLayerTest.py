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

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
cifar = unpickle("C:/Users/Dennis/Documents/studium/cifar-10-batches-py/data_batch_1")
cifar = cifar[b'data']

#single_img_reshaped = np.transpose(np.reshape(np.array(cifar[0]),(3, 32,32)), (1,2,0))
single_img = cifar[5].reshape(3,32,32).transpose([1,2,0]).flatten()
def vizLayer(cl,height,width,name):
    fSizeAxis = int(math.sqrt(cl.filterSize))
    fmSize = cl.fStepsOneAxis ** 2
    viz = Viz(height,width,cl.filterAmount,fmSize,fSizeAxis,cl.fStepsOneAxis,name)
    return viz

#img=mpimg.imread('images/XWing32x32.png')

img=mpimg.imread('images/trooper32.png')
#img=np.append(img[:,:,:1].flatten(),np.append(img[:,:,1:2],img[:,:,2:3]))
#img=img[:,:,1:2].flatten()
print(img.shape)
channels = 3
img=img.flatten()
filterAmount = 9
layer = []
#input, channels, filterSize, filterAmount, stride, learnRate
CL1 = ConvLayerLight(single_img,channels, 9, 6, 2, 0.00000001)

#CL1 = ConvLayerLight(img,channels, 9, 5, 3, 0.01)
#CL1 = ConvLayerLight(dataBatch[0],channels, 9, 4, 3, 0.05)

CL2 = ConvLayerLight(CL1.featureMaps.flatten(),CL1.filterAmount, 9, 16, 2, 0.00000001)
CL3 = ConvLayerLight(CL2.featureMaps.flatten(),CL2.filterAmount, 9, 24, 1, 0.0001)

def trainConvLayer(prevLayer,currLayer,epochs):

    Viz = vizLayer(currLayer, 6, 14, len(prevLayer))
    filterAmount = 0
    prevData = 0
    for i in range(epochs):
        if len(prevLayer) <= 0:
            filterAmount = currLayer.filterAmount
            #prevOut = dataBatch[i]
            #prevOut = img
            single_img_reshaped = cifar[i].reshape(3,32,32).transpose([1,2,0]).flatten()

            prevOut = single_img_reshaped
        else:
            filterAmount = prevLayer[-1].filterAmount
            #prevLayer[0].updateInput(dataBatch[i])
            #prevLayer[0].updateInput(img)
            single_img_reshaped = cifar[i].reshape(3,32,32).transpose([1,2,0]).flatten()

            prevLayer[0].updateInput(single_img_reshaped)

            prevLayer[0].slide(False)
            prevOut = prevLayer[-1].featureMaps.flatten('F')

        for j in range(len(prevLayer)-1):
            prevData = prevLayer[j].featureMaps.flatten('F')
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

CL1.setBiasVisible(1)
CL1.setBiasesFMs(np.full(CL1.filterAmount,1))
trainConvLayer(layer,CL1,200)
CL2.setBiasVisible(1)
CL2.setBiasesFMs(np.full(CL2.filterAmount,0.1))
trainConvLayer(layer,CL2,150)
#CL3.setBiasVisible(1)
#CL3.setBiasesFMs(np.full(CL3.filterAmount,0.1))
#trainConvLayer(layer,CL3,150)

#CL2.guidedBackwardsActivation(CL2.featureMaps.reshape(CL2.filterAmount,CL2.fStepsOneAxis**2),16)
CL1.guidedBackwardsActivation(CL1.featureMaps.reshape(CL1.filterAmount,CL1.fStepsOneAxis**2),5)
Viz = vizLayer(CL1, 5, 9, 0)
Viz.setInputs(CL1.input,CL1.filter,CL1.featureMaps.flatten(),CL1.reconstrInput,CL1.channels)
Viz.visualizeFeatures(CL1.reconstrInput,True)
