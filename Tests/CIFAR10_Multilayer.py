from ConvAutoencoder.ConvLayerLight import ConvLayerLight
from ConvAutoencoder.Visualizer import Viz
from LinearSystem.LinearSystem import LinearSystem
import numpy as np
import math

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def createTargets(labels):
    labelsLength = len(labels)
    targets = np.zeros((labelsLength,10)) + 0.01
    for i in range(labelsLength):
        targets[i,[labels[i]]] = 0.99
    return targets

def vizLayer(cl,height,width,name):
    fSizeAxis = int(math.sqrt(cl.filterSize))
    fmSize = cl.fStepsOneAxis ** 2
    viz = Viz(height,width,cl.filterAmount,fmSize,fSizeAxis,cl.fStepsOneAxis,name)
    return viz


cifar = unpickle("C:/Users/Dennis/Documents/studium/cifar-10-batches-py/data_batch_1")
cifarLabels = cifar[b'labels']
cifar = cifar[b'data']
cifarTargets = createTargets(cifarLabels)


#single_img_reshaped = np.transpose(np.reshape(np.array(cifar[0]),(3, 32,32)), (1,2,0))
single_img = cifar[5].reshape(3,32,32).transpose([1,2,0]).flatten()
channels = 3
filterAmount = 9
layer = []
#input, channels, filterSize, filterAmount, stride, learnRate
CL1 = ConvLayerLight(single_img,channels, 9, 4, 3, 0.00000001)
CL2 = ConvLayerLight(CL1.featureMaps.flatten(),CL1.filterAmount, 9, 12, 2, 0.00000001)
CL3 = ConvLayerLight(CL2.featureMaps.flatten(),CL2.filterAmount, 9, 32, 2, 0.00000001)

def trainConvLayer(prevLayer,currLayer,epochs):

    Viz = vizLayer(currLayer, 6, 14, len(prevLayer))
    filterAmount = 0
    prevData = 0
    for i in range(epochs):
        if len(prevLayer) <= 0:
            filterAmount = currLayer.filterAmount
            single_img_reshaped = cifar[i].reshape(3,32,32).transpose([1,2,0]).flatten()

            prevOut = single_img_reshaped
        else:
            filterAmount = prevLayer[-1].filterAmount
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

        if i%5 == 0:
            print(i)
            Viz.setInputs(currLayer.input,currLayer.filter,currLayer.featureMaps.flatten(),currLayer.reconstrInput,currLayer.channels)
            Viz.visualizeNet(False)

    Viz.endViz()
    layer.append(currLayer)

CL1.setBiasVisible(0)
CL1.setBiasesFMs(np.full(CL1.filterAmount,1))
trainConvLayer(layer,CL1,51)
CL2.setBiasVisible(0)
CL2.setBiasesFMs(np.full(CL2.filterAmount,1))
trainConvLayer(layer,CL2,51)
CL3.setBiasVisible(9)
CL3.setBiasesFMs(np.full(CL3.filterAmount,1))
trainConvLayer(layer,CL3,51)

#CL2.guidedBackwardsActivation(CL2.featureMaps.reshape(CL2.filterAmount,CL2.fStepsOneAxis**2),16)
#temp = CL2.reconstrInput.reshape(CL1.fStepsOneAxis,CL1.fStepsOneAxis,CL1.filterAmount).transpose(2,0,1)
#CL1.guidedBackwardsActivation(temp.reshape(CL1.filterAmount,CL1.fStepsOneAxis**2),2)
#Viz = vizLayer(CL1, 5, 9, 0)
#Viz.setInputs(CL1.input,CL1.filter,CL1.featureMaps.flatten(),CL1.reconstrInput,CL1.channels)
#Viz.visualizeFeatures(CL1.reconstrInput,True)

ls = LinearSystem(len(CL3.featureMaps.flatten()),10)
for i in range(1000):
    CL1.updateInput(cifar[i].reshape(3,32,32).transpose([1,2,0]).flatten())
    CL1.slide(False)
    CL2.updateInput(CL1.featureMaps.flatten())
    CL2.slide(False)
    CL3.updateInput(CL2.featureMaps.flatten())
    CL3.slide(False)

    ls.setData(CL3.featureMaps.flatten(),cifarTargets[i,:])
    ls.train()

ls.solveLS()
print(cifarTargets[999,:])
ls.run(CL3.featureMaps.flatten())
