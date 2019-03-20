from ConvAutoencoder.ConvLayerLight import ConvLayerLight
from ConvAutoencoder.Visualizer import Viz
from ConvAutoencoder.LinearSystem import LinearSystem
import numpy as np
import math

training_data_file = open("C:/Users/Dennis/Documents/studium/mnist_train.csv",'r')
data = training_data_file.readlines()
training_data_file.close()

test_data_file = open("C:/Users/Dennis/Documents/studium/mnist_test.csv",'r')
dataTest = test_data_file.readlines()
test_data_file.close()

layer = []
dataTestLength = len(dataTest)
#dataLength = len(data)
testBatch = np.zeros((dataTestLength,784))
testTargets = np.zeros((dataTestLength,10)) + 0.01
dataBatch = np.zeros((dataTestLength,784))
targets = np.zeros((dataTestLength,10)) + 0.01

#MNIST traing & test data with targets
for i in range(dataTestLength):
    oneDigit = data[i % dataTestLength].split(',')
    oneDigitTest = dataTest[i % dataTestLength].split(',')

    temp = (np.asfarray(oneDigit[1:]) / 255.0 * 0.99) + 0.01
    tempTest = (np.asfarray(oneDigitTest[1:]) / 255.0 * 0.99) + 0.01

    dataBatch[i] = temp
    targets[i,int(oneDigit[0])] = 0.99

    testBatch[i] = tempTest
    testTargets[i, int(oneDigitTest[0])] = 0.99

#vizualisation setup
def vizLayer(cl,height,width,name):
    fSizeAxis = int(math.sqrt(cl.filterSize))
    fmSize = cl.fStepsOneAxis ** 2
    viz = Viz(height,width,cl.filterAmount,fmSize,fSizeAxis,cl.fStepsOneAxis,name)
    return viz

def trainConvLayer(prevLayer,currLayer,iterations):

    Viz = vizLayer(currLayer, 6, 14, len(prevLayer))
    for i in range(iterations):
        if len(prevLayer) <= 0:
            prevOut = dataBatch[i]
        else:
            prevLayer[0].updateInput(dataBatch[i])
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


channels = 1
filterAmount = 9
#input, channels, filterSize, filterAmount, stride, learnRate
CL1 = ConvLayerLight(dataBatch[0],channels, 9, 4, 3, 0.05)
CL2 = ConvLayerLight(CL1.featureMaps.flatten(),CL1.filterAmount, 9, 12, 3, 0.005)

CL1.setBiasVisible(1)
CL1.setBiasesFMs(np.full(CL1.filterAmount,1))
trainConvLayer(layer,CL1,50)
CL2.setBiasVisible(1)
CL2.setBiasesFMs(np.full(CL2.filterAmount,1))
trainConvLayer(layer,CL2,250)

#CL2.guidedBackwardsActivation(CL2.featureMaps.reshape(CL2.filterAmount,CL2.fStepsOneAxis**2),16)
#temp = CL2.reconstrInput.reshape(CL1.fStepsOneAxis,CL1.fStepsOneAxis,CL1.filterAmount).transpose(2,0,1)
#CL1.guidedBackwardsActivation(temp.reshape(CL1.filterAmount,CL1.fStepsOneAxis**2),2)
#Viz = vizLayer(CL1, 5, 9, 0)
#Viz.setInputs(CL1.input,CL1.filter,CL1.featureMaps.flatten(),CL1.reconstrInput,CL1.channels)
#Viz.visualizeFeatures(CL1.reconstrInput,True)


ls = LinearSystem(len(CL2.featureMaps.flatten()),10)

print("Started training ...")
#TRAINING
for i in range(dataTestLength):
    CL1.updateInput(dataBatch[i])
    CL1.slide(False)
    CL2.updateInput(CL1.featureMaps.flatten())
    CL2.slide(False)

    ls.setData(CL2.featureMaps.flatten(),targets[i,:])
    ls.train()

ls.solveLS()

falsePredicted = 0

print("Started test ...")
#TEST
for i in range(dataTestLength):
    CL1.updateInput(testBatch[i])
    CL1.slide(False)
    CL2.updateInput(CL1.featureMaps.flatten())
    CL2.slide(False)

    temp = np.argmax(testTargets[i,:])
    result = np.argmax(ls.run(CL2.featureMaps.flatten()))
    if result != temp:
        falsePredicted += 1
    #print("exspected: ",temp)
    #print("actual: ",result)
    #print("")

correct = dataTestLength - falsePredicted
temp = correct/dataTestLength*100

print("correct predicted: ",temp,"%")