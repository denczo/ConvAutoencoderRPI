from ConvAutoencoder.ConvLayerViz import ConvLayerViz
from ConvAutoencoder.LinearSystem import LinearSystem
import numpy as np

#=== MNIST DATA PREPARATION ===
training_data_file = open("C:/Users/Dennis/Documents/studium/mnist_train.csv",'r')
data = training_data_file.readlines()
training_data_file.close()

test_data_file = open("C:/Users/Dennis/Documents/studium/mnist_test.csv",'r')
dataTest = test_data_file.readlines()
test_data_file.close()

prevLayer = []
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

#=== 2 LAYER CONVOLUTIONAL AUTOENCODER INITIALIZATION ===
#input, channels, filterSize, filterAmount, stride, learnRate
CL1 = ConvLayerViz(dataBatch[0], 1, 9, 4, 3, 0.05)
CL2 = ConvLayerViz(CL1.featureMaps.flatten(),CL1.filterAmount, 9, 12, 3, 0.005)
CL1.setBiasVisible(1)
CL1.setBiasesFMs(np.full(CL1.filterAmount,1))
CL1.trainConvLayer(prevLayer,CL1,50,dataBatch)
prevLayer.append(CL1)
CL2.setBiasVisible(1)
CL2.setBiasesFMs(np.full(CL2.filterAmount,1))
CL2.trainConvLayer(prevLayer,CL2,250,dataBatch)
prevLayer.append(CL2)

#CL2.guidedBackwardsActivation(CL2.featureMaps.reshape(CL2.filterAmount,CL2.fStepsOneAxis**2),16)
#temp = CL2.reconstrInput.reshape(CL1.fStepsOneAxis,CL1.fStepsOneAxis,CL1.filterAmount).transpose(2,0,1)
#CL1.guidedBackwardsActivation(temp.reshape(CL1.filterAmount,CL1.fStepsOneAxis**2),2)
#Viz = vizLayer(CL1, 5, 9, 0)
#Viz.setInputs(CL1.input,CL1.filter,CL1.featureMaps.flatten(),CL1.reconstrInput,CL1.channels)
#Viz.visualizeFeatures(CL1.reconstrInput,True)

#=== TRAINING ===
ls = LinearSystem(len(CL2.featureMaps.flatten()),10)
print("Started training ...")
for i in range(dataTestLength):
    CL1.updateInput(dataBatch[i])
    CL1.slide(False)
    CL2.updateInput(CL1.featureMaps.flatten())
    CL2.slide(False)

    ls.setData(CL2.featureMaps.flatten(),targets[i,:])
    ls.train()

ls.solveLS()

#=== TEST ===
falsePredicted = 0
print("Started test ...")
for i in range(dataTestLength):
    CL1.updateInput(testBatch[i])
    CL1.slide(False)
    CL2.updateInput(CL1.featureMaps.flatten())
    CL2.slide(False)

    temp = np.argmax(testTargets[i,:])
    result = np.argmax(ls.run(CL2.featureMaps.flatten()))
    if result != temp:
        falsePredicted += 1

correct = dataTestLength - falsePredicted
temp = correct/dataTestLength*100

print("correct predicted: ",temp,"%")