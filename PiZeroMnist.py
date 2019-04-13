from ConvAutoencoder.ConvLayer import ConvLayer
from ConvAutoencoder.LinearSystem import LinearSystem
import numpy as np
import time
from os.path import dirname, realpath


filepath = realpath(__file__)
dirOfFile = dirname(filepath)
parentDir = dirname(dirOfFile)
parentParentDir = dirname(parentDir)

#=== MNIST DATA PREPARATION ===
print("Started MNIST data preparation ...")
training_data_file = open(parentDir+"/mnist_train.csv",'r')
data = training_data_file.readlines()
training_data_file.close()

test_data_file = open(parentDir+"/mnist_test.csv",'r')
dataTest = test_data_file.readlines()
test_data_file.close()

prevLayer = []
dataTestLength = len(dataTest)
testBatch = np.zeros((dataTestLength,784))
testTargets = np.zeros((dataTestLength,10)) + 0.01
#dataLength = len(data)
dataLength = dataTestLength
dataBatch = np.zeros((dataLength,784))
targets = np.zeros((dataLength,10)) + 0.01

#MNIST traing & test data with targets
for i in range(dataLength):
    oneDigit = data[i % dataLength].split(',')
    temp = (np.asfarray(oneDigit[1:]) / 255.0 * 0.99) + 0.01
    dataBatch[i] = temp
    targets[i,int(oneDigit[0])] = 0.99
    if i < dataTestLength:
        oneDigitTest = dataTest[i % dataTestLength].split(',')
        tempTest = (np.asfarray(oneDigitTest[1:]) / 255.0 * 0.99) + 0.01
        testBatch[i] = tempTest
        testTargets[i, int(oneDigitTest[0])] = 0.99

#=== 2 LAYER CONVOLUTIONAL AUTOENCODER INITIALIZATION ===
print("Started feature learning ...")
epochs = 4
start = time.time()
#input, channels, filterSize, filterAmount, stride, learnRate
CL1 = ConvLayer(dataBatch[0], 1, 9, 4, 3, 0.05)
CL2 = ConvLayer(CL1.featureMaps.flatten(),CL1.filterAmount, 9, 12, 3, 0.005)

CL1.setBiasVisible(1)
CL1.setBiasesFMs(np.full(CL1.filterAmount,1))
for epoch in range(epochs):
    CL1.trainConvLayer(prevLayer,CL1,50,dataBatch)
prevLayer.append(CL1)
CL2.setBiasVisible(1)
CL2.setBiasesFMs(np.full(CL2.filterAmount,1))
for epoch in range(epochs):
    CL2.trainConvLayer(prevLayer,CL2,50,dataBatch)
prevLayer.append(CL2)
end = time.time()
print("Finished feature learning: ",round(end-start,2),"s")

#=== TRAINING ===
ls = LinearSystem(len(CL2.featureMaps.flatten()),10)
print("Started dense-layer training ...")

for i in range(2000):
    CL1.updateInput(dataBatch[i])
    CL1.slide(False)
    CL2.updateInput(CL1.featureMaps.flatten())
    CL2.slide(False)
    ls.setData(CL2.featureMaps.flatten(),targets[i,:])
    ls.train()

ls.solveLS()
oldEnd = end
end = time.time()
print("Finished training: ",round(end-oldEnd,2),"s")
print("Duration entire training: ",round(end-start,2),"s")

#=== TEST ===
falsePredicted = 0
print("Started test ...")
for i in range(400):
    CL1.updateInput(testBatch[i])
    CL1.slide(False)
    CL2.updateInput(CL1.featureMaps.flatten())
    CL2.slide(False)

    temp = np.argmax(testTargets[i,:])
    result = np.argmax(ls.run(CL2.featureMaps.flatten()))
    if result != temp:
        falsePredicted += 1
        #print("Exspected: ", temp, " Actual: ", result, "<= WRONG")
    else:
        #print("Exspected: ", temp, " Actual: ", result)
        pass
correct = dataTestLength - falsePredicted
temp = correct/dataTestLength*100

print("correct predicted: ",temp,"%")
