from ConvAutoencoder.ConvLayerViz import ConvLayerViz
from ConvAutoencoder.LinearSystem import LinearSystem
import numpy as np
import time
from os.path import dirname, realpath
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

filepath = realpath(__file__)
dirOfFile = dirname(filepath)
parentDir = dirname(dirOfFile)
parentParentDir = dirname(parentDir)

#=== MNIST DATA PREPARATION ===
print("Started MNIST data preparation ...")
training_data_file = open(parentDir+"\mnist_train.csv",'r')
data = training_data_file.readlines()
training_data_file.close()

test_data_file = open(parentDir+"\mnist_test.csv",'r')
dataTest = test_data_file.readlines()
test_data_file.close()

prevLayer = []
dataTestLength = len(dataTest)
dataLength = len(data)
testBatch = np.zeros((dataTestLength,784))
testTargets = np.zeros((dataTestLength,10)) + 0.01
dataBatch = np.zeros((dataLength,784))
targets = np.zeros((dataLength,10)) + 0.01

#MNIST traing & test data with targets
for i in range(dataLength):
    oneDigit = data[i % dataTestLength].split(',')
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
start = time.time()
iterations = 1000
#input, channels, filterSize, filterAmount, stride, learnRate
CL1 = ConvLayerViz(dataBatch[0], 1, 9, 4, 3, 0.05)
CL2 = ConvLayerViz(CL1.featureMaps.flatten(),CL1.filterAmount, 9, 12, 3, 0.005)
CL1.setBiasVisible(1)
CL1.setBiasesFMs(np.full(CL1.filterAmount,1))
CL1.trainConvLayer(prevLayer,CL1,50,dataBatch)
errorL1 = CL1.error
prevLayer.append(CL1)
CL2.setBiasVisible(1)
CL2.setBiasesFMs(np.full(CL2.filterAmount,1))
CL2.trainConvLayer(prevLayer,CL2,250,dataBatch)
errorL2 = CL2.error
"""
x = np.arange(iterations)
p = sp.polyfit(x, errorL1, deg=10)
yL1 = sp.polyval(p, x)
p = sp.polyfit(x, errorL2, deg=15)
yL2 = sp.polyval(p, x)

plt.plot(yL1,label="erste Schicht")
plt.plot(yL2,label="zweite Schicht",linestyle='dashed')
plt.xlabel('Trainingsmuster (MNIST)')
plt.ylabel('Mittlere Quadratische Fehler')
plt.legend()
plt.show()
"""
prevLayer.append(CL2)
end = time.time()
print("Finished feature learning: ",round(end-start,2),"s")

#CL2.guidedBackwardsActivation(CL2.featureMaps.reshape(CL2.filterAmount,CL2.fStepsOneAxis**2),16)
#temp = CL2.reconstrInput.reshape(CL1.fStepsOneAxis,CL1.fStepsOneAxis,CL1.filterAmount).transpose(2,0,1)
#CL1.guidedBackwardsActivation(temp.reshape(CL1.filterAmount,CL1.fStepsOneAxis**2),2)
#Viz = vizLayer(CL1, 5, 9, 0)
#Viz.setInputs(CL1.input,CL1.filter,CL1.featureMaps.flatten(),CL1.reconstrInput,CL1.channels)
#Viz.visualizeFeatures(CL1.reconstrInput,True)

#=== TRAINING ===
ls = LinearSystem(len(CL2.featureMaps.flatten()),10)
print("Started training ...")
for i in range(20000):
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

y_actu = []
y_pred = []
falsePredicted = 0
print("Started test ...")
for i in range(10000):
    CL1.updateInput(testBatch[i])
    CL1.slide(False)
    CL2.updateInput(CL1.featureMaps.flatten())
    CL2.slide(False)

    temp = np.argmax(testTargets[i,:])
    result = np.argmax(ls.run(CL2.featureMaps.flatten()))
    if result != temp:
        falsePredicted += 1
    #print(temp, result)
    y_actu.append(temp)
    y_pred.append(result)
correct = dataTestLength - falsePredicted
temp = correct/dataTestLength*100

print("correct predicted: ",temp,"%")

plt.figure(figsize = (10,7))

df_cm = pd.DataFrame(confusion_matrix(y_actu, y_pred), range(10),
                  range(10))
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, annot=True,annot_kws={"size": 12},cmap=sn.cm.rocket_r,fmt="d",square=True)# font size
sn.cubehelix_palette(8,reverse=True)
plt.xlabel("vorhergesagt")
plt.ylabel("tatsächlich")
plt.show()