from ConvAutoencoder.ConvLayerViz import ConvLayerViz
from ConvAutoencoder.LinearSystem import LinearSystem
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd


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

cifar = unpickle("C:/Users/Dennis/Documents/studium/cifar-10-batches-py/data_batch_1")
cifarLabelNames = unpickle("C:/Users/Dennis/Documents/studium/cifar-10-batches-py/batches.meta")
cifarLabels = cifar[b'labels']
print(cifarLabelNames)
cifarTest = unpickle("C:/Users/Dennis/Documents/studium/cifar-10-batches-py/test_batch")
cifarTest = cifarTest[b'data']
cifarLabelNames = cifarLabelNames[b'label_names']
print(cifarLabelNames)
print(cifarLabels)
print(cifarLabelNames[0].decode('utf-8'))
print(len(cifarLabelNames))
cifar = cifar[b'data']
cifarTargets = createTargets(cifarLabels)
cifarTest = cifarTest.reshape(10000,3,32,32).transpose([0,2,3,1])
cifarTest = cifarTest.reshape(10000,3072)
cifar = cifar.reshape(10000,3,32,32).transpose([0,2,3,1])
cifar = cifar.reshape(10000,3072)

#normalization
cifarN = np.zeros((10000,3072))
for i in range(10000):
    cifarN[i] = (cifar[i] / 255.0 * 0.99) + 0.01

cifar = cifarN

#single_img_reshaped = np.transpose(np.reshape(np.array(cifar[0]),(3, 32,32)), (1,2,0))
#single_img = cifar[5].reshape(3,32,32).transpose([1,2,0]).flatten()

prevLayer = []
#=== 3 LAYER CONVOLUTIONAL AUTOENCODER INITIALIZATION ===
print("Started feature learning ...")
start = time.time()
iterations = 10000
#input, channels, filterSize, filterAmount, stride, learnRate
#0.0000001
CL1 = ConvLayerViz(cifar[0],3, 9, 5, 2, 0.005)
CL2 = ConvLayerViz(CL1.featureMaps.flatten(),CL1.filterAmount, 9, 6, 2, 0.0005)
CL3 = ConvLayerViz(CL2.featureMaps.flatten(),CL2.filterAmount, 9, 7, 2, 0.00005)

CL1.setBiasVisible(1)
CL1.setBiasesFMs(np.full(CL1.filterAmount,1))
CL1.trainConvLayer(prevLayer,CL1,50,cifar)
prevLayer.append(CL1)
CL2.setBiasVisible(1)
CL2.setBiasesFMs(np.full(CL2.filterAmount,1))
CL2.trainConvLayer(prevLayer,CL2,250,cifar)
prevLayer.append(CL2)
CL3.setBiasVisible(1)
CL3.setBiasesFMs(np.full(CL3.filterAmount,1))
CL3.trainConvLayer(prevLayer,CL3,1200,cifar)
prevLayer.append(CL3)

#CONFUSION MATRIX
"""
x = np.arange(iterations)
p = sp.polyfit(x, CL1.error, deg=10)
errorL1 = sp.polyval(p, x)
p = sp.polyfit(x, CL2.error, deg=15)
errorL2 = sp.polyval(p, x)
p = sp.polyfit(x, CL3.error, deg=15)
errorL3 = sp.polyval(p, x)

plt.plot(errorL1,label="erste Schicht")
plt.plot(errorL2,label="zweite Schicht",linestyle='dashed')
plt.plot(errorL3,label="dritte Schicht",linestyle='dotted')
plt.xlabel('Trainingsmuster (CIFAR-10)')
plt.ylabel('Mittlere Quadratische Fehler')
plt.legend()
plt.show()
"""
end = time.time()
print("Finished feature learning: ",round(end-start,2),"s")

#=== TRAINING ===

ls = LinearSystem(len(CL3.featureMaps.flatten()),10)
print("Started dense-layer training ...")
for i in range(10000):
    CL1.updateInput(cifar[i])
    CL1.slide(False)
    CL2.updateInput(CL1.featureMaps.flatten())
    CL2.slide(False)
    CL3.updateInput(CL2.featureMaps.flatten())
    CL3.slide(False)

    ls.setData(CL3.featureMaps.flatten(),cifarTargets[i,:])
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
iterations = 10000
for i in range(iterations):
    CL1.updateInput(cifar[i])
    CL1.slide(False)
    CL2.updateInput(CL1.featureMaps.flatten())
    CL2.slide(False)
    CL3.updateInput(CL2.featureMaps.flatten())
    CL3.slide(False)

    #temp = np.argmax(cifarTargets[i,:])
    temp = cifarLabels[i]

    result = np.argmax(ls.run(CL3.featureMaps.flatten()))
    if result != temp:
        falsePredicted += 1
    y_actu.append(cifarLabelNames[temp].decode('utf-8'))
    y_pred.append(cifarLabelNames[result].decode('utf-8'))
correct = iterations - falsePredicted
temp = correct/iterations*100

print("correct predicted: ",temp,"%")

#plt.figure(figsize = (12,12))

#names = [x.decode('utf-8') for x in cifarLabelNames]

#df_cm = pd.DataFrame(confusion_matrix(y_actu, y_pred), names, names)
#sn.set(font_scale=1.4)#for label size
#sn.heatmap(df_cm, annot=True,annot_kws={"size": 12},cmap=sn.cm.rocket_r,fmt="d",square=True)# font size
#sn.cubehelix_palette(8,reverse=True)
#plt.xlabel("vorhergesagt")
#plt.ylabel("tatsächlich")
#plt.savefig("C:/Users/Dennis/Documents/studium/confMatrix.png")
#plt.show()