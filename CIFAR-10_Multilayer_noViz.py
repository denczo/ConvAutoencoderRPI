from ConvAutoencoder.ConvLayerViz import ConvLayer
from ConvAutoencoder.LinearSystem import LinearSystem
import numpy as np
import time

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
cifarLabels = cifar[b'labels']
cifar = cifar[b'data']
cifarTargets = createTargets(cifarLabels)
cifar = cifar.reshape(10000,3,32,32).transpose([0,2,3,1])
cifar = cifar.reshape(10000,3072)

#normalization
cifarN = np.zeros((10000,3072))
for i in range(10000):
    cifarN[i] = (cifar[i] / 255.0 * 0.99) + 0.01
cifar = cifarN

prevLayer = []
#=== 3 LAYER CONVOLUTIONAL AUTOENCODER INITIALIZATION ===
print("Started feature learning ...")
start = time.time()
#input, channels, filterSize, filterAmount, stride, learnRate
CL1 = ConvLayer(cifar[0],3, 9, 5, 2, 0.005)
CL2 = ConvLayer(CL1.featureMaps.flatten(),CL1.filterAmount, 9, 6, 2, 0.0005)
CL3 = ConvLayer(CL2.featureMaps.flatten(),CL2.filterAmount, 9, 7, 2, 0.00005)


CL1.setBiasVisible(1)
CL1.setBiasesFMs(np.full(CL1.filterAmount,1))
CL1.trainConvLayer(prevLayer,CL1,100,cifar)
prevLayer.append(CL1)
CL2.setBiasVisible(1)
CL2.setBiasesFMs(np.full(CL2.filterAmount,1))
CL2.trainConvLayer(prevLayer,CL2,250,cifar)
prevLayer.append(CL2)
CL3.setBiasVisible(1)
CL3.setBiasesFMs(np.full(CL3.filterAmount,1))
CL3.trainConvLayer(prevLayer,CL3,1000,cifar)
prevLayer.append(CL3)


end = time.time()
print("Finished feature learning: ",round(end-start,2),"s")

#=== TRAINING ===

ls = LinearSystem(len(CL3.featureMaps.flatten()),10)
print("Started dense-layer training ...")
for i in range(8000):
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
falsePredicted = 0
print("Started test ...")
iterations = 2000
for i in range(iterations):
    CL1.updateInput(cifar[i])
    CL1.slide(False)
    CL2.updateInput(CL1.featureMaps.flatten())
    CL2.slide(False)
    CL3.updateInput(CL2.featureMaps.flatten())
    CL3.slide(False)

    temp = np.argmax(cifarTargets[i,:])
    result = np.argmax(ls.run(CL3.featureMaps.flatten()))
    if result != temp:
        falsePredicted += 1

correct = iterations - falsePredicted
temp = correct/iterations*100

print("correct predicted: ",temp,"%")
