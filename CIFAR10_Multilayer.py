from ConvAutoencoder.ConvLayerViz import ConvLayerViz
from ConvAutoencoder.LinearSystem import LinearSystem
import time
import numpy as np

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
#single_img_reshaped = np.transpose(np.reshape(np.array(cifar[0]),(3, 32,32)), (1,2,0))
#single_img = cifar[5].reshape(3,32,32).transpose([1,2,0]).flatten()

prevLayer = []
#=== 3 LAYER CONVOLUTIONAL AUTOENCODER INITIALIZATION ===
print("Started feature learning ...")
start = time.time()
#input, channels, filterSize, filterAmount, stride, learnRate
CL1 = ConvLayerViz(cifar[0],3, 9, 5, 2, 0.0000001)
CL2 = ConvLayerViz(CL1.featureMaps.flatten(),CL1.filterAmount, 9, 6, 2, 0.00000001)
CL3 = ConvLayerViz(CL2.featureMaps.flatten(),CL2.filterAmount, 9, 7, 2, 0.000000001)
#CL4 = ConvLayerViz(CL3.featureMaps.flatten(),CL3.filterAmount, 9, 30, 1, 0.000000001)

CL1.setBiasVisible(1)
CL1.setBiasesFMs(np.full(CL1.filterAmount,1))
CL1.trainConvLayer(prevLayer,CL1,150,cifar)
prevLayer.append(CL1)
CL2.setBiasVisible(1)
CL2.setBiasesFMs(np.full(CL2.filterAmount,1))
CL2.trainConvLayer(prevLayer,CL2,500,cifar)
prevLayer.append(CL2)
CL3.setBiasVisible(1)
CL3.setBiasesFMs(np.full(CL3.filterAmount,1))
CL3.trainConvLayer(prevLayer,CL3,500,cifar)
prevLayer.append(CL3)
#CL4.setBiasVisible(1)
#CL4.setBiasesFMs(np.full(CL4.filterAmount,-1))
#CL4.trainConvLayer(prevLayer,CL4,150,cifar)
end = time.time()
print("Finished feature learning: ",round(end-start,2),"s")


#CL2.guidedBackwardsActivation(CL2.featureMaps.reshape(CL2.filterAmount,CL2.fStepsOneAxis**2),16)
#temp = CL2.reconstrInput.reshape(CL1.fStepsOneAxis,CL1.fStepsOneAxis,CL1.filterAmount).transpose(2,0,1)
#CL1.guidedBackwardsActivation(temp.reshape(CL1.filterAmount,CL1.fStepsOneAxis**2),2)
#Viz = vizLayer(CL1, 5, 9, 0)
#Viz.setInputs(CL1.input,CL1.filter,CL1.featureMaps.flatten(),CL1.reconstrInput,CL1.channels)
#Viz.visualizeFeatures(CL1.reconstrInput,True)

#=== TRAINING ===

ls = LinearSystem(len(CL3.featureMaps.flatten()),10)
print("Started dense-layer training ...")
for i in range(1000):
    CL1.updateInput(cifar[i])
    CL1.slide(False)
    CL2.updateInput(CL1.featureMaps.flatten())
    CL2.slide(False)
    CL3.updateInput(CL2.featureMaps.flatten())
    CL3.slide(False)
    #CL4.updateInput(CL3.featureMaps.flatten())
    #CL4.slide(False)

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
iterations = 1000
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
