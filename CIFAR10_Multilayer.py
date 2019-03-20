from ConvAutoencoder.ConvLayerViz import ConvLayerViz
from ConvAutoencoder.LinearSystem import LinearSystem
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
#input, channels, filterSize, filterAmount, stride, learnRate
CL1 = ConvLayerViz(cifar[0],3, 9, 6, 3, 0.000000005)
CL2 = ConvLayerViz(CL1.featureMaps.flatten(),CL1.filterAmount, 9, 12, 3, 0.000000005)
#CL3 = ConvLayerViz(CL2.featureMaps.flatten(),CL2.filterAmount, 9, 32, 2, 0.00000001)


CL1.setBiasVisible(1)
CL1.setBiasesFMs(np.full(CL1.filterAmount,1))
CL1.trainConvLayer(prevLayer,CL1,75,cifar)
prevLayer.append(CL1)
CL2.setBiasVisible(1)
CL2.setBiasesFMs(np.full(CL2.filterAmount,1))
CL2.trainConvLayer(prevLayer,CL2,250,cifar)
prevLayer.append(CL2)
#CL3.setBiasVisible(1)
#CL3.setBiasesFMs(np.full(CL3.filterAmount,1))
#CL3.trainConvLayer(prevLayer,CL3,250,cifar)

#CL2.guidedBackwardsActivation(CL2.featureMaps.reshape(CL2.filterAmount,CL2.fStepsOneAxis**2),16)
#temp = CL2.reconstrInput.reshape(CL1.fStepsOneAxis,CL1.fStepsOneAxis,CL1.filterAmount).transpose(2,0,1)
#CL1.guidedBackwardsActivation(temp.reshape(CL1.filterAmount,CL1.fStepsOneAxis**2),2)
#Viz = vizLayer(CL1, 5, 9, 0)
#Viz.setInputs(CL1.input,CL1.filter,CL1.featureMaps.flatten(),CL1.reconstrInput,CL1.channels)
#Viz.visualizeFeatures(CL1.reconstrInput,True)

#=== TRAINING ===
ls = LinearSystem(len(CL2.featureMaps.flatten()),10)
print("Started training ...")
for i in range(10000):
    CL1.updateInput(cifar[i])
    CL1.slide(False)
    CL2.updateInput(CL1.featureMaps.flatten())
    CL2.slide(False)
    #CL3.updateInput(CL2.featureMaps.flatten())
    #CL3.slide(False)

    ls.setData(CL2.featureMaps.flatten(),cifarTargets[i,:])
    ls.train()

ls.solveLS()


#=== TEST ===
falsePredicted = 0
print("Started test ...")
iterations = 1000
for i in range(iterations):
    CL1.updateInput(cifar[i])
    CL1.slide(False)
    CL2.updateInput(CL1.featureMaps.flatten())
    CL2.slide(False)

    temp = np.argmax(cifarTargets[i,:])
    result = np.argmax(ls.run(CL2.featureMaps.flatten()))
    if result != temp:
        falsePredicted += 1

correct = iterations - falsePredicted
temp = correct/iterations*100

print("correct predicted: ",temp,"%")
