from ConvAutoencoder.ConvLayerLight import ConvLayerLight
from ConvAutoencoder.Visualizer import Viz
import numpy as np
import math

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

def vizLayer(cl,height,width):
    fSizeAxis = int(math.sqrt(cl.filterSize))
    fmSize = cl.fStepsOneAxis ** 2
    viz = Viz(height,width,filterAmount,fmSize,fSizeAxis,cl.fStepsOneAxis)
    return viz


#img=mpimg.imread('images/stardestroyer128x128.png')

#img=np.append(img[:,:,:1].flatten(),np.append(img[:,:,1:2],img[:,:,2:3]))
#CAEL = ConvLayerLight(img[:,:,:1].flatten(),9, 5, 1, 0.001)

channels = 1
filterAmount = 9
#CAEL = ConvLayerLight(img,channels, 9, filterAmount, 3, 0.001)

#input, channels, filterSize, filterAmount, stride, learnRate
CL1 = ConvLayerLight(dataBatch[0],channels, 9, filterAmount, 1, 0.01)
Viz1 = vizLayer(CL1,5,9)




for i in range(98):
    CL1.updateInput(dataBatch[i])
    if i < 100:
        CL1.slide(True)
    else:
        CL1.slide(False)

    if i%25 == 0:
        #fm = CL1.featureMaps[0,:]
        #filterT = CL1.filter[4,:].reshape(3,3)
        #filterT = filterT.T
        #fonInput = CL1.slideDeconv(fm,filterT)
        print(i)
        Viz1.setInputs(CL1.input,CL1.filter,CL1.featureMaps.flatten(),None,channels,None)
        Viz1.visualizeNet()


CL2 = ConvLayerLight(CL1.featureMaps.flatten(),CL1.filterAmount, 9, 9, 2, 0.001)
Viz2 = vizLayer(CL2,10,18)
for i in range(105):
    CL1.updateInput(dataBatch[i])
    CL1.slide(False)

    CL2.updateInput(CL1.featureMaps.flatten())
    if i < 100:
        CL2.slide(True)
    else:
        CL2.slide(False)

    if i%25 == 0:
        print(i)
        Viz2.setInputs(CL2.input,CL2.filter,CL2.featureMaps.flatten(),None,CL1.filterAmount, None)
        Viz2.visualizeNet()


CL3 = ConvLayerLight(CL2.featureMaps.flatten(),CL2.filterAmount, 9, 9, 2, 0.001)
Viz3 = vizLayer(CL3,10,18)

for i in range(500):
    CL1.updateInput(dataBatch[i])
    CL1.slide(False)
    CL2.updateInput(CL1.featureMaps.flatten())
    CL2.slide(False)
    CL3.updateInput(CL2.featureMaps.flatten())

    if i < 500:
        CL3.slide(True)
    else:
        CL3.slide(False)

    if i%25 == 0:
        print(i)
        Viz3.setInputs(CL3.input,CL3.filter,CL3.featureMaps.flatten(),None,CL2.filterAmount, None)
        Viz3.visualizeNet()

