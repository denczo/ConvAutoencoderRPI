from ConvAutoencoder.ConvLayer import ConvLayer
#from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
#import os


#os.chdir(os.path.dirname(__file__))
#path = Path(os.getcwd()).parent

training_data_file = open("C:/Users/Dennis/Documents/studium/mnist_train.csv",'r')
data = training_data_file.readlines()
training_data_file.close()

dataLength = len(data)
amountData = 1000
dataBatch = np.zeros((amountData,784))

for i in range(amountData):
    oneDigit = data[i % dataLength].split(',')
    temp = (np.asfarray(oneDigit[1:]) / 255.0 * 0.99) + 0.01
    dataBatch[i] = temp

img=mpimg.imread('images/XWing32x32.png')
img=np.append(img[:,:,:1].flatten(),np.append(img[:,:,1:2],img[:,:,2:3]))

#filterSize, filterAmount, stride, learnRate
CAE = ConvLayer(9, 5, 1, 0.0001)
CAE.setInput(img,3)
fmSize = CAE.fStepsOneAxis ** 2
fig = plt.figure(figsize=(8, 8))


for i in range(500):
    #CAE.updateInput(dataBatch[i])

    if i%499 == 0:
        fig.clear
        print(i)

        for i in range(CAE.filterAmount):
            learned = CAE.featureMaps.T
            learned = learned[i * fmSize:(i + 1) * fmSize]
            learned = learned.reshape(CAE.fStepsOneAxis, CAE.fStepsOneAxis)
            fig.add_subplot(5, 5, 10+i + 1)
            plt.axis('off')
            plt.imshow(learned, interpolation='None')

        CAE.readFilter()
        for i in range(CAE.filterAmount):
            fig.add_subplot(5, 5, 5+i + 1)
            filter = CAE.filter[i]
            filter = filter.reshape(3,3)
            plt.axis('off')
            plt.imshow(filter, interpolation='None')

        for i in range(CAE.channels):
            fig.add_subplot(5, 5, 2+i + 1)
            real = CAE.input[i * CAE.inputSize:(i + 1) * CAE.inputSize]
            real = real.reshape(32, 32)
            plt.axis('off')
            plt.imshow(real, interpolation='None')

        for i in range(CAE.channels):
            fig.add_subplot(5, 5, 18+i)
            real = CAE.reconstr[i * CAE.inputSize:(i + 1) * CAE.inputSize]
            real = real.reshape(32, 32)
            plt.axis('off')
            plt.imshow(real, interpolation='None')
        CAE.readFilter()
        plt.draw()
        plt.pause(0.0001)

    CAE.forwardActivation()
    CAE.backwardsActivation()
    CAE.contrastiveDivergence()
plt.show()
