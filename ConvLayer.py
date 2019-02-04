import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#from scipy import sparse

class ConvLayer:

    #only for cubic images
    def __init__(self, filterSize, filterAmount, stride, learnRate):
        self.filter = np.random.uniform(low=-0.01, high=0.01, size=(filterAmount, filterSize))
        self.stride = stride
        self.lernRate = learnRate
        self.filterAmount = filterAmount
        self.filterSize = filterSize
        self.reLu = lambda x: x * (x > 0)

    def setInput(self, input, channels):
        self.inputSize = int(len(input)/channels)
        # one channel 1D = inputSize
        self.input = input
        self.channels = channels
        self.reconstr = np.zeros(self.inputSize*3)
        # how many steps filter can slide in X or Y direction over image
        self.fStepsOneAxis = int((math.sqrt(self.inputSize) - math.sqrt(self.filterSize)) / self.stride) + 1
        self.featureMaps = np.zeros((self.filterAmount * (self.fStepsOneAxis) ** 2))
        self.featureMaps.shape = (1, len(self.featureMaps))
        self.convMatrix = np.zeros((((self.fStepsOneAxis) ** 2) * self.filterAmount, self.inputSize*channels))
        self.createConvMatrix()

    def createConvMatrix(self):
        filterSizeX = int(math.sqrt(self.filterSize))
        inputSizeX = int(math.sqrt(self.inputSize))
        allSteps = self.fStepsOneAxis ** 2

        for filter in range(self.filterAmount):
            f = self.filter[filter]
            f.shape = (filterSizeX,filterSizeX)
            for step in range(allSteps):
                x = step % self.fStepsOneAxis
                y = int(step / self.fStepsOneAxis)
                temp = np.zeros((inputSizeX,inputSizeX))
                temp[y:filterSizeX+y,x:filterSizeX+x] = f
                temp.shape = (1,self.inputSize)
                self.convMatrix[step+filter*allSteps,0:self.inputSize] = temp
        temp = self.convMatrix[0:filter*allSteps,0:self.inputSize]
        for channel in range(self.channels):
            self.convMatrix[0:filter*allSteps,channel*self.inputSize:(channel+1)*self.inputSize] = temp

    def convolution(self, convMatrix, input):
        return np.dot(convMatrix,input)

    #it's a transposed convolution, not a mathematical deconvolution
    def deconvolution(self, convMatrix, featureMap):
        return np.dot(convMatrix.T,featureMap.T)

    def forwardActivation(self):
        temp = self.convolution(self.convMatrix,self.input)
        self.featureMaps = self.reLu(np.add(self.featureMaps,temp))

    def backwardsActivation(self):
        temp = self.reLu(self.deconvolution(self.convMatrix, self.featureMaps))
        self.reconstr = temp

    def contrastiveDivergence(self):
        temp = np.subtract(self.input.T, self.reconstr.T)
        self.convMatrix = np.add(self.convMatrix, self.lernRate * np.dot(self.featureMaps.T,temp))

img=mpimg.imread('stardestroyer32x32.png')
img=np.append(img[:,:,:1].flatten(),np.append(img[:,:,1:2],img[:,:,2:3]))
fig = plt.figure(figsize=(10, 10))

#filterSize, filterAmount, stride, learnRate
CAE = ConvLayer(9,9,1,0.000001)
CAE.setInput(img,3)

for i in range(100):
    if i%25 == 0:
        print(i)
        fig.clear()
        for i in range(CAE.filterAmount):
            learned = CAE.featureMaps.T
            learned = learned[i*900:(i+1)*900]
            learned = learned.reshape(30, 30)
            fig.add_subplot(1, 9, i + 1)
            plt.axis('off')
            plt.imshow(learned, interpolation='None')

        for i in range(CAE.channels):
            fig.add_subplot(3, 3, i+1)
            real = CAE.input[i*CAE.inputSize:(i+1)*CAE.inputSize]
            real = real.reshape(32, 32)
            plt.axis('off')
            plt.imshow(real, interpolation='None')

        for i in range(CAE.channels):
            fig.add_subplot(3, 3, i+7)
            real = CAE.reconstr[i*CAE.inputSize:(i+1)*CAE.inputSize]
            real = real.reshape(32, 32)
            plt.axis('off')
            plt.imshow(real, interpolation='None')
        plt.draw()
        plt.pause(0.0001)

    CAE.forwardActivation()
    CAE.backwardsActivation()
    CAE.contrastiveDivergence()
plt.show()
