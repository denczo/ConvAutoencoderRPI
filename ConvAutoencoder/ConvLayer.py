import numpy as np
import math
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

    def updateInput(self,input):
        self.input = input

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

    def readFilter(self):
        filterSizeX = int(math.sqrt(self.filterSize))
        inputSizeX = int(math.sqrt(self.inputSize))
        allSteps = self.fStepsOneAxis ** 2
        for f in range(self.filterAmount):
            self.filter[0]
            tempFilter = 0
            for step in range(allSteps):
                x = step % self.fStepsOneAxis
                y = int(step / self.fStepsOneAxis)
                tempMx = self.convMatrix[allSteps*f+step].reshape(inputSizeX,inputSizeX,self.channels)
                tempFilter = np.add(tempFilter,tempMx[y:filterSizeX+y,x:filterSizeX+x,0])
            self.filter[f] = tempFilter.reshape(1,self.filterSize)

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

