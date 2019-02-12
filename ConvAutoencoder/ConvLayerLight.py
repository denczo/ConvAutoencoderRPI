import numpy as np
import math

class ConvLayerLight:

    #only for cubic images
    def __init__(self, input, filterSize, filterAmount, stride, learnRate):
        self.filter = np.random.uniform(low=-0.01, high=0.01, size=(filterAmount, filterSize))
        self.stride = stride
        self.lernRate = learnRate
        self.filterAmount = filterAmount
        self.filterSize = filterSize
        self.hidden = np.zeros(filterAmount)
        self.reconstr = np.zeros(filterSize)
        self.reLu = lambda x: x * (x > 0)
        self.input = input
        self.inputSize = len(input)
        self.axisLength = int(math.sqrt(len(input)))
        self.fStepsOneAxis = int(((self.axisLength - math.sqrt(self.filterSize)) + 1) / self.stride)

    def forwardActivation(self,cutout):
        self.hidden = self.reLu(np.dot(self.filter,cutout))

    def backwardActivation(self):
        self.reconstr = self.reLu(np.dot(self.filter.T,self.hidden))

    def contrastiveDivergence(self,cutout):
        temp = np.subtract(cutout,self.reconstr)
        self.filter = np.add(self.filter, self.lernRate*np.dot(self.hidden,temp.T))

    def slide(self):
        temp = self.input.reshape(self.axisLength,self.axisLength)
        allSteps = self.fStepsOneAxis ** 2
        filterSizeX = int(math.sqrt(self.filterSize))

        for step in range(allSteps-1):
            x = (step+self.stride) % self.fStepsOneAxis
            y = int((step+self.stride) / self.fStepsOneAxis)
            cutout = temp[y:filterSizeX+y,x:filterSizeX+x]
            cutout = cutout.reshape(self.filterSize,1)
            self.forwardActivation(cutout)
            self.backwardActivation()
            self.contrastiveDivergence(cutout)

    def updateInput(self,input):
        self.input = input

    def createConvMatrix(self):

        filterSizeX = int(math.sqrt(self.filterSize))
        inputSizeX = self.axisLength
        allSteps = self.fStepsOneAxis ** 2
        self.convMatrix = np.zeros((((self.fStepsOneAxis) ** 2) * self.filterAmount, inputSizeX**2))
        self.featureMaps = np.zeros((self.filterAmount * (self.fStepsOneAxis) ** 2))
        self.featureMaps.shape = (1, len(self.featureMaps))
        self.recon = np.zeros(self.inputSize)

        for filter in range(self.filterAmount):
            f = self.filter[filter]
            f.shape = (filterSizeX,filterSizeX)
            for step in range(allSteps):
                x = step % self.fStepsOneAxis
                y = int(step / self.fStepsOneAxis)
                temp = np.zeros((inputSizeX,inputSizeX))
                temp[y:filterSizeX+y,x:filterSizeX+x] = f
                temp.shape = (1,self.axisLength**2)
                self.convMatrix[step+filter*allSteps,0:self.axisLength**2] = temp

        self.featureMaps = self.reLu(np.add(self.featureMaps,np.dot(self.convMatrix,self.input)))
        self.recon = self.reLu(np.dot(self.convMatrix.T,self.featureMaps.T))
