import numpy as np
import math

class ConvLayerLight:

    #only for cubic images
    def __init__(self, input, channels, filterSize, filterAmount, stride, learnRate):
        self.filter = np.random.uniform(low=-0.1, high=0.1, size=(filterAmount, filterSize*channels))
        self.filterAmount = filterAmount
        self.filterSize = filterSize
        self.stride = stride
        self.lernRate = learnRate
        self.input = input
        self.channels = channels
        self.inputSize = int(len(input)/channels)
        self.axisLength = int(math.sqrt(self.inputSize))
        self.reconstrFilter = np.zeros(filterSize)
        self.reconstrInput = np.zeros(self.inputSize*channels)
        self.hidden = np.zeros(filterAmount)
        self.fStepsOneAxis = int(((self.axisLength - math.sqrt(self.filterSize))) / self.stride + 1)
        self.allFSteps = self.fStepsOneAxis**2
        self.reLu = lambda x: x * (x > 0)
        self.biasV = 0
        self.biasFMs = np.zeros(filterAmount)
        self.featureMaps = np.zeros((self.filterAmount, self.fStepsOneAxis ** 2))

    def updateInput(self,input):
        self.input = input

    def setBiasVisible(self, value):
        self.biasV = value

    def setBiasesFMs(self, value):
        for i in range(self.filterAmount):
            self.biasFMs[i] = value[i]

    def forwardActivation(self,cutout):
        self.hidden = self.reLu(np.dot(self.filter,cutout) + self.biasV)

    def backwardActivation(self,filter):
        temp = np.add(self.hidden.T,self.biasFMs)
        self.reconstrFilter = self.reLu(np.dot(filter.T,temp.T))

    def contrastiveDivergence(self,cutout):
        temp = np.subtract(cutout,self.reconstrFilter)
        self.filter = np.add(self.filter, self.lernRate*np.dot(self.hidden,temp.T))

    #slide all filter over the entire image with given stride and computes convolution
    def slide(self,trainig,observe):
        inputR = self.input.reshape(self.axisLength,self.axisLength,self.channels)
        reconstrR = self.reconstrInput.reshape(self.axisLength,self.axisLength,self.channels)
        filterSizeX = int(math.sqrt(self.filterSize))
        x = 0
        for step in range(self.allFSteps-1):

            if x >= (self.axisLength-filterSizeX):
                x = 0

            y = int((step*self.stride) / (self.axisLength-filterSizeX+1))*self.stride
            #print(x,y,step)
            cutout = inputR[y:filterSizeX+y,x:filterSizeX+x,:]
            cutout = cutout.reshape(self.filterSize*self.channels,1)

            self.forwardActivation(cutout)
            self.backwardActivation(self.filter)
            if trainig:
                self.contrastiveDivergence(cutout)

            self.featureMaps[:,step] = self.hidden.T

            if observe:
                #reconstruction of image with individual filter (sets all filter except chosen ones to 0)
                self.observeFilter([1,4])
                self.backwardActivation(self.obsFilter)

            reconstrR[y:filterSizeX+y,x:filterSizeX+x,:] = self.reconstrFilter.reshape(filterSizeX,filterSizeX,self.channels)
            x += self.stride

        self.reconstrInput = reconstrR.flatten()

    def guidedBackwardsActivation(self,featureMaps):
        reconstrR = self.reconstrInput.reshape(self.axisLength, self.axisLength, self.channels)
        filterSizeX = int(math.sqrt(self.filterSize))

        for step in range(self.allFSteps - 1):
            x = (step + self.stride) % self.fStepsOneAxis
            y = int((step + self.stride) / self.fStepsOneAxis)
            self.hidden = featureMaps[:,step]
            self.backwardActivation(self.filter)

            reconstrR[y:filterSizeX + y, x:filterSizeX + x, :] = self.reconstrFilter.reshape(filterSizeX, filterSizeX, self.channels)
            self.reconstrInput = reconstrR.flatten()

    #all filter except the chosen ones are set to 0
    def observeFilter(self,observed):
        self.obsFilter = np.zeros((self.filterAmount, self.filterSize*self.channels))
        for i in range(len(observed)):
            f = observed[i]
            if f < self.filterAmount:
                self.obsFilter[f,:] = self.filter[i,:]
