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
        self.reconstr = np.zeros(filterSize)
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

    def backwardActivation(self):
        temp = np.add(self.hidden.T,self.biasFMs)
        self.reconstr = self.reLu(np.dot(self.filter.T,temp.T))

    def contrastiveDivergence(self,cutout):
        temp = np.subtract(cutout,self.reconstr)
        self.filter = np.add(self.filter, self.lernRate*np.dot(self.hidden,temp.T))

    #slide all filter over the entire image with given stride and computes convolution
    def slide(self,trainig):
        inputR = self.input.reshape(self.axisLength,self.axisLength,self.channels)
        #temp = self.input.reshape(self.axisLength,self.axisLength)
        filterSizeX = int(math.sqrt(self.filterSize))

        for step in range(self.allFSteps-1):
            x = (step+self.stride) % self.fStepsOneAxis
            y = int((step+self.stride) / self.fStepsOneAxis)
            cutout = inputR[y:filterSizeX+y,x:filterSizeX+x,:]
            cutout = cutout.reshape(self.filterSize*self.channels,1)
            self.forwardActivation(cutout)
            self.backwardActivation()
            if trainig:
                self.contrastiveDivergence(cutout)
            self.featureMaps[:,step] = self.hidden.T

    #TODO does not work for multiple channels
    def slideDeconv(self, featureMap, filterT):
        #fmR = featureMap.reshape(self.fStepsOneAxis, self.fStepsOneAxis)
        #filterT = filterT[:,:,:1]
        #filterT.shape = (3,3)
        filterSizeX = int(math.sqrt(self.filterSize))
        fOnInput = np.zeros((self.axisLength,self.axisLength))
        for step in range(self.allFSteps - 1):
            x = (step + self.stride) % self.fStepsOneAxis
            y = int((step + self.stride) / self.fStepsOneAxis)
            temp = self.reLu(np.dot(featureMap[step],filterT))
            temp = temp.reshape(filterSizeX,filterSizeX)
            fOnInput[y:filterSizeX+y,x:filterSizeX+x] = temp
        return fOnInput