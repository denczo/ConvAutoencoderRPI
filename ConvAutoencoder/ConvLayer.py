import numpy as np
import math

#basic convolutional layer
class ConvLayer:

    #only for cubic images
    def __init__(self, input, channels, filterSize, filterAmount, stride, learnRate):
        self.filter = np.random.uniform(low=-0.001, high=0.001, size=(filterAmount, filterSize*channels))
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
        self.hidden.shape = (filterAmount,1)
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

    #filter * input cutout
    def forwardActivation(self,filter,cutout):
        self.hidden = self.reLu(np.dot(filter,cutout) + self.biasV)

    #filter * features
    def backwardActivation(self,filter):
        temp = np.add(self.hidden.T,self.biasFMs)
        self.reconstrFilter = self.reLu(np.dot(filter.T,temp.T))

    def contrastiveDivergence(self,cutout):
        temp = np.subtract(cutout,self.reconstrFilter)
        self.filter = np.add(self.filter, self.lernRate*np.dot(self.hidden,temp.T))

    #slide all filter over the entire image with given stride and computes convolution
    def slide(self,trainig):
        inputR = self.input.reshape(self.axisLength,self.axisLength,self.channels)
        reconstrR = self.reconstrInput.reshape(self.axisLength,self.axisLength,self.channels)
        filterSizeX = int(math.sqrt(self.filterSize))
        convStep = 0
        for y in range(0,self.axisLength-filterSizeX+1,self.stride):
            for x in range(0,self.axisLength-filterSizeX+1,self.stride):
                cutout = inputR[y:filterSizeX+y,x:filterSizeX+x,:]
                cutout = cutout.reshape(self.filterSize*self.channels,1)
                self.forwardActivation(self.filter,cutout)
                self.backwardActivation(self.filter)
                if trainig:
                    self.contrastiveDivergence(cutout)
                #print(self.hidden)
                else:
                    self.featureMaps[:,convStep] = self.hidden.T
                    convStep += 1
                #reconstrR[y:filterSizeX+y,x:filterSizeX+x,:] = self.reconstrFilter.reshape(filterSizeX,filterSizeX,self.channels)

        #self.reconstrInput = reconstrR.flatten()

    def trainConvLayer(self, prevLayer, currLayer, iterations, dataBatch):

        for i in range(iterations):
            if len(prevLayer) <= 0:
                prevOut = dataBatch[i]
            else:
                prevLayer[0].updateInput(dataBatch[i])
                prevLayer[0].slide(False)
                for j in range(len(prevLayer) - 1):
                    prevData = prevLayer[j].featureMaps.flatten('F')
                    prevLayer[j + 1].updateInput(prevData)
                    prevLayer[j + 1].slide(False)

                prevOut = prevLayer[-1].featureMaps.flatten('F')

            currLayer.updateInput(prevOut)
            currLayer.slide(True)
