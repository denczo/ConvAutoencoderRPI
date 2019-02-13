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
    def slide(self):
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
            self.contrastiveDivergence(cutout)


    #TODO need to be more efficient for raspberry zero
    def createFeatureMaps(self):
        fSizeAxis = int(math.sqrt(self.filterSize))
        self.convMatrix = np.zeros(((self.fStepsOneAxis ** 2) * self.filterAmount, self.inputSize*self.channels))
        self.featureMaps = np.zeros((self.filterAmount * self.fStepsOneAxis ** 2))
        self.featureMaps.shape = (1, len(self.featureMaps))

        for filter in range(self.filterAmount):
            f = self.filter[filter]
            f.shape = (fSizeAxis, fSizeAxis, self.channels)
            for step in range(self.allFSteps):
                x = step % self.fStepsOneAxis
                y = int(step / self.fStepsOneAxis)

                temp = np.zeros((self.axisLength, self.axisLength, self.channels))
                #part of matrix as big as the filter
                temp[y:fSizeAxis + y, x:fSizeAxis + x, :] = f
                temp.shape = (1, self.axisLength ** 2 * self.channels)
                print(temp.shape, self.convMatrix.shape)
                self.convMatrix[step + filter * self.allFSteps, 0:self.axisLength ** 2 * self.channels] = temp
                print(self.convMatrix.shape)

    def createConvMatrix(self):

        filterSizeX = int(math.sqrt(self.filterSize))
        inputSizeX = self.axisLength
        allSteps = self.fStepsOneAxis ** 2
        self.convMatrix = np.zeros((((self.fStepsOneAxis) ** 2) * self.filterAmount, self.inputSize*self.channels))
        self.featureMaps = np.zeros((self.filterAmount * self.fStepsOneAxis ** 2))
        self.featureMaps.shape = (1, len(self.featureMaps))
        self.recon = np.zeros(self.inputSize)

        for filter in range(self.filterAmount):
            f = self.filter[filter]
            f.shape = (filterSizeX,filterSizeX,self.channels)
            for step in range(allSteps):
                x = step % self.fStepsOneAxis
                y = int(step / self.fStepsOneAxis)
                temp = np.zeros((inputSizeX,inputSizeX,self.channels))
                temp[y:filterSizeX+y,x:filterSizeX+x,:] = f
                temp.shape = (1,self.axisLength**2*self.channels)
                #print(temp.shape,self.convMatrix.shape)
                self.convMatrix[step+filter*allSteps,0:self.axisLength**2*self.channels] = temp

        self.featureMaps = self.reLu(np.add(self.featureMaps,np.dot(self.convMatrix,self.input)))
        self.recon = self.reLu(np.dot(self.convMatrix.T,self.featureMaps.T))
