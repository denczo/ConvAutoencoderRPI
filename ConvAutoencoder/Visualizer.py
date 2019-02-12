import matplotlib.pyplot as plt
import math

class Viz:

    elementsY = 5
    rowInput = 0
    rowFilter = 1
    rowFMs = 2
    rowRecon = 3

    def __init__(self, x,y, fAmount, fmSize, fSizeAxis, fStepsAxis):
        self.fig = plt.figure(figsize=(y, x))
        self.fAmount = fAmount
        self.fmSize = fmSize
        self.fSizeAxis = fSizeAxis
        self.fStepsAxis = fStepsAxis

    def setInputs(self, input, filter, fms, recon, channels):
        self.channels = channels
        self.inputSize = len(input)
        self.imSizeAxis = int(math.sqrt(self.inputSize/channels))
        self.input = input
        self.filter = filter
        self.fms = fms
        self.recon = recon

    def showFilter(self):
        for i in range(self.fAmount):
            plt.axis('off')
            self.fig.add_subplot(Viz.elementsY,self.fAmount,Viz.rowFilter*self.fAmount+i +1)
            data = self.filter[i]
            data = data.reshape(self.fSizeAxis,self.fSizeAxis)
            plt.imshow(data, interpolation='None')

    def showFMs(self):
        for i in range(self.fAmount):
            plt.axis('off')
            self.fig.add_subplot(Viz.elementsY, self.fAmount, Viz.rowFMs * self.fAmount + i +1)
            data = self.fms.T
            data = data[i * self.fmSize:(i + 1) * self.fmSize]
            data = data.reshape(self.fStepsAxis, self.fStepsAxis)
            plt.imshow(data, interpolation='None')

    def showInput(self, input, row):
        for i in range(self.channels):
            plt.axis('off')
            self.fig.add_subplot(Viz.elementsY, self.fAmount, row * self.fAmount + i +1)
            data = input[i* self.inputSize:(i+1)*self.inputSize]
            data = data.reshape(self.imSizeAxis, self.imSizeAxis)
            plt.imshow(data, interpolation='None')

    def visualizeNet(self):
        self.showInput(self.input,Viz.rowInput)
        self.showFilter()
        self.showFMs()
        self.showInput(self.recon,Viz.rowRecon)
        plt.draw()
        plt.pause(0.0001)