import matplotlib.pyplot as plt
import math
plt.rcParams['toolbar'] = 'None'
import numpy as np

#ONLY FOR A SINGLE LAYER
class Viz:

    elementsY = 4
    rowInput = 0
    rowFilter = 1
    rowFMs = 2
    rowRecon = 3
    rot = 90
    fontSize = 8

    def __init__(self, x,y, fAmount, fmSize, fSizeAxis, fStepsAxis, layerNr):
        self.fig = plt.figure(figsize=(y, x))
        self.fig.canvas.set_window_title('Convolutional Autoencoder | TRAINING '+str(layerNr)+' LAYER')
        self.fAmount = fAmount
        self.fmSize = fmSize
        self.fSizeAxis = fSizeAxis
        self.fStepsAxis = fStepsAxis

    def setInputs(self, input, filter, fms, recon, channels):
        self.channels = channels
        self.inputSize = int(len(input)/channels)
        self.imSizeAxis = int(math.sqrt(self.inputSize))
        self.input = input
        self.filter = filter
        self.fms = fms
        self.recon = recon

    #only for first Layer
    def showFilter(self):
        for i in range(self.fAmount):
            self.fig.add_subplot(Viz.elementsY,self.fAmount,Viz.rowFilter*self.fAmount+i +1)
            data = self.filter[i]
            if self.channels > 1:
                data = data.reshape(self.fSizeAxis, self.fSizeAxis, self.channels)
                data = np.add(data[:,:,0:1],data[:,:,1:2],data[:,:,2:3]).reshape(self.fSizeAxis,self.fSizeAxis)
            else:
                data = data.reshape(self.fSizeAxis, self.fSizeAxis)

            plt.imshow(data, cmap='gray',interpolation='None')
            plt.xticks([])
            plt.yticks([])
            if i < 1:
                plt.ylabel('Filter',rotation = Viz.rot,fontsize=Viz.fontSize)

    def showFMs(self):
        for i in range(self.fAmount):
            self.fig.add_subplot(Viz.elementsY, self.fAmount, Viz.rowFMs * self.fAmount + i +1)
            data = self.fms.T
            data = data[i * self.fmSize:(i + 1) * self.fmSize]
            data = data.reshape(self.fStepsAxis, self.fStepsAxis)
            plt.imshow(data, interpolation='None')
            plt.xticks([])
            plt.yticks([])
            if i < 1:
                l = plt.ylabel('Feature Maps',rotation = Viz.rot,fontsize=Viz.fontSize)

    def showInput(self, input, row):
        for i in range(self.channels):
            self.fig.add_subplot(Viz.elementsY, self.fAmount, row * self.fAmount + i +1)
            data = input[i* self.inputSize:(i+1)*self.inputSize]
            data = data.reshape(self.imSizeAxis, self.imSizeAxis)
            plt.imshow(data, interpolation='None')
            plt.xticks([])
            plt.yticks([])
            if i < 1 and row == Viz.rowInput:
                plt.ylabel('Input',rotation = Viz.rot,fontsize=Viz.fontSize)
            elif i < 1 and row == Viz.rowRecon:
                plt.ylabel('Reconstruction',rotation = Viz.rot,fontsize=Viz.fontSize)

    def showFeatures(self, reconstr, row):
        for i in range(self.channels):
            self.fig.add_subplot(Viz.elementsY, self.fAmount, row * self.fAmount + i + 1)
            data = reconstr[i * self.inputSize:(i + 1) * self.inputSize]
            data = data.reshape(self.imSizeAxis, self.imSizeAxis)
            plt.imshow(data, interpolation='None')
            plt.xticks([])
            plt.yticks([])
            if i < 1 :
                plt.ylabel('Features', rotation=Viz.rot, fontsize=Viz.fontSize)

    def visualizeNet(self,stop):
        self.fig.clear()
        self.showInput(self.input,Viz.rowInput)
        self.showFilter()
        self.showFMs()
        self.showInput(self.recon,Viz.rowRecon)
        plt.draw()
        plt.pause(0.0001)
        if stop:
            plt.show()

    def visualizeFeatures(self,reconstr,stop):
        self.fig.clear()
        self.showInput(self.input, Viz.rowInput)
        self.showInput(reconstr, Viz.rowFilter)
        plt.draw()
        plt.pause(0.0001)
        if stop:
            plt.show()

    def endViz(self):
        plt.close()