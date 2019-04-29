from ConvAutoencoder.ConvLayer import ConvLayer
from ConvAutoencoder.Visualizer import Viz
import math
import numpy as np

#convolutional layer with vizualisation
class ConvLayerViz(ConvLayer):

    def __init__(self, input, channels, filterSize, filterAmount, stride, learnRate):
        super().__init__(input, channels, filterSize, filterAmount, stride, learnRate)
        self.reconstrInput = np.zeros(self.inputSize*channels)
        self.error = []

    # vizualisation setup
    def _vizLayer(self, cl, height, width, name):
        fSizeAxis = int(math.sqrt(cl.filterSize))
        fmSize = cl.fStepsOneAxis ** 2
        viz = Viz(height, width, cl.filterAmount, fmSize, fSizeAxis, cl.fStepsOneAxis, name)
        return viz

    def trainConvLayer(self, prevLayer, currLayer, iterations, dataBatch):

        Viz = self._vizLayer(currLayer, 6, 14, len(prevLayer))
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

            if i % 25000 == 0 and i > 0:
                print(i)
                Viz.setInputs(currLayer.input, currLayer.filter, currLayer.featureMaps.flatten(),
                              currLayer.reconstrInput, currLayer.channels)
                Viz.visualizeNet(False)

        Viz.endViz()

    # slide all filter over the entire image with given stride and computes convolution
    def slide(self, training):
        inputR = self.input.reshape(self.axisLength, self.axisLength, self.channels)
        reconstrR = self.reconstrInput.reshape(self.axisLength, self.axisLength, self.channels)
        filterSizeX = int(math.sqrt(self.filterSize))
        convStep = 0
        for y in range(0, self.axisLength - filterSizeX + 1, self.stride):
            for x in range(0, self.axisLength - filterSizeX + 1, self.stride):
                cutout = inputR[y:filterSizeX + y, x:filterSizeX + x, :]
                cutout = cutout.reshape(self.filterSize * self.channels, 1)
                self._forwardActivation(self.filter, cutout)
                self._backwardActivation(self.filter)
                if training:
                    self._contrastiveDivergence(cutout)
                    # print(self.hidden)
                self.featureMaps[:, convStep] = self.hidden.T
                convStep += 1
                reconstrR[y:filterSizeX+y,x:filterSizeX+x,:] = self.reconstrFilter.reshape(filterSizeX,filterSizeX,self.channels)
        self.reconstrInput = reconstrR.flatten()
        if training:
            self.error.append(np.square(np.subtract(self.input, self.reconstrInput)).mean(axis=None))
