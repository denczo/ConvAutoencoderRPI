from ConvAutoencoder.ConvLayer import ConvLayer
from ConvAutoencoder.Visualizer import Viz
import math
import numpy as np

#convolutional layer with vizualisation
class ConvLayerViz(ConvLayer):

    def __init__(self, input, channels, filterSize, filterAmount, stride, learnRate):
        super().__init__(input, channels, filterSize, filterAmount, stride, learnRate)

    def guidedBackwardsActivation(self,featureMaps,obsFilter):
        reconstrR = self.reconstrInput.reshape(self.axisLength, self.axisLength, self.channels)
        filterSizeX = int(math.sqrt(self.filterSize))

        convStep = 0
        for y in range(0, self.axisLength - filterSizeX, self.stride):
            for x in range(0, self.axisLength - filterSizeX, self.stride):
                self.hidden = featureMaps[:,convStep]
                convStep += 1

                # reconstruction of image with individual filter (sets all filter except chosen ones to 0)
                self.observeFilter(obsFilter)
                self.backwardActivation(self.obsFilter)
                reconstrR[y:filterSizeX + y, x:filterSizeX + x, :] = self.reconstrFilter.reshape(filterSizeX, filterSizeX, self.channels)

        self.reconstrInput = reconstrR.flatten()

    # all filter except the chosen ones are set to 0
    def observeFilter(self, observed):
        self.obsFilter = np.zeros((self.filterAmount, self.filterSize * self.channels))
        if observed > self.filterAmount:
            observed = self.filterAmount
        else:
            observed = abs(observed)

        for i in range(observed):
            self.obsFilter[i, :] = self.filter[i, :]

    # vizualisation setup
    def vizLayer(self, cl, height, width, name):
        fSizeAxis = int(math.sqrt(cl.filterSize))
        fmSize = cl.fStepsOneAxis ** 2
        viz = Viz(height, width, cl.filterAmount, fmSize, fSizeAxis, cl.fStepsOneAxis, name)
        return viz

    def trainConvLayer(self, prevLayer, currLayer, iterations, dataBatch):

        Viz = self.vizLayer(currLayer, 6, 14, len(prevLayer))
        for i in range(iterations):
            if len(prevLayer) <= 0:
                prevOut = dataBatch[i]
            else:
                prevLayer[0].updateInput(dataBatch[i])
                prevLayer[0].slide(False)
                prevOut = prevLayer[-1].featureMaps.flatten('F')

            for j in range(len(prevLayer) - 1):
                prevData = prevLayer[j].featureMaps.flatten('F')
                prevLayer[j + 1].updateInput(prevData)
                prevLayer[j + 1].slide(False)

            currLayer.updateInput(prevOut)
            currLayer.slide(True)

            if i % 25 == 0:
                print(i)
                Viz.setInputs(currLayer.input, currLayer.filter, currLayer.featureMaps.flatten(),
                              currLayer.reconstrInput, currLayer.channels)
                Viz.visualizeNet(False)

        Viz.endViz()