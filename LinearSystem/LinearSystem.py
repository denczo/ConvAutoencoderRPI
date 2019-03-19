import numpy as np

class LinearSystem:

    def __init__(self, inputSize, targetSize):
        self.input = np.zeros(inputSize)
        self.inputSize = inputSize
        self.input.shape = (self.inputSize,1)
        self.outputSize = targetSize
        self.weigths = np.zeros((self.inputSize, self.outputSize))
        self.output = np.zeros(self.outputSize)
        self.targets = np.zeros(self.outputSize)
        self.tempMatrix = np.zeros((self.inputSize,self.inputSize,self.outputSize))
        self.tempVector = np.zeros((self.inputSize,self.outputSize))

    def setData(self,input,targets):
        self.input = input
        self.input.shape = (self.inputSize, 1)
        self.targets = targets

    def train(self):
        #alle ableitungen nach wij für ein muster
        for oj in range(self.outputSize):
            temp = np.dot(self.input,self.input.T)
            #print(temp,temp.shape)
            self.tempMatrix[:,:,oj] = np.add(self.tempMatrix[:,:,oj],temp)
            self.tempVector[:,oj] = np.add(self.tempVector[:,oj],np.dot(self.input.T,self.targets[oj]))

    def solveLS(self):
        for oj in range(len(self.output)):
            self.weigths[:,oj] = np.linalg.solve(self.tempMatrix[:,:,oj],self.tempVector[:,oj])

    def run(self,input):
        return np.dot(input,self.weigths)
        #input.shape = (len(input),1)
        #print(self.weigths.shape,input.shape)
        #print(np.dot(input,self.weigths))
