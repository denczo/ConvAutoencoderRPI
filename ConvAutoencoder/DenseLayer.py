import numpy as np

class DenseLayer:

    def __init__(self,input,output):
        self.input = np.zeros(input)
        self.input.shape = (input,1)
        self.output = np.zeros(output)
        self.output.shape = (output, 1)
        self.targets = np.zeros(output)
        self.targets.shape = (output, 1)
        self.actuals = np.zeros(output)
        self.actuals.shape = (output, 1)
        self.weights = np.random.uniform(low=-0.1, high=0.1, size=(input, output))

    def computeOutput(self):
        self.output = np.dot(self.weights.T,self.input)

    def computeActuals(self):
        self.actual = np.subtract(self.output,self.targets)

    def computeWeights(self):
        self.weights = np.add(self.weights, np.dot(self.input.T,self.actual))

    def solveEquation(self):
        self.computeOutput()
        self.computeActuals()
        self.computeWeights()

    def classify(self):
        self.computeOutput()
        print(self.output)


