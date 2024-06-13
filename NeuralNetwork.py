import numpy as np
import random

class NeuralNetwork:

    def __init__(self, layerSizes : list[int]) -> None:
        
        self.layerSizes = layerSizes
        self.layerAmount = len(layerSizes)

        # suppose layerSizes = [4,2,4] layerSizes[1:] = [2,4]
        # becomes list comprehension [np.random.randn(b,1) for b in [2,4]]
        self.biases = [np.random.randn(b, 1) for b in layerSizes[1:]]

        # each neuron is attached to following neurons (except output neurons) with a weight
        # each neuron that is not the input neuron n, is attached by a set of weights m, making weights dimension = n x m
        # zip(a,b) function creates an iterator of tuples combining elements iterated through a,b
        # self.weights[0] refers to weights connecting layer 0 to layer 1, so on..
        self.weights = [np.random.randn(y, x) for y, x in zip(layerSizes[1:], layerSizes[:-1])]
        self.batchSize = 50

    # given input a representing the activation value is a numpy array with dimensions (n,1)
    # following the formula, sigmoid(w*a + b) returns the activation value of each neuron in the next layer, and z = (w*x + b) used for the chain rule
    def feedForward(self, a):
        activationValuesList = []
        activationValuesList.append(a)
        zValuesList = []
        i = 0
        for b, w in zip(self.biases, self.weights):
            zValue = np.dot(w, activationValuesList[i]) + b
            zValuesList.append(zValue)      
            activationValuesList.append(self.sigmoid(zValue))
            i=+1
        return activationValuesList, zValuesList

    def train(self, epochs, learningRate ):
        for i in range(epochs):
            self.updateBatch(createBatch(self.batchSize), learningRate)
    def updateBatch(self, batch ,learningRate):
        biasGradientsSum = [np.zeros_like(b) for b in self.biases]
        weightGradientsSum = [np.zeros_like(w) for w in self.weights]

        for inputArray, expectedOutput in batch:
            
            # backpropagation helps us calculate partial derivative of the loss function w.r.t. weight and bias
            weightGradients, biasGradients = self.backProp(inputArray, expectedOutput)

            # summing gradients of batches to calculate its average when updating weights and biases
            biasGradientsSum = [nb+dnb for nb,dnb in zip(biasGradientsSum, biasGradients)]
            weightGradientsSum = [nw+dnw for nw,dnw in zip(weightGradientsSum, weightGradients)]

        #note this part is out of the for loop, accumulated gradients are divided by len(batch)
        self.weights =[ (weight - learningRate/len(batch) * wGradient) 
                        for weight, wGradient in zip(self.weights, weightGradientsSum)]
        self.biases = [ (b - (learningRate/len(batch) * nb)) for b,nb in zip(self.biases, biasGradientsSum)]
        #after this step, one epoch is complete, and we repeat for n epochs

    def backProp(self, inputarray, expectedOutput):
        
        aValueslist, zValuesList = self.feedForward(inputarray)
        
        weightGradients = [np.zeros_like(w) for w in self.weights]
        biasGradients = [np.zeros_like(b) for b in self.biases]
        #numpy allows this line, elsewise typeError would be raised

        # Chain rule begings
        delta = self.costDerivative(aValueslist[-1], expectedOutput) * self.sigmoidDerivative(zValuesList[-1])
        biasGradients[-1] = delta
        weightGradients[-1] = (delta * aValueslist[-2].T)
    
        #starting from 2 because we did the first backprop just above, for loop is for hidden layer implementations
        for i in range(2, self.layerAmount):
            sigmoidDeriv = self.sigmoidDerivative(zValuesList[-i])     
            delta = np.dot(self.weights[-i + 1].T, delta) * sigmoidDeriv            
            biasGradients[-i] = delta
            weightGradients[-i] =(delta * aValueslist[-i-1].T)
        return weightGradients, biasGradients

    # Sigmoid function, where z is a numpy array representation w*a + b
    def sigmoid(self, z):
        return 1.0/(1.0+np.exp(-z))
    
    # DO NOT SWITCH OUTPUT AND EXPECTED EVER! THE COST DERIVATIVE MUST BE OUTPUT - EXPECTED
    def costDerivative(self, output, expected):
        return (output - expected)
    
    def sigmoidDerivative(self, z):
        return self.sigmoid(z)*(1-self.sigmoid(z))
    
    
    # Functions below are only code for the specific task of classifying odd and even integers.
    def predict(self, testBatch):
        for input, number in testBatch:
            print(f"{number},", end= "")
            a,z = self.feedForward(input)
            a = a[-1]
            print("Guessed:")
            print(a)
            if np.argmax(a) == 0:
                print("odd")
            else:
                print("even")
            print()

def createBatch(n):
    batch = []
    for i in range(n):
        testNumber = random.randint(0,9)
        input = [0 for i in range(10)]
        input[testNumber] = 1
        output = getIsEvenArray(testNumber)
        batchTuple = (np.reshape(input, (10,1)), np.reshape(output, (2,1)))
        batch.append(batchTuple)
    return batch

def getIsEvenArray(n):
    return [[0],[1]] if n%2==0 else [[1],[0]]

def createTestBatch(n):
    testBatch = []
    for i in range(n):
        testNumber = random.randint(0,9)
        input = [0 for i in range(10)]
        input[testNumber] = 1
        testBatch.append((np.reshape(input, (10,1)), testNumber))
    return testBatch

nn = NeuralNetwork([10,4,2])

# train(number of epochs, learning rate)
# As you increase the number of epochs, you can observe that the NN will slowly converge to the right prediction.
nn.train(20000, 0.06)

# testing the NN with a new batch
testBatch = createTestBatch(50)
nn.predict(testBatch)