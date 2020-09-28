import numpy as np
import math


class Network:

    # weights and biases are made in __init, layers is created in feedforward
    weights = []
    biases = []
    layers = []

    # two hidden layers of 16 nodes each
    # one output layer of 10 nodes (0 - 9)
    # initialize near zero, half pos, half neg
    def __initialize(self):
        b1 = np.random.random(16)
        b2 = np.random.random(16)
        b3 = np.random.random(10)
        self.biases = np.array([b1, b2, b3])
        for b in self.biases:
            self.__fillmatrix(b)
        self.__savebiases()

        w1 = np.random.random((16, 784))
        w2 = np.random.random((16, 16))
        w3 = np.random.random((10, 16))
        self.weights = np.array([w1, w2, w3])
        for w in self.weights:
            self.__fillmatrix(w)
        self.__saveweights()

    def __init__(self, brandnew, biasfile, weightfile):
        if brandnew:
            self.__initialize()
        else:
            self.weights = np.load(weightfile, allow_pickle=True)
            self.biases = np.load(biasfile, allow_pickle=True)
        if np.isnan(np.sum(self.weights[0])) or np.isnan(np.sum(self.biases[0])):
            raise ValueError

    def shiftgradient(self, grad):
        for i in range(len(grad[0])):
            self.biases[i] = np.subtract(self.biases[i], grad[0][i])
            self.weights[i] = np.subtract(self.weights[i], grad[1][i])

    def __fillmatrix(m):
        for i in range(len(m)):
            if m.ndim == 2:
                for j in range(len(m[0])):
                    m[i][j] = (m[i][j] - .5)
            else:
                m[i] = (m[i] - .5)

    # single input sigmoid
    def __sigmoid(self, v):
        try:
            return 1 / (1 + math.exp(-v))
        except OverflowError:
            return 1

    def feedforward(self, i):
        self.layers = []
        v = np.array(i)
        for b, w in zip(self.biases, self.weights):
            self.layers.append(v.copy())
            v = np.add(np.dot(w, v), b)
            for i in range(len(v)):
                v[i] = self.__sigmoid(v[i])
        self.layers.append(v.copy())
        return v

    def identify(self, image):
        v = self.feedforward(image)
        print(v)
        m = 0
        output = 0
        for i in range(len(v)):
            if v[i] > m:
                m = v[i]
                output = i
        return output

    def save(self):
        self.__savebiases()
        self.__saveweights()

    def __saveweights(self):
        np.save('weights.npy', self.weights)

    def __savebiases(self):
        np.save('biases.npy', self.biases)
