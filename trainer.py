import numpy as np
import random

class Trainer:

    nw = None
    cost = []

    def __init__(self, network):
        self.nw = network
        self.loadcost()

    def savecost(self):
        with open('cost.txt', 'a') as f:
            for c in self.cost:
                f.write((str(c)))
                f.write('\n')
        f.close()

    def loadcost(self):
        with open('cost.txt', 'r') as f:
            self.cost = []
            for l in f:
                self.cost.append(float(l))
        f.close()

    # single input derivative of z
    def derivativesigmoid(self, s):
        return s * (1-s)

    # (Ai -Yi) vector
    def __costvector(self, i, l):
        v = self.nw.feedforward(i)
        t = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        t[l] = 1
        d = np.subtract(v, t)
        return d

    # squares and sums costvector
    def __cost(self, i, l):
        v = self.__costvector(i, l)
        v = np.multiply(v, v)
        c = np.sum(v)
        return c

    # average cost for all given images
    def avgcost(self, images, labels):
        sumcost = 0
        for i in range(len(images)):
            sumcost = sumcost + self.__cost(images[i], labels[i])
        return sumcost / len(images)

    def backprop(self, image, label):
        self.nw.feedforward(image)
        layers = self.nw.layers
        biases = self.nw.biases
        weights = self.nw.weights

        bp = []
        wp = []
        ap = None
        cnt = len(biases) - 1
        while cnt >= 0:
            if cnt == len(biases) - 1:
                bp.append(self.__initialBprime(image, label, layers[len(layers) - 1]))
            else:
                bp.insert(0, self.__bprime(ap, layers[cnt]))
            wp.insert(0, np.outer(bp[0], layers[cnt]))
            if cnt > 0:
                ap = self.__aprime(weights[cnt], bp[0])
            cnt = cnt - 1
        return [bp, wp]

    def gradient(self, images, labels):
        dbiases = []
        dweights = []
        cnt = 0
        for image, label in zip(images, labels):
            grad = self.backprop(image, label)
            if len(dbiases) == 0:
                dbiases = grad[0]
                dweights = grad[1]
            else:
                if np.isnan(np.sum(grad[0][0])) or np.isnan(np.sum(grad[1][0])):
                    print(grad[0])
                dbiases = self.__runningavg(dbiases, grad[0], cnt)
                dweights = self.__runningavg(dweights, grad[1], cnt)
                if np.isnan(np.sum(dbiases[0])) or np.isnan(np.sum(dweights[0][0])):
                    print(dbiases, dweights)
            cnt = cnt + 1
        return [dbiases, dweights]

    def __initialBprime(self, image, label, zlayer):
        cv = self.__costvector(image, label)
        ret = []
        for i in range(len(zlayer)):
            ret.append(2 * self.derivativesigmoid(zlayer[i]) * cv[i])
        ret = np.array(ret)
        return ret

    def __bprime(self, aprime, layer):
        ret = []
        for i in range(len(aprime)):
            ret.append(aprime[i] * self.derivativesigmoid(layer[i]))
        ret = np.array(ret)
        return ret

    def __aprime(self, weights, bp):
        ret = []
        for i in range(len(weights[0])):
            ret.append(np.sum(np.multiply(weights[:, i], bp)))
        ret = np.array(ret)
        return ret

    def __runningavg(self, current, next, cnt):
        ret = []
        for i in range(len(current)):
            x = np.divide(np.add(np.multiply(cnt, current[i]), next[i]), cnt+1)
            ret.append(x)
        return np.array(ret)

    # data is list: [images, labels]
    def sgd(self, batchsize, epochs, data):
        for i in range(epochs):
            batchdata = self.getbatches(data, batchsize)
            for j in range(len(batchdata[0])):
                grad = self.gradient(batchdata[0][j], batchdata[1][j])
                self.nw.shiftgradient(grad)
            st = 'finished epoch ' + str(i+1) + ' of ' + str(epochs)
            print(st)
        c = self.avgcost(data[0], data[1])
        self.cost.append(c)
        self.nw.save()
        self.savecost()
        st = 'average cost: ' + str(c)
        print(st)

    def getbatches(self, data, batchsize):
        images = []
        labels = []
        shuf = list(range(len(data[0])))
        random.shuffle(shuf)
        iterate = iter(shuf)
        batchimages = []
        batchlabels = []
        for n in iterate:
            batchimages.append(data[0][n])
            batchlabels.append(data[1][n])
            if len(batchimages) == batchsize:
                images.append(batchimages)
                labels.append(batchlabels)
                batchimages = []
                batchlabels = []
        return [images, labels]
