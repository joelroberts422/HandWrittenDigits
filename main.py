from mnist import MNIST
from network import Network
from trainer import Trainer

def test(net, im, lab):
    correctcnt = 0
    guesses = [0,0,0,0,0,0,0,0,0,0]
    correctguesses = [0,0,0,0,0,0,0,0,0,0]
    labelcnt = [0,0,0,0,0,0,0,0,0,0]
    for i in range(len(im)):
        output = net.identify(im[i])
        if (i % 1000 == 0):
            print(output, lab[i])
        guesses[output] = guesses[output] + 1
        labelcnt[lab[i]] = labelcnt[lab[i]] + 1
        if output == lab[i]:
            correctguesses[output] = correctguesses[output] + 1
            correctcnt = correctcnt + 1
    stats = []
    for i  in range(len(correctguesses)):
        stats.append(correctguesses[i]/labelcnt[i])
    print(guesses, correctguesses, stats)
    return correctcnt / len(im)


def loadtraining(data):
    images, labels = data.load_training()
    process(images)
    return [images, labels]

def loadtesting(data):
    testimages, testlabels = data.load_testing()
    process(testimages)
    return [testimages, testlabels]

def process(images):
    for image in images:
        for i in range(len(image)):
            image[i] = image[i]/256

def train(trainer, network, batchsize, epochs, targetcost, training, testing):
    n = 1
    while trainer.cost[len(ash.cost) - 1] > targetcost:
        trainer.sgd(batchsize, epochs, training)
        st = 'training round ' + str(n) + ' complete'
        print(st)
        n = n + 1
        p = test(network, testing[0], testing[1])
        st = 'proportion identified: ' + str(p)
        print(st)

mndata = MNIST('Dataset')
training = loadtraining(mndata)
testing = loadtesting(mndata)
print("data ready")

bses = 'biases.npy'
wts = 'weights.npy'
neural = Network(False, bses, wts)
neural.save()
ash = Trainer(neural)
print('here')

#print(test(neural, testing[0], testing[1]))
train(ash, neural, 600, 5, .5, training, testing)


# plt.plot(cost)
# plt.ylabel("cost")
# plt.xlabel("turns")
# plt.show()
#
# print(test(neural, testing[0], testing[1]))
