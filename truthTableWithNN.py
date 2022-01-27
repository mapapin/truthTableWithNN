import numpy as np
from matplotlib import pyplot as plt

def eta(i, iter):
    print(f"\r[{'-' * round(i * 100 / iter / 2)}{' ' * round(50 - (i * 100 / iter / 2))}]\
    {round(i * 100 / iter)}%", end='')

HIDEN_NEURONS = 10
EPOCH = 1000
LEARNING_RATE = 0.1

OK = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
RESET = '\033[0m'


class NN():
    def __init__(self, X, y):
        self.input = X
        self.y = y
        self.w1 = np.random.randn(HIDEN_NEURONS, X.shape[0])
        self.w2 = np.random.randn(y.shape[0], HIDEN_NEURONS)
        self.b1 = np.zeros((HIDEN_NEURONS, 1))
        self.b2 = np.zeros((y.shape[0], 1))
        self.m = X.shape[1]
        self.losses = np.zeros((EPOCH, 1))
        self.output = np.zeros(y.shape)

    def forwardProp(self, X):
        z1 = np.dot(self.w1, X) + self.b1
        a1 = NN.sigmoid(z1)
        z2 = np.dot(self.w2, a1) + self.b2
        a2 = NN.sigmoid(z2)

        logprobs = np.multiply(np.log(a2), self.y) + np.multiply(np.log(1 - a2), (1 - self.y))
        cost = -np.sum(logprobs) / self.m
        return (cost, a1, a2)

    def backwardProp(self, a1, a2):
        dz2 = a2 - self.y
        dw2 = np.dot(dz2, a1.T) / self.m
        db2 = np.sum(dz2, axis=1, keepdims=True)
        da1 = np.dot(self.w2.T, dz2)
        dz1 = np.multiply(da1, a1 * (1 - a1))
        dw1 = np.dot(dz1, self.input.T) / self.m
        db1 = np.sum(dz1, axis=1, keepdims=True) / self.m

        self.w1 = self.w1 - LEARNING_RATE * dw1
        self.w2 = self.w2 - LEARNING_RATE * dw2
        self.b1 = self.b1 - LEARNING_RATE * db1
        self.b2 = self.b2 - LEARNING_RATE * db2

    def train(self, epoch, debug=False):
        for i in range(epoch):
            self.losses[i, 0], a1, a2 = self.forwardProp(self.input)
            self.backwardProp(a1, a2)
            eta(i, EPOCH)
        if debug:
            plt.figure()
            plt.plot(self.losses)
            plt.xlabel("EPOCHS")
            plt.ylabel("Loss value")
            plt.show()
        print()

    def test(self, X):
        _, _, a2 = self.forwardProp(X)
        self.output = (a2 >= 0.5) * 1.0
        return self.output[0].astype('int32')

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

def chooseLogicToTrain(keyword='AND'):
    if keyword == 'AND':
        X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
        y = np.array(X[0] & X[1])
        return X, y

    if keyword == 'OR':
        X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
        y = np.array(X[0] | X[1])
        return X, y

    if keyword == 'XOR':
        X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
        y = np.array(X[0] ^ X[1])
        return X, y

    if keyword == 'NOR':
        X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
        y = np.array(((X[0] | X[1]) ^ 1))
        return X, y

def createTestCases(nb):
        tests = []
        for _ in range(nb):
            tests.append(np.random.randint(2, size=(2, 4)))
        return tests

def verifTestCases(nb, tests, nn, keyword='AND'):
    print(f'\n\tTest with keyword : {WARNING}{keyword}{RESET}\n')

    score = 0
    for i in range(nb):
        if keyword == 'AND':
            res = tests[i][0] & tests[i][1]
            if str(res) == str(nn.test(tests[i])):
                print(res, f"{OK}OK{RESET}")
                score += 1
            else:
                print(f"{FAIL}KO{RESET}")

        if keyword == 'OR':
            res = tests[i][0] | tests[i][1]
            if str(res) == str(nn.test(tests[i])):
                print(res, f"{OK}OK{RESET}")
                score += 1
            else:
                print(f"{FAIL}KO{RESET}")

        if keyword == 'XOR':
            res = tests[i][0] ^ tests[i][1]
            if str(res) == str(nn.test(tests[i])):
                print(res, f"{OK}OK{RESET}")
                score += 1
            else:
                print(f"{FAIL}KO{RESET}")

        if keyword == 'NOR':
            res = np.array((tests[i][0] | tests[i][1]) ^ 1)
            if str(res) == str(nn.test(tests[i])):
                print(res, f"{OK}OK{RESET}")
                score += 1
            else:
                print(f"{FAIL}KO{RESET}")
    
    return score * 100 / nb


if __name__ == '__main__':
    NB_TESTS = 20
    LOGICAL_KEYWORD = 'AND'
    X, y = chooseLogicToTrain(LOGICAL_KEYWORD)

    nn = NN(X, y)
    nn.train(EPOCH, debug=False)


    testCases = createTestCases(NB_TESTS)

    score = verifTestCases(NB_TESTS, testCases, nn, LOGICAL_KEYWORD)
    print(f'Score : {score}%')