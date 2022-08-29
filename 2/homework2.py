import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
iris = load_iris()
# store features matrix in "X"
X = iris.data
# store response vector in "y"
y = iris.target


def sigmoid(x):
    return 1/(1+np.exp(-x))


class NN:
    def __init__(self) -> None:
        self.input = 4
        self.hidden = 5
        self.output = 3
        self.lr = 0.01
        self.steps = 5000
        self.w1 = np.random.random((self.input, self.hidden))
        self.w2 = np.random.random((self.hidden, self.output))

    def forward(self, input_var):
        hidden_var = sigmoid(input_var.dot(self.w1))
        out_var = sigmoid(hidden_var.dot(self.w2))
        return hidden_var, out_var

    def backward(self, target, out_var, hidden_var, input_var):
        n = target.shape[0]
        loss = np.sum((out_var-target).T.dot(out_var-target))/n
        delta2 = (out_var-target)*out_var*(1-out_var)
        deltaw2 = self.lr*hidden_var.T.dot(delta2)/n
        delta1 = delta2.dot(self.w2.T)*hidden_var*(1-hidden_var)
        deltaw1 = self.lr*input_var.T.dot(delta1)/n
        return deltaw1, deltaw2, loss

    def train(self, input, target):
        Loss = []
        for i in range(self.steps):
            input_var = input
            target_var = target
            hidden_var, output = self.forward(input_var)
            delta_w1, delta_w2, loss = self.backward(
                target_var, output, hidden_var, input_var)
            self.w1 -= delta_w1
            self.w2 -= delta_w2
            Loss.append(loss)
        Loss = np.array(Loss)
        return Loss


if __name__ == '__main__':
    nn = NN()
    y = np.eye(3)[y]
    loss = nn.train(X, y)
    plt.plot(np.linspace(1, 5000, 5000), loss)
    plt.xlabel('steps')
    plt.ylabel('loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.show()
