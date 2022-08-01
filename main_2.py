import os
from math import ceil
from typing import List
import numpy as np

from sklearn.model_selection import StratifiedKFold
import pandas as pd



def relu(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, x, 0)

def relu_derivative(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1, 0)

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    return sigmoid(x) * (1 - sigmoid(x))


def softmax(x: np.ndarray) -> float:
    max_row = np.max(x, axis=1, keepdims=True)
    nom = np.exp(x - max_row)
    denom = np.sum(nom, axis=1, keepdims=True)
    prob = nom / denom
    return prob

def cross_entropy(probs: np.ndarray, labels: np.ndarray) -> float:
    one_sample_loss = -np.sum(labels * np.log(probs + 1e-8) + (1-labels) * np.log(1 - probs + 1e-8), axis=1)
    return np.mean(one_sample_loss)

class NeuralNetwork(object):
    def __init__(self, neurons_wrt_layers: List[int], n_input: int, n_output: int, learning_rate: float=1e-3) -> None:
        self.L = len(neurons_wrt_layers) + 1
        self.neurons_wrt_layers = neurons_wrt_layers
        self.n_input = n_input
        self.n_output = n_output
        self.learning_rate = learning_rate
        
        self.weights = {}

        n_in = self.n_input
        i = 1
        for n_out in self.neurons_wrt_layers:
            self.weights["W_{}".format(i)] = np.random.normal(0, scale=np.sqrt(2/(n_in + n_out)), size=(n_in, n_out))
            self.weights["b_{}".format(i)] = np.zeros(shape=(1, n_out), dtype=np.float32)

            n_in = n_out
            i += 1

        n_out = self.n_output
        self.weights["W_{}".format(i)] = np.random.normal(0, scale=np.sqrt(2/(n_in + n_out)), size=(n_in, n_out))
        self.weights["b_{}".format(i)] = np.zeros(shape=(1, n_out), dtype=np.float32)

        self.outputs = {}

        self.derivates = {}

    def forward(self, x: np.ndarray):
        if len(x.shape) == 1:
            x = np.expand_dims(x, axis=0)

        self.outputs["a_0"] = x
        for i in range(1, self.L + 1):
            assert self.outputs["a_{}".format(i-1)].shape[1] == self.weights["W_{}".format(i)].shape[0]
            self.outputs["z_{}".format(i)] = np.matmul(self.outputs["a_{}".format(i-1)], self.weights["W_{}".format(i)]) + self.weights["b_{}".format(i)]

            if i < self.L:
                #self.outputs["a_{}".format(i)] = relu(x=self.outputs["z_{}".format(i)])
                self.outputs["a_{}".format(i)] = sigmoid(x=self.outputs["z_{}".format(i)])
            elif i == self.L:
                self.outputs["a_{}".format(i)] = softmax(x=self.outputs["z_{}".format(i)])

        return self.outputs["a_{}".format(self.L)]

    def backward(self, y: np.ndarray):

        if len(y.shape) == 1:
            y = np.expand_dims(y, axis=0)

        self.derivates["e_{}".format(self.L)] = self.outputs["a_{}".format(self.L)] - y
        self.derivates["W_{}".format(self.L)] = np.matmul(self.outputs["a_{}".format(self.L - 1)].T, self.derivates["e_{}".format(self.L)])
        self.derivates["b_{}".format(self.L)] = np.sum(self.derivates["e_{}".format(self.L)], axis=0, keepdims=True)

        for i in range(self.L - 1, 0, -1):
            #self.derivates["e_{}".format(i)] = np.matmul(self.derivates["e_{}".format(i+1)], self.weights["W_{}".format(i+1)].T) * relu_derivative(self.outputs["z_{}".format(i)])
            self.derivates["e_{}".format(i)] = np.matmul(self.derivates["e_{}".format(i+1)], self.weights["W_{}".format(i+1)].T) * sigmoid_derivative(self.outputs["z_{}".format(i)])
            self.derivates["W_{}".format(i)] = np.matmul(self.outputs["a_{}".format(i-1)].T, self.derivates["e_{}".format(i)])
            self.derivates["b_{}".format(i)] = np.sum(self.derivates["e_{}".format(i)], axis=0, keepdims=True)

        loss = cross_entropy(probs=self.outputs["a_{}".format(self.L)], labels=y)

        return loss

    def update(self):

        #print("Update")
        
        for i in range(1, self.L + 1):
            self.weights["W_{}".format(i)] -= self.learning_rate * self.derivates["W_{}".format(i)]
            self.weights["b_{}".format(i)] -= self.learning_rate * self.derivates["b_{}".format(i)]
        #print(self.derivates)


'''x = np.random.normal(0, 1, size=(5, 2))
prob = softmax(x=x)
labels = np.random.randint(0, 2, size=(prob.shape[0], prob.shape[1]))
print(x, relu(x), relu_derivative(x), prob)
print("Prob: {}, Labels: {}, Cross entropy: {}".format(prob, labels, cross_entropy(probs=prob, labels=labels)))
network = NeuralNetwork(neurons_wrt_layers=[2, 3, 3], n_input=3, n_output=3, learning_rate=0.1)

print(network.weights)
print(network.L)

# = np.random.randint(0, 5, size=(3, 2))
#y = np.random.randint(0, 5, size=(2, 2))

#print(x, "\n" , y, "\n", np.matmul(x, y))
#print(x, y, np.matmul(x, y))

x = np.random.rand(2, 3)
y = np.random.randint(0, 3, size=(2))
labels = np.zeros(shape=(y.shape[0], 3))
labels[np.arange(y.shape[0]), y] = 1
y = labels

print("Output: {}".format(network.forward(x=x)))
print("Weights: {}".format(network.weights))
print("Outputs: {}".format(network.outputs))
print("Loss: {}".format(network.backward(y=y)))
print("Derivatives: {}".format(network.derivates))'''


if __name__ == "__main__":

    csv_path = r"C:\Users\Thanh\Downloads\voice_gender\voice.csv"
    batch_size = 64

    df = pd.read_csv(csv_path)
    df['label'] = df['label'].replace({'male':1,'female':0})

    x = df.drop("label", axis=1).to_numpy(dtype=np.float)
    y = df["label"].values
    labels = np.zeros(shape=(y.shape[0], 2))
    labels[np.arange(y.shape[0]), y] = 1
    x = (x - np.min(x, axis=0, keepdims=True))/(np.max(x, axis=0, keepdims=True) - np.min(x, axis=0, keepdims=True))

    skf = StratifiedKFold(n_splits=5)
    for train_index, test_index in skf.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

    network = NeuralNetwork(neurons_wrt_layers=[24, 32, 24], n_input=x.shape[1], n_output=2, learning_rate=0.1)

    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    print(y_train, np.argmax(y_train, axis=1))
    old_weights = network.weights
    for epoch in range(200):
        for i in range(ceil(y_train.shape[0] / batch_size)):
            batch_x = x_train[i * batch_size:i * batch_size + batch_size]
            batch_y = y_train[i * batch_size:i * batch_size + batch_size]

            probs = network.forward(x=batch_x)
            loss = network.backward(y=batch_y)
            network.update()
            label = np.argmax(batch_y, axis=1)
            predict = np.argmax(probs, axis=1)
            acc = (label == predict).sum()/label.shape[0]
            #print(i * batch_size, i * batch_size + batch_size)
            #print(batch_y)
            if i % 10 == 0:
                print("Epoch: {} Step: {} Loss: {}, Accuracy: {}".format(epoch + 1, i+1, loss, acc))

        probs = network.forward(x=x_test)
        label = np.argmax(y_test, axis=1)
        predict = np.argmax(probs, axis=1)
        #print((label == predict).sum()/label.shape[0])
        weights = network.weights
        changes = {}
        for keys in weights:
            changes[keys] = np.sum((weights[keys] - old_weights[keys])**2)

        old_weights = weights

        #print(changes)
        

    