import os
import pickle
from typing import List
from math import ceil
import numpy as np
#import datasets.mnist.loader as mnist
from sklearn.preprocessing import OneHotEncoder

# Store column oriented

def relu(x: np.ndarray):
    return np.where(x >= 0, x, 0)

def relu_derivative(x: np.ndarray):
    return np.where(x >= 0, 1, 0)


def softmax(x: np.ndarray):
    #nom = np.exp(x - np.max(x, axis=1, keepdims=True))
    max_col = np.max(x, axis=0, keepdims=True)
    nom = np.exp(x - max_col)
    #prob = nom / np.sum(nom, axis=1, keepdims=True)
    demom = np.sum(nom, axis=0, keepdims=True)
    prob = nom / demom
    return prob


def cross_entropy(prob: np.ndarray, labels: np.ndarray):
    return np.mean(-np.sum(labels * np.log(prob + 1e-8), axis=0)) #+ (1-labels) * np.log(1 - prob + 1e-8), axis=0))


class NeuralNetwork(object):

    def __init__(self, neurons_wrt_layers: List[int], n_input: int, n_output: int, learning_rate: float=1e-3) -> None:
        self.L = len(neurons_wrt_layers) + 1
        self.neurons_wrt_layers = neurons_wrt_layers
        self.n_input = n_input
        self.n_output = n_output
        self.learning_rate = learning_rate


        # Initialize weights
        self.weights = {}

        n_in = self.n_input
        i = 1
        for n_out in self.neurons_wrt_layers:
            self.weights["W_{}".format(i)] = np.random.normal(0, scale=np.sqrt(2/(n_in + n_out)), size=(n_in, n_out))
            self.weights["b_{}".format(i)] = np.zeros(shape=(n_out, 1), dtype=np.float32)


            n_in = n_out
            i += 1

        n_out = self.n_output
        self.weights["W_{}".format(i)] = np.random.normal(0, scale=np.sqrt(2/(n_in + n_out)), size=(n_in, n_out))
        self.weights["b_{}".format(i)] = np.zeros(shape=(n_out, 1), dtype=np.float32)

        self.outputs = {}

        self.derivates = {}

    
    def forward(self, x: np.ndarray) -> np.ndarray:
        x = x.T
        if len(x.shape) == 1:
            x = np.expand_dims(x, axis=1)
        self.outputs["a_0"] = x
        for i in range(1, self.L+1):
            #print(i)
            self.outputs["z_{}".format(i)] = np.matmul(self.weights["W_{}".format(i)].T, self.outputs["a_{}".format(i-1)]) + self.weights["b_{}".format(i)]
            if i < self.L:
                self.outputs["a_{}".format(i)] = relu(self.outputs["z_{}".format(i)])
            else:
                self.outputs["a_{}".format(i)] = softmax(self.outputs["z_{}".format(i)])

        return self.outputs["a_{}".format(i)]
        

    def backward(self, labels: np.ndarray) -> float:

        labels = labels.T
        if len(labels.shape) == 1:
            labels = np.expand_dims(labels, axis=1)

        loss = cross_entropy(prob=self.outputs["a_{}".format(self.L)], labels=labels)

        for i in range(self.L, 0, -1):
            if i == self.L:
                self.derivates["e_{}".format(i)] = self.outputs["a_{}".format(i)] - labels
            else:
                self.derivates["e_{}".format(i)] = np.matmul(self.weights["W_{}".format(i+1)], self.derivates["e_{}".format(i+1)]) * relu_derivative(self.outputs["a_{}".format(i)])

            self.derivates["W_{}".format(i)] = np.matmul(self.outputs["a_{}".format(i-1)],  self.derivates["e_{}".format(i)].T)
            self.derivates["b_{}".format(i)] = np.sum(self.derivates["e_{}".format(i)], axis=1, keepdims=True)

        return loss
    
    
    def update(self):

        for i in range(self.L, 0, -1):
            #print(i)
            #print(self.weights["W_{}".format(i)].shape, self.derivates["W_{}".format(i)].shape)
            self.weights["W_{}".format(i)] -= self.learning_rate * self.derivates["W_{}".format(i)]
            self.weights["b_{}".format(i)] -= self.learning_rate * self.derivates["b_{}".format(i)]


    def save_weights(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.weights, f)
            f.close()

    #def train_step(self, x, y):

def pre_process_data(train_x, train_y, test_x, test_y):
    # Normalize
    train_x = train_x / 255.
    test_x = test_x / 255.
 
    enc = OneHotEncoder(sparse=False, categories='auto')
    train_y = enc.fit_transform(train_y.reshape(len(train_y), -1))
 
    test_y = enc.transform(test_y.reshape(len(test_y), -1))
 
    return train_x, train_y, test_x, test_y


if __name__ == "__main__":
    n_input = 3
    n_output = 3
    num_examples = 500
    batch_size = 10
    network = NeuralNetwork(neurons_wrt_layers=[2, 3, 2], n_input=n_input, n_output=n_output, learning_rate=1e-1)
    x = np.random.rand(num_examples, n_output)
    y = np.random.randint(0, n_output, size=(num_examples))
    labels = np.zeros(shape=(y.shape[0], n_output))
    labels[np.arange(y.shape[0]), y] = 1
    y = labels
    #print(y)

    for _ in range(10000):
        for i in range(ceil(y.shape[0]/batch_size)):
            x_in = x[i:i+batch_size]
            y_in = y[i:i+batch_size]
            out = network.forward(x_in)

            #print("Out: {}".format(out))

            loss = network.backward(labels=y_in)
            print(loss)
            network.update()
            #print("Derivatives: {}".format(network.derivates))

    '''train_x, train_y, test_x, test_y = mnist.get_data()
 
    train_x, train_y, test_x, test_y = pre_process_data(train_x, train_y, test_x, test_y)'''

    #csv_path = 

    
