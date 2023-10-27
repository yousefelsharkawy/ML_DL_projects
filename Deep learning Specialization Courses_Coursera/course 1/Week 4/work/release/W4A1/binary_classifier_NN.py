# Description: This file contains the implementation of a binary classifier using a neural network classifier

# import libraries
import numpy as np

# create a binary classifier using a neural network classifier
class BinaryClassifierNN:

    def __init__(self, layer_dims,activations):
        self.layer_dims = layer_dims
        assert (len(layer_dims) - 1) == len(activations), "The number of hidden layers and activations must be the same"
        assert activations[-1] == 'sigmoid', "The output layer must have a sigmoid activation"
        self.activations = activations
        self.num_layers = len(layer_dims)
        self.parameters = self.initialize_parameters()
        self.costs = []

    
    def initialize_parameters(self):
        np.random.seed(3)
        #assert self.layer_dims[-1] == 1, "The output layer must have 1 neuron"
        parameters = {}
        for l in range(1, self.num_layers):
            parameters['W' + str(l)] = np.random.randn(self.layer_dims[l], self.layer_dims[l-1]) * 0.01
            parameters['b' + str(l)] = np.zeros((self.layer_dims[l], 1))
        return parameters
    
    # define the activation functions
    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def relu(self, Z):
        return np.maximum(0, Z)
    
    def tanh(self, Z):
        return np.tanh(Z)
    
    # define the forward propagation
    def forward_propagation(self, X):
        cashes = {'A0': X}
        A = X
        for l in range(1, self.num_layers):
            Z = np.dot(self.parameters['W' + str(l)], A) + self.parameters['b' + str(l)]
            if self.activations[l-1] == 'sigmoid':
                A = self.sigmoid(Z)
            elif self.activations[l-1] == 'relu':
                A = self.relu(Z)
            elif self.activations[l-1] == 'tanh':
                A = self.tanh(Z)
            cashes['Z' + str(l)] = Z
            cashes['A' + str(l)] = A
        return A, cashes
    
    # define the cost function
    def compute_cost(self, A, Y):
        m = Y.shape[1]
        cost = (-1/m) * (np.sum(Y*np.log(A) + (1-Y)*np.log(1-A)))
        return cost
    
    # define the backward activations 
    def sigmoid_backward(self, dA, A):
        g_dash = A * (1 - A)
        return dA * g_dash
    
    def relu_backward(self, dA, Z):
        # we wil multiply dA by 1 if Z > 0 and 0 if Z <= 0, so instead we pass dA if Z > 0 and 0 if Z <= 0
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0 # keep dA and reset the rest -where dZ <= 0-  to 0
        return dZ
    
    def tanh_backward(self, dA, A):
        g_dash = 1 - A**2
        return dA * g_dash
    
    # define the backward propagation
    def backward_propagation(self, A, Y, cashes):
        grads = {}
        m = Y.shape[1]
        dA = - (np.divide(Y, A) - np.divide(1 - Y, 1 - A))
        for l in reversed(range(1, self.num_layers)):
            print(l)
            print(dA.shape)
            if self.activations[l-1] == 'sigmoid':
                dZ = self.sigmoid_backward(dA, cashes['A' + str(l)])
            elif self.activations[l-1] == 'relu':
                dZ = self.relu_backward(dA, cashes['Z' + str(l)])
            elif self.activations[l-1] == 'tanh':
                dZ = self.tanh_backward(dA, cashes['A' + str(l)])
            assert dZ.shape == dA.shape
            dW = (1/m) * np.dot(dZ, cashes['A' + str(l-1)].T)
            db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
            dA = np.dot(self.parameters['W' + str(l)].T, dZ)
            grads['dW' + str(l)] = dW
            grads['db' + str(l)] = db
        return grads
    
    # define the update parameters
    def update_parameters(self, grads, learning_rate):
        for l in range(1, self.num_layers): # num_layers is not included, but since it contains the input layers , things add up
            self.parameters['W' + str(l)] -= learning_rate * grads['dW' + str(l)]
            self.parameters['b' + str(l)] -= learning_rate * grads['db' + str(l)]

    # define the train function
    def train(self, X, Y, learning_rate, num_iterations, print_cost=True):
        for i in range(num_iterations):
            Y_prid, cashes = self.forward_propagation(X)
            cost = self.compute_cost(Y_prid, Y)
            self.costs.append(cost)
            grads = self.backward_propagation(Y_prid, Y, cashes)
            self.update_parameters(grads, learning_rate)
            if print_cost and i % 100 == 0:
                print("Cost after iteration {}: {}".format(i, cost))
        return self.parameters,self.costs
    
    # define the predict function
    def predict(self, X):
        A, cashes = self.forward_propagation(X)
        predictions = (A > 0.5)
        return predictions
    
    # define the accuracy function
    def accuracy(self, X, Y):
        predictions = self.predict(X)
        accuracy = np.mean(predictions == Y) # 1 if they are equal, 0 if they are not, so the mean is the accuracy (count of 1s / total count)
        return accuracy
    
