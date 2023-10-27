# Description: This file contains the implementation of a binary classifier using a neural network classifier

# import libraries
import numpy as np

# create a binary classifier using a neural network classifier
class BinaryClassifierNN:

    def __init__(self, layer_dims,activations,initialization_method = "he"):
        self.layer_dims = layer_dims
        assert (len(layer_dims) - 1) == len(activations), "The number of hidden layers and activations must be the same"
        assert activations[-1] == 'sigmoid', "The output layer must have a sigmoid activation"
        self.activations = activations
        self.num_layers = len(layer_dims)
        self.parameters = self.initialize_parameters(initialization_method)
        self.costs = []
        

    
    def initialize_parameters(self, initialization_method):
        np.random.seed(3)
        #assert self.layer_dims[-1] == 1, "The output layer must have 1 neuron"
        parameters = {}
        for l in range(1, self.num_layers): # the index of the last layer is num_layers - 1 that's why num_layers is not included
            #print(l)
            if initialization_method == "random":
                parameters['W' + str(l)] = np.random.randn(self.layer_dims[l], self.layer_dims[l-1]) * 0.01
                parameters['b' + str(l)] = np.zeros((self.layer_dims[l], 1))
            elif initialization_method == "he":
                parameters['W' + str(l)] = np.random.randn(self.layer_dims[l], self.layer_dims[l-1]) * np.sqrt(2/self.layer_dims[l-1])
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
    def forward_propagation(self, X, keep_prob = None):
        cashes = {'A0': X} # this is not efficient with larger datasets
        A = X # first activation is the input layer 
        # dropout on the input layer
        if keep_prob != None:
            D = np.random.rand(A.shape[0], A.shape[1])
            D = (D < keep_prob[0]).astype(int) # convert the matrix to 0s and 1s
            A = np.multiply(A, D)
            A /= keep_prob[0]
            cashes['D0'] = D

        
        for l in range(1, self.num_layers):
            Z = np.dot(self.parameters['W' + str(l)], A) + self.parameters['b' + str(l)]
            if self.activations[l-1] == 'sigmoid':
                A = self.sigmoid(Z)
            elif self.activations[l-1] == 'relu':
                A = self.relu(Z)
            elif self.activations[l-1] == 'tanh':
                A = self.tanh(Z)
            # apply dropout
            if keep_prob != None:
                if l != self.num_layers - 1: # we don't apply dropout on the output layer
                    # print("l = ", l)
                    # print("A shape = ", A.shape)
                    D = np.random.rand(A.shape[0], A.shape[1]) # create a matrix of random numbers with the same shape as A between 0 and 1
                    D = (D < keep_prob[l]).astype(int) # convert the matrix to 0s and 1s
                    A = np.multiply(A, D) # multiply by the activations to shut down those that correspond to 0
                    A /= keep_prob[l] # divide by the keep probability to keep the expected value of the activations the same as before dropping out some of them
                    cashes['D' + str(l)] = D # save the D matrix to use it in the backward propagation
                    # print("D shape = ", cashes['D' + str(l)].shape)
            cashes['Z' + str(l)] = Z
            cashes['A' + str(l)] = A
        return A, cashes
    
    # define the cost function
    def compute_cost(self, A, Y, lambd):
        m = Y.shape[1]
        # handle the case when A is 0 or 1
        A[A == 0] = 1e-10
        A[A == 1] = 1 - 1e-10
        cost = (-1/m) * (np.sum(Y*np.log(A) + (1-Y)*np.log(1-A)))
        if lambd != 0:
            L2_regularization_cost = 0
            for l in range(1, self.num_layers):
                L2_regularization_cost += np.sum(np.square(self.parameters['W' + str(l)]))
            L2_regularization_cost *= (lambd/(2*m))
            cost += L2_regularization_cost
        return cost
    
    # define the backward activations they take dA and return dZ by multiplying dA by the derivative of the activation function element wise
    def sigmoid_backward(self, dA, A):
        g_dash = A * (1 - A)
        return dA * g_dash
    
    def relu_backward(self, dA, Z):
        # we wil multiply dA by 1 if Z > 0 and 0 if Z <= 0, so instead we pass dA if Z > 0 and 0 if Z <= 0
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0 # keep dA and reset the rest -where dZ <= 0-  to 0
        return dZ
    
    def tanh_backward(self, dA, A):
        g_dash = (1 - np.power(A, 2))
        return dA * g_dash
    
    # define the backward propagation
    def backward_propagation(self, A, Y, cashes, lambd, keep_prob):
        grads = {}
        m = Y.shape[1]
        dA = - (np.divide(Y, A) - np.divide(1 - Y, 1 - A))
        for l in reversed(range(1, self.num_layers)):
            #print("l = ", l)
            if self.activations[l-1] == 'sigmoid':
                dZ = self.sigmoid_backward(dA, cashes['A' + str(l)])
            elif self.activations[l-1] == 'relu':
                dZ = self.relu_backward(dA, cashes['Z' + str(l)])
            elif self.activations[l-1] == 'tanh':
                dZ = self.tanh_backward(dA, cashes['A' + str(l)])
            assert dZ.shape == dA.shape
            if lambd != 0:
                dW = (1/m) * np.dot(dZ, cashes['A' + str(l-1)].T) + (lambd/m) * self.parameters['W' + str(l)]
            else:
                dW = (1/m) * np.dot(dZ, cashes['A' + str(l-1)].T)
            db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
            dA = np.dot(self.parameters['W' + str(l)].T, dZ) 
            # apply dropout
            if keep_prob is not None:
                # notice that we subtract 1 because the dA that we work with now is the dA of the layer l-1 (the next layer in the backward propagation)
                # print("l = ", l - 1)
                # print("dA shape = ", dA.shape)
                # print("D shape = ", cashes['D' + str(l - 1)].shape)
                # print("keep_prob = ", keep_prob[l - 1])
                dA = np.multiply(dA, cashes['D' + str(l - 1)]) # we will use the same D that we created in the forward propagation to shut down the upstream of the neurons that were shut down in the forward propagation
                dA /= keep_prob[l - 1] # divide by the keep probability to keep the expected value of the derivatives the same as before dropping out some of them

            grads['dW' + str(l)] = dW
            grads['db' + str(l)] = db
        return grads
    
    # define the update parameters
    def update_parameters(self, grads, learning_rate):
        for l in range(1, self.num_layers): # num_layers is not included, but since it contains the input layers , things add up
            self.parameters['W' + str(l)] -= learning_rate * grads['dW' + str(l)]
            self.parameters['b' + str(l)] -= learning_rate * grads['db' + str(l)]

    # define the train function
    def train(self, X, Y, learning_rate, num_iterations, lambd = 0, keep_prob = None, print_cost=True):
        if keep_prob != None:
            assert len(keep_prob) == self.num_layers - 1, "The number of keep probabilities must be the same as the number of hidden layers + the input layer"
        for i in range(num_iterations):
            Y_prid, cashes = self.forward_propagation(X,keep_prob)
            cost = self.compute_cost(Y_prid, Y, lambd)
            self.costs.append(cost)
            grads = self.backward_propagation(Y_prid, Y, cashes, lambd, keep_prob)
            self.update_parameters(grads, learning_rate)
            if print_cost and i % 1000 == 0:
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
    

if __name__ == "__main__":
    clf  = BinaryClassifierNN([2, 4, 1], ['relu', 'sigmoid'])
