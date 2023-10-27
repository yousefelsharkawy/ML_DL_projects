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
    def forward_propagation(self, X, keep_prob = None, parameters = None):
        if parameters is None:
            parameters = self.parameters
        cashes = {'A0': X} # this is not efficient with larger datasets
        A = X # first activation is the input layer 
        # dropout on the input layer
        if keep_prob is not None:
            D = np.random.rand(A.shape[0], A.shape[1])
            D = (D < keep_prob[0]).astype(int) # convert the matrix to 0s and 1s
            A = np.multiply(A, D)
            A /= keep_prob[0]
            cashes['D0'] = D

        
        for l in range(1, self.num_layers):
            Z = np.dot(parameters['W' + str(l)], A) + parameters['b' + str(l)]
            if self.activations[l-1] == 'sigmoid':
                A = self.sigmoid(Z)
            elif self.activations[l-1] == 'relu':
                A = self.relu(Z)
            elif self.activations[l-1] == 'tanh':
                A = self.tanh(Z)
            # apply dropout
            if keep_prob != None:
                print("from inside dropout")
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
    def compute_cost(self, A, Y, lambd = 0):
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
    def backward_propagation(self, A, Y, cashes, lambd = 0, keep_prob = None):
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
            grads['db' + str(l)] = db
            grads['dW' + str(l)] = dW
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

    def gradient_check(self, X, Y, epsilon = 1e-7):
        # take the current parameters and reshape them into a vector
        #print("parameters keys:" , self.parameters.keys())
        parameters_values = self.dictionary_to_vector(self.parameters)
        
        ## get the gradients
        # apply forward propagation on the current parameters
        A, cashes = self.forward_propagation(X)
        # compute the gradients using backward propagation
        grads = self.backward_propagation(A, Y, cashes)
        # reshape the gradients into a vector
        # reverse the order of the grads keys to match the order of the parameters keys
        grads = {key:grads[key] for key in reversed(grads.keys())}
        #print("grads keys:" , grads.keys())
        grads_values = self.dictionary_to_vector(grads)

        num_parameters = parameters_values.shape[0]
        J_plus = np.zeros((num_parameters, 1))
        J_minus = np.zeros((num_parameters, 1))
        grad_approx = np.zeros((num_parameters, 1))

        for i in range(num_parameters):
            # compute J_plus[i]
            thetaplus = np.copy(parameters_values) # copy the parameters values to avoid changing the original values
            thetaplus[i][0] += epsilon # nudge only the intended parameter of derivative and leave the rest as they are 
            # calculate the cost after nudging the parameter to the right and fixing the rest of the parameters
            A, cashes = self.forward_propagation(X, parameters = self.vector_to_dictionary(thetaplus))
            J_plus[i] = self.compute_cost(A, Y)
            
            # compute J_minus[i]
            thetaminus = np.copy(parameters_values) # copy the parameters values to avoid changing the original values
            thetaminus[i][0] -= epsilon # nudge only the intended parameter of derivative and leave the rest as they are
            # calculate the cost after nudging the parameter to the left and fixing the rest of the parameters
            A, cashes = self.forward_propagation(X, parameters = self.vector_to_dictionary(thetaminus))
            J_minus[i] = self.compute_cost(A, Y)
            
            # compute grad_approx[i]
            grad_approx[i] = (J_plus[i] - J_minus[i])/ ( 2 * epsilon)

        numerator = np.linalg.norm(grad_approx - grads_values)
        denominator = np.linalg.norm(grads_values) + np.linalg.norm(grad_approx)
        difference = numerator/denominator

        if difference > 2e-7:
            print("\033[91m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
        else:
            print("\033[92m" + "The backward propagation works fine! difference = " + str(difference) + "\033[0m")
        
    
    def dictionary_to_vector(self, parameters):
        count = 0
        for key in parameters.keys():
            #print("key = ", key)
            new_vector = np.reshape(parameters[key], (-1,1))
            if count == 0:
                theta = new_vector
            else:
                theta = np.concatenate((theta, new_vector), axis=0)
            count += 1
        return theta
    
    def vector_to_dictionary(self, theta):
        parameters = {}
        L = len(self.layer_dims)
        start = 0
        for l in range(1, L):
            cuurrent_W_shape = self.layer_dims[l]*self.layer_dims[l-1]
            current_b_shape = self.layer_dims[l]
            parameters['W' + str(l)] = theta[start:start + cuurrent_W_shape].reshape((self.layer_dims[l], self.layer_dims[l-1]))
            parameters['b' + str(l)] = theta[start + cuurrent_W_shape: start + cuurrent_W_shape +current_b_shape].reshape((self.layer_dims[l], 1))
            start += cuurrent_W_shape + current_b_shape
        return parameters        


if __name__ == "__main__":
    clf  = BinaryClassifierNN([2, 4, 1], ['relu', 'sigmoid'])
