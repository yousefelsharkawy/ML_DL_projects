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
    def forward_propagation(self, X, keep_prob = None, parameters = None, batch_norm = None):
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
            # batch normalization
            if batch_norm is not None:
                if batch_norm[l-1] == True:
                    #print("batch normalization is applied to layer ", l)
                    cashes['Z' + str(l)] = Z
                    mue = np.mean(Z, axis=1, keepdims=True)
                    sigma = np.var(Z, axis=1, keepdims=True)
                    # update the moving averages of the batch normalization mue and sigma
                    self.mue['mue' + str(l)] = 0.9 * self.mue['mue' + str(l)] + 0.1 * mue
                    self.sigma['sigma' + str(l)] = 0.9 * self.sigma['sigma' + str(l)] + 0.1 * sigma
                    Z_norm = (Z - mue) / np.sqrt(sigma + 1e-8)
                    cashes['Z_norm' + str(l)] = Z_norm
                    Z_tilde = np.multiply(self.parameters['gamma' + str(l)], Z_norm) + self.parameters['beta' + str(l)]
                    cashes['Z_tilde' + str(l)] = Z_tilde
                    Z = Z_tilde
            else:
                #print("batch normalization is not applied to layer ", l)
                cashes['Z' + str(l)] = Z

            # apply the activation function
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
            cashes['A' + str(l)] = A
        return A, cashes
    
    # define the cost function
    def compute_cost(self, A, Y, M, lambd = 0):
        # handle the case when A is 0 or 1
        A[A == 0] = 1e-10
        A[A == 1] = 1 - 1e-10
        cost = (-1/ M) * (np.sum(Y*np.log(A) + (1-Y)*np.log(1-A)))
        if lambd != 0:
            L2_regularization_cost = 0
            for l in range(1, self.num_layers):
                L2_regularization_cost += np.sum(np.square(self.parameters['W' + str(l)]))
            L2_regularization_cost *= (lambd/(2*M))
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
    def backward_propagation(self, A, Y, cashes, lambd = 0, keep_prob = None, batch_norm = None):
        grads = {}
        m = Y.shape[1]
        dA = - (np.divide(Y, A) - np.divide(1 - Y, 1 - A))
        for l in reversed(range(1, self.num_layers)):
            #print("l = ", l)
            if batch_norm is None or batch_norm[l-1] == False:
                if self.activations[l-1] == 'sigmoid':
                    dZ = self.sigmoid_backward(dA, cashes['A' + str(l)])
                elif self.activations[l-1] == 'relu':
                    dZ = self.relu_backward(dA, cashes['Z' + str(l)])
                elif self.activations[l-1] == 'tanh':
                    dZ = self.tanh_backward(dA, cashes['A' + str(l)])
                assert dZ.shape == dA.shape
            # apply batch normalization derivatives (if batch normalization is applied, this means that A resulted from the activation function is Z_tilde and not Z)
            if batch_norm is not None and batch_norm[l-1] == True:
                if self.activations[l-1] == 'sigmoid':
                    dZ_tilde = self.sigmoid_backward(dA, cashes['A' + str(l)])
                elif self.activations[l-1] == 'relu':
                    dZ_tilde = self.relu_backward(dA, cashes['Z_tilde' + str(l)])
                elif self.activations[l-1] == 'tanh':
                    dZ_tilde = self.tanh_backward(dA, cashes['A' + str(l)])
                assert dZ_tilde.shape == dA.shape
                dZ_norm = np.multiply(self.parameters['gamma' + str(l)], dZ_tilde)
                dgamma = np.sum(np.multiply(dZ_tilde, cashes['Z_norm' + str(l)]), axis=1, keepdims=True)
                dbeta = np.sum(dZ_tilde, axis=1, keepdims=True)
                dZ = (1/m) * (1/np.sqrt(np.var(cashes['Z' + str(l)], axis=1, keepdims=True) + 1e-8)) * (m*dZ_norm - np.sum(dZ_norm, axis=1, keepdims=True) - np.multiply(cashes['Z_norm' + str(l)], np.sum(np.multiply(dZ_norm, cashes['Z_norm' + str(l)]), axis=1, keepdims=True)))
                grads['dgamma' + str(l)] = dgamma
                grads['dbeta' + str(l)] = dbeta
                

                    
                    
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
    def update_parameters(self, grads, learning_rate, adam_counter, optimizer = "gd", beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, batch_norm = None):
        if optimizer == "adam":
            v_corrected = {} # temporary variables to store the corrected values of v and s and use them in the update equations
            s_corrected = {}


        for l in range(1, self.num_layers): # num_layers is not included, but since it contains the input layers , things add up
            if optimizer == "gd":
                self.parameters['W' + str(l)] -= learning_rate * grads['dW' + str(l)]
                self.parameters['b' + str(l)] -= learning_rate * grads['db' + str(l)]
                if batch_norm is not None and batch_norm[l-1] == True:
                    # update the batch normalization parameters gamma and beta that determine the mean and variance of the normalized Z
                    self.parameters['gamma' + str(l)] -= learning_rate * grads['dgamma' + str(l)]
                    self.parameters['beta' + str(l)] -= learning_rate * grads['dbeta' + str(l)]

            elif optimizer == "momentum":
                self.v["dW" + str(l)] = beta1 * self.v["dW" + str(l)] + (1 - beta1) * grads['dW' + str(l)]
                self.v["db" + str(l)] = beta1 * self.v["db" + str(l)] + (1 - beta1) * grads['db' + str(l)]
                self.parameters['W' + str(l)] = self.parameters['W' + str(l)] - learning_rate * self.v["dW" + str(l)]
                self.parameters['b' + str(l)] = self.parameters['b' + str(l)] - learning_rate * self.v["db" + str(l)]

                if batch_norm is not None and batch_norm[l-1] == True:
                    # update the batch normalization parameters gamma and beta that determine the mean and variance of the normalized Z
                    self.parameters['gamma' + str(l)] -= learning_rate * grads['dgamma' + str(l)]
                    self.parameters['beta' + str(l)] -= learning_rate * grads['dbeta' + str(l)]

            elif optimizer == "rmsprop":
                self.s["dW" + str(l)] = beta1 * self.s["dW" + str(l)] + (1 - beta1) * np.square(grads['dW' + str(l)])
                self.s["db" + str(l)] = beta1 * self.s["db" + str(l)] + (1 - beta1) * np.square(grads['db' + str(l)])
                self.parameters['W' + str(l)] = self.parameters['W' + str(l)] - learning_rate * (grads['dW' + str(l)] / (np.sqrt(self.s["dW" + str(l)]) + epsilon))
                self.parameters['b' + str(l)] = self.parameters['b' + str(l)] - learning_rate * (grads['db' + str(l)] / (np.sqrt(self.s["db" + str(l)]) + epsilon))

                if batch_norm is not None and batch_norm[l-1] == True:
                    # update the batch normalization parameters gamma and beta that determine the mean and variance of the normalized Z
                    self.parameters['gamma' + str(l)] -= learning_rate * grads['dgamma' + str(l)]
                    self.parameters['beta' + str(l)] -= learning_rate * grads['dbeta' + str(l)]
            
            elif optimizer == "adam":
                self.v["dW" + str(l)] = beta1 * self.v["dW" + str(l)] + (1 - beta1) * grads['dW' + str(l)]
                self.v["db" + str(l)] = beta1 * self.v["db" + str(l)] + (1 - beta1) * grads['db' + str(l)]
                self.s["dW" + str(l)] = beta2 * self.s["dW" + str(l)] + (1 - beta2) * np.square(grads['dW' + str(l)])
                self.s["db" + str(l)] = beta2 * self.s["db" + str(l)] + (1 - beta2) * np.square(grads['db' + str(l)])
                # correct the values of v and s
                v_corrected["dW" + str(l)] = self.v["dW" + str(l)] / (1 - np.power(beta1, adam_counter))
                v_corrected["db" + str(l)] = self.v["db" + str(l)] / (1 - np.power(beta1, adam_counter))
                s_corrected["dW" + str(l)] = self.s["dW" + str(l)] / (1 - np.power(beta2, adam_counter))
                s_corrected["db" + str(l)] = self.s["db" + str(l)] / (1 - np.power(beta2, adam_counter))
                # Update the parameters
                self.parameters['W' + str(l)] = self.parameters['W' + str(l)] - learning_rate * (v_corrected["dW" + str(l)] / (np.sqrt(s_corrected["dW" + str(l)]) + epsilon))
                self.parameters['b' + str(l)] = self.parameters['b' + str(l)] - learning_rate * (v_corrected["db" + str(l)] / (np.sqrt(s_corrected["db" + str(l)]) + epsilon))

                if batch_norm is not None and batch_norm[l-1] == True:
                    # update the batch normalization parameters gamma and beta that determine the mean and variance of the normalized Z
                    self.parameters['gamma' + str(l)] -= learning_rate * grads['dgamma' + str(l)]
                    self.parameters['beta' + str(l)] -= learning_rate * grads['dbeta' + str(l)]

            
    # define the train function
    def train(self, X, Y , learning_rate, num_epochs, batch_size = 64, lambd = 0, keep_prob = None, print_cost=True, optimizer = "gd", beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, batch_norm = None):
        # preparing some important variables
        seed = 10 # The seed for randomly generating the mini batches each epoch
        M = X.shape[1] # number of training examples in the whole dataset, we use it to divide the accumulated cost by it to get the average cost of all examples
        adam_counter = 0 # this variable is used to count the number of iterations of the adam optimizer, it is used to correct the values of v and s in the adam optimizer
        if keep_prob != None:
            assert len(keep_prob) == self.num_layers - 1, "The number of keep probabilities must be the same as the number of hidden layers + the output layer"
        # Prepare the tracked exponential moving averages if the optimizer is momentum or RMSprop or Adam
        if optimizer == "momentum":
            self.v = self.initialize_averages(self.parameters, optimizer)
        elif optimizer == "rmsprop":
            self.s = self.initialize_averages(self.parameters, optimizer)
        elif optimizer == "adam":
            self.v, self.s = self.initialize_averages(self.parameters, optimizer)
        
        # batch normalization
        if batch_norm is not None:
            self.mue = {}
            self.sigma = {}
            assert len(batch_norm) == self.num_layers - 1, "The number of batch normalization parameters must be the same as the number of hidden layers + the output layer"
            # initialize the batch normalization parameters for the corresponding layers with True
            for l in range(1, self.num_layers):
                if batch_norm[l-1] == True:
                    #print("batch normalization is applied to layer ", l)
                    self.parameters['gamma' + str(l)] = np.ones((self.layer_dims[l], 1))
                    self.parameters['beta' + str(l)] = np.zeros((self.layer_dims[l], 1))

                    # initialize the moving averages of the batch normalization mue and sigma
                    self.mue['mue' + str(l)] = np.zeros((self.layer_dims[l], 1))
                    self.sigma['sigma' + str(l)] = np.zeros((self.layer_dims[l], 1))

        
        
        # Start the training loop
        for i in range(num_epochs):

            # Define and prepare the random mini batches
            seed = seed + 1 # so that we generate different mini batches each epoch
            mini_batches = self.random_mini_batches(X, Y, batch_size=batch_size, seed=seed)
            total_cost = 0 # we will accummulate the cost of all the mini batches in this variable

            for mini_batch in mini_batches:
                X_batch, Y_batch = mini_batch
                Y_prid_batch, cashes = self.forward_propagation(X_batch,keep_prob = keep_prob ,batch_norm = batch_norm)
                cost = self.compute_cost(Y_prid_batch, Y_batch, M, lambd)
                total_cost += cost
                #self.costs.append(cost)
                grads = self.backward_propagation(A = Y_prid_batch, Y = Y_batch, cashes = cashes, lambd = lambd, keep_prob = keep_prob, batch_norm = batch_norm)
                if optimizer == "adam":
                    adam_counter += 1
                self.update_parameters(grads, learning_rate,adam_counter=adam_counter,optimizer=optimizer,beta1=beta1,beta2=beta2,epsilon=epsilon,batch_norm = batch_norm)
            
            self.costs.append(total_cost)
            if print_cost and i % 1000 == 0:
                print("Cost after iteration {}: {}".format(i, total_cost)) 
        return self.parameters,self.costs
    

    def random_mini_batches(self, X, Y, batch_size = 64, seed = 0):
        np.random.seed(seed)
        m = X.shape[1]
        mini_batches = []
        # shuffle the data
        permutation = list(np.random.permutation(m))
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation]
        # partition the data
        num_complete_batches = m // batch_size
        #print("num_complete_batches = ", num_complete_batches)
        for k in range(num_complete_batches):
            mini_batch_X = shuffled_X[:, k*batch_size : (k+1)*batch_size]
            mini_batch_Y = shuffled_Y[:, k*batch_size : (k+1)*batch_size]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        # handle the case when the last batch is not complete
        if m % batch_size != 0:
            #print("from inside the incomplete loop")
            mini_batch_X = shuffled_X[:, num_complete_batches*batch_size:]
            mini_batch_Y = shuffled_Y[:, num_complete_batches*batch_size:]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        return mini_batches
    
    # optimizers helper functions
    def initialize_averages(self, parameters, optimizer):
        L = len(parameters) // 2
        

        if optimizer == "momentum" or optimizer == "rmsprop":
            v = {} # will be called v with momentum and s with RMSprop (that is how we will receive it in the caller function)
            # The variable keeps track of the exponentialy weighted averages of the gradients in case of momentum and the squared gradients in case of RMSprop, but they are initialized in the same way to zeros
            for l in range(1, L + 1):
                v["dW" + str(l)] = np.zeros(parameters["W" + str(l)].shape)
                v["db" + str(l)] = np.zeros(parameters["b" + str(l)].shape)

            return v
        elif optimizer == "adam":
            v = {}
            s = {}
            # in here we will keep track of both the exponentially weighted averages of the gradients and the squared gradients and initialize them to zeros
            for l in range(1, L + 1):
                v["dW" + str(l)] = np.zeros(parameters["W" + str(l)].shape)
                v["db" + str(l)] = np.zeros(parameters["b" + str(l)].shape)
                s["dW" + str(l)] = np.zeros(parameters["W" + str(l)].shape)
                s["db" + str(l)] = np.zeros(parameters["b" + str(l)].shape)

            return v, s

    
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
    model = BinaryClassifierNN(layer_dims=[2, 5, 4, 1],activations=['relu', 'relu', 'sigmoid'])
    # generate small dataset
    np.random.seed(1)
    X = np.random.randn(2, 1000)
    Y = np.random.randint(0, 2, (1, 1000))
    params, costs = model.train(X, Y, learning_rate=0.01, num_epochs=1000, print_cost=True, batch_norm=[True, False, True])