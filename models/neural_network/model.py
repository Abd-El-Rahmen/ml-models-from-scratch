import numpy as np 

# Defining the activation functions

# ReLU activation function
def relu(x):
    return np.maximum(0,x)

# Derivative of ReLU
def d_relu(x):
    return (x > 0).astype(float)

# Sigmoid activation function
def sigmoid(x):
    return 1/(1+ np.exp(x))

# Derviative of Sigmoid
def d_sigmoid(x):
    return x/(1 - x)

# Neural Networks
class NeuralNetwork:
    # Initialize the NN
    def __init__(self, input_size , hidden_size , output_size):

        # Dfine the size of the layers
        self.layer_sizes= [input_size] + hidden_size + output_size

        self.weights = []
        self.biases = []

        for i in range(len(self.layer_sizes) -1 ):
            weight = np.random.rand(self.layer_sizes[i] , self.layer_sizes[i+1])
            bias = np.zeros(1,self.layer_sizes[i])
            self.weights.append(weight)
            self.biases.append(bias)
   
    # Forward propagation
    def forward(self,X):
        # Forward propagation through all layers
        self.activations = [X]
        self.z_values = []

        for i in range(len(self.layer_sizes) -1):
            z_value = np.dot(self.activations[-1],self.weights[i]) + self.biases[i]
            self.z_values.append(z_value)

            if i < len(self.layer_sizes) -2 : # we are in a hidden layer
                activation = relu(z_value) 
            else: # we are in the output layer
                activation = sigmoid(z_value)
            self.activations.append(activation)
        return self.activations[-1]

    # Backward propagation
    def backward(self, X, y, learning_rate):
        m = X.shape[0]
        error = self.activations[-1] - y

        for i in range(len(self.layer_sizes) -1 ,-1 , -1): # to start from the last to the first layer
            if i == len(self.layer_sizes) - 2 :
                dZ = error * d_sigmoid(self.activations[i+1]) 
            else:
                dZ = error * d_relu(self.activations[i+1])
            
            # Calculate gradients
            dW = np.dot(self.activations[i].T,dZ) / m
            dB = np.sum(dZ,axis=0,keepdims=True) / m

            # Update weights and biases using gradient descent
            self.weights[i] -= learning_rate * dW
            self.biases[i] -= learning_rate * dB

            #Propagate error backward
            error = np.dot(dZ , self.weights[i].T)

    # Train the network
    def train(self , X , y , epochs = 100, learning_rate = 0.01):
        for epoch in range(epochs):
            # Forward pass
            self.forward(X)
            # Backward pass
            self.backward(X, y , learning_rate)

            # print loss every 10 epochs
            if epoch % 100 == 0:
                loss = np.mean(np.square(y - self.activations[-1]))
                print(f"Epoch {epoch}/{epochs} - Loss: {loss:.6f}")


    
            


            
        













