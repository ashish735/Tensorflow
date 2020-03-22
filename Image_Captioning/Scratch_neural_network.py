import numpy as np

def sigmoid(x):   #activation function
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):    #loss derivative function to adjust weights
    return x * (1 -x )

training_inputs = np.array ([[0,0,1],
                            [1,1,1],
                            [1,0,1],
                            [0,1,1]])

training_outputs = np.array ([[0,1,1,0]]).T

np.random.seed(1)   #seed gives same random values

synaptics_weights = 2 * np.random.random((3, 1)) - 1

print('Random Starting Synaptic Weights:')
print(synaptics_weights)

for iteration in range(200000):

    input_layer = training_inputs

    outputs = sigmoid(np.dot(input_layer, synaptics_weights))  #summation of weights and inputs
                                                                #then activation function
    error = training_outputs - outputs

    adjustments = error * sigmoid_derivative(outputs)    #backpropagation

    synaptics_weights += np.dot(input_layer.T, adjustments)   #backpropagation

print('Synaptics weights after training')
print(synaptics_weights)

print('Outputs after training:')
print(outputs)