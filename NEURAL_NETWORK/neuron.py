import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt

nnfs.init()


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


# X, y = spiral_data(100, 3)
# dense1 = Layer_Dense(2, 3)
# activation1 = Activation_ReLU()

# dense2 = Layer_Dense(3, 3)
# activation2 = Activation_Softmax()

# dense1.forward(X)
# activation1.forward(dense1.output)

# dense2.forward(activation1.output)
# activation2.forward(dense2.output)

# print(activation2.output[:5])

import math

softmax_output = [0.7, 0.1, 0.2]
target_output = [1, 0, 0]


loss = -(
    math.log(softmax_output[0]) * target_output[0]
    + math.log(softmax_output[1]) * target_output[1]
    + math.log(softmax_output[2]) * target_output[2]
)
print(loss)
