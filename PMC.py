import random


class myPMC: 
    def __init__(self, numLayers, TabNumNeuronPerLayer):
        self.numLayers = numLayers
        self.TabNumNeuronPerLayer = TabNumNeuronPerLayer

    def initialize_network(self):
        self.network = []
        for i in range(self.numLayers):
            layer = []
            for j in range(self.TabNumNeuronPerLayer[i]):
                if i == 0:
                    neuron = {
                        'a' : 0
                    }
                else:   
                    neuron = {
                        'weights': [random.uniform(-1, 1) for _ in range(self.TabNumNeuronPerLayer[i-1])],
                        'bias': random.uniform(-1, 1),
                        'z': 0,
                        'a': 0
                    }
                layer.append(neuron)
            self.network.append(layer)

    def activation_function(self, z):
        return 1 / (1 + pow(2.71828, -z))

    def display_network(self):
        for i, layer in enumerate(self.network):
            print(f"Layer {i}:")
            for j, neuron in enumerate(layer):
                if i == 0:
                    print(f"  Neuron Input {j}: a: {neuron['a']}")
                else:
                    print(f"  Neuron {j}: Weights: {neuron['weights']}, Bias: {neuron['bias']}, z: {neuron['z']}, a: {neuron['a']}")

    def forward_propagation(self, input_data):
        for i in range(self.numLayers):
            for j in range(self.TabNumNeuronPerLayer[i]):
                cur_neuron = self.network[i][j]
                if i == 0:
                    cur_neuron['a'] = input_data[j]
                elif i < self.numLayers - 1:
                    prev_layer = self.network[i-1]
                    cur_neuron['z'] = sum(w * prev_layer[k]['a'] for k, w in enumerate(cur_neuron['weights'])) + cur_neuron['bias']
                    cur_neuron['a'] = self.activation_function(cur_neuron['z'])
                elif i == self.numLayers - 1:
                    prev_layer = self.network[i-1]
                    cur_neuron['z'] = sum(w * prev_layer[k]['a'] for k, w in enumerate(cur_neuron['weights'])) + cur_neuron['bias']
        
        denominator = sum(pow(2.71828, self.network[-1][k]['z'] - max(neuron['z'] for neuron in self.network[-1])) for k in range(self.TabNumNeuronPerLayer[-1]))
        for i in range(self.TabNumNeuronPerLayer[-1]):
            cur_neuron = self.network[-1][i]
            cur_neuron['a'] = pow(2.71828, cur_neuron['z'] - max(neuron['z'] for neuron in self.network[-1])) / denominator
                    
    def backward_propagation(self, target_output):
        for i in range(self.numLayers - 1, 0, -1):
            for j in range(self.TabNumNeuronPerLayer[i]):
                cur_neuron = self.network[i][j]
                if i == self.numLayers - 1:
                    cur_neuron['delta'] = cur_neuron['a'] - target_output[j]
                else:
                    next_layer = self.network[i + 1]
                    cur_neuron['delta'] = sum(next_layer[k]['weights'][j] * next_layer[k]['delta'] for k in range(self.TabNumNeuronPerLayer[i + 1])) * (cur_neuron['a'] * (1 - cur_neuron['a']))


    def update_weights(self, learning_rate):
        for i in range(1, self.numLayers):
            for j in range(self.TabNumNeuronPerLayer[i]):
                cur_neuron = self.network[i][j]
                prev_layer = self.network[i - 1]
                for k in range(self.TabNumNeuronPerLayer[i - 1]):
                    cur_neuron['weights'][k] -= learning_rate * cur_neuron['delta'] * prev_layer[k]['a']
                cur_neuron['bias'] -= learning_rate * cur_neuron['delta']

    def train(self, input_data, target_output, learning_rate, epochs):
        for epoch in range(epochs):
            for x, y in zip(input_data, target_output):
                self.forward_propagation(x)
                self.backward_propagation(y)
                self.update_weights(learning_rate)
                if epoch % 100 == 0:
                    print(f"Epoch {epoch + 1}/{epochs}, Input: {x}, Target: {y}, Output: {[neuron['a'] for neuron in self.network[-1]]}")
                    print("-" * 50)
    

myPMC = myPMC(numLayers=3, TabNumNeuronPerLayer=[2, 3, 2])
myPMC.initialize_network()
myPMC.display_network()

input_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
target_output = [[0, 1], [1, 0], [1, 0], [0, 1]]
learning_rate = 0.1

myPMC.train(input_data, target_output, learning_rate, epochs=10000)
myPMC.display_network()