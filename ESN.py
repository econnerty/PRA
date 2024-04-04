import numpy as np

# Manual ESN Implementation
class SimpleESN:
    def __init__(self, n_reservoir, spectral_radius, sparsity,alpha=1.0,leaky_rate=0.5):
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.state = np.zeros(n_reservoir)
        self.W = None
        self.W_in = None
        self.W_bias = None  # Bias weights
        self.W_out = None
        self.alpha = alpha
        self.leaky_rate = leaky_rate
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def initialize_weights(self):
        # Internal weights
        W = np.random.rand(self.n_reservoir, self.n_reservoir) - 0.5
        W[np.random.rand(*W.shape) < self.sparsity] = 0  # Set sparsity
        radius = np.max(np.abs(np.linalg.eigvals(W)))
        self.W = W * (self.spectral_radius / radius)  # Scale weights
        
        # Input weights
        self.W_in = np.random.rand(self.n_reservoir, 1) * 2 - 1
        
        # Initialize bias weights
        self.W_bias = np.random.rand(self.n_reservoir) * 2 - 1

    def update_state(self, input):
        # Include bias term in the pre-activation signal
        pre_activation = (1-self.leaky_rate) * np.dot(self.W, self.state) + self.leaky_rate * (np.dot(self.W_in, input) + self.W_bias)
        self.state = np.tanh(pre_activation)
        #self.state = self.leaky_rate * updated_state + (1-self.leaky_rate) * self.state

    
    def predict(self, inputs):
        return np.dot(inputs, self.W_out.T)