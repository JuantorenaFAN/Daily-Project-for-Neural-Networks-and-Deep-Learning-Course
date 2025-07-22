import numpy as np
import pickle
import os

class ActivationFunction:
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x):
        return (x > 0).astype(float)
    
    @staticmethod
    def sigmoid(x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def sigmoid_derivative(x):
        s = ActivationFunction.sigmoid(x)
        return s * (1 - s)
    
    @staticmethod
    def tanh(x):
        return np.tanh(x)
    
    @staticmethod
    def tanh_derivative(x):
        return 1 - np.tanh(x)**2

class ThreeLayerNN:
    def __init__(self, input_size, hidden_size, output_size, activation='relu', random_seed=42):
        np.random.seed(random_seed)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation_type = activation
        
        if activation == 'relu':
            self.activation = ActivationFunction.relu
            self.activation_derivative = ActivationFunction.relu_derivative
        elif activation == 'sigmoid':
            self.activation = ActivationFunction.sigmoid
            self.activation_derivative = ActivationFunction.sigmoid_derivative
        elif activation == 'tanh':
            self.activation = ActivationFunction.tanh
            self.activation_derivative = ActivationFunction.tanh_derivative
        else:
            raise ValueError("Unsupported activation function")
        
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))
        
        self.z1 = None
        self.a1 = None
        self.z2 = None
        self.a2 = None
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.activation(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        
        exp_scores = np.exp(self.z2 - np.max(self.z2, axis=1, keepdims=True))
        self.a2 = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        return self.a2
    
    def compute_loss(self, y_pred, y_true, reg_lambda=0.01):
        m = y_true.shape[0]
        y_pred_clipped = np.clip(y_pred, 1e-12, 1 - 1e-12)
        cross_entropy_loss = -np.mean(np.sum(y_true * np.log(y_pred_clipped), axis=1))
        l2_reg = reg_lambda * (np.sum(self.W1**2) + np.sum(self.W2**2)) / 2
        return cross_entropy_loss + l2_reg
    
    def backward(self, X, y_true, reg_lambda=0.01):
        m = X.shape[0]
        
        dz2 = self.a2 - y_true
        dW2 = np.dot(self.a1.T, dz2) / m + reg_lambda * self.W2
        db2 = np.mean(dz2, axis=0, keepdims=True)
        
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.activation_derivative(self.z1)
        dW1 = np.dot(X.T, dz1) / m + reg_lambda * self.W1
        db1 = np.mean(dz1, axis=0, keepdims=True)
        
        return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
    
    def update_parameters(self, gradients, learning_rate):
        self.W1 -= learning_rate * gradients['dW1']
        self.b1 -= learning_rate * gradients['db1']
        self.W2 -= learning_rate * gradients['dW2']
        self.b2 -= learning_rate * gradients['db2']
    
    def predict(self, X):
        y_pred = self.forward(X)
        return np.argmax(y_pred, axis=1)
    
    def predict_proba(self, X):
        return self.forward(X)
    
    def accuracy(self, X, y_true):
        y_pred = self.predict(X)
        if len(y_true.shape) > 1:
            y_true = np.argmax(y_true, axis=1)
        return np.mean(y_pred == y_true)
    
    def get_weights(self):
        return {'W1': self.W1.copy(), 'b1': self.b1.copy(), 'W2': self.W2.copy(), 'b2': self.b2.copy()}
    
    def set_weights(self, weights):
        self.W1 = weights['W1'].copy()
        self.b1 = weights['b1'].copy()
        self.W2 = weights['W2'].copy()
        self.b2 = weights['b2'].copy()
    
    def save_model(self, filepath):
        model_data = {
            'weights': self.get_weights(),
            'config': {
                'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'output_size': self.output_size,
                'activation': self.activation_type
            }
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        config = model_data['config']
        if (config['input_size'] != self.input_size or 
            config['hidden_size'] != self.hidden_size or
            config['output_size'] != self.output_size):
            raise ValueError("Model configuration mismatch")
        
        self.set_weights(model_data['weights'])

class SGDOptimizer:
    def __init__(self, learning_rate=0.01, decay_rate=0.95, decay_steps=1000):
        self.initial_lr = learning_rate
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.step_count = 0
    
    def get_lr(self):
        return self.learning_rate
    
    def step_decay(self):
        self.step_count += 1
        if self.step_count % self.decay_steps == 0:
            self.learning_rate *= self.decay_rate
    
    def reset(self):
        self.learning_rate = self.initial_lr
        self.step_count = 0 