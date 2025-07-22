#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from neural_network import ThreeLayerNN, SGDOptimizer
from train import Trainer
from test import load_and_test_model, test_random_baseline
from hyperparameter_search import run_grid_search, run_random_search
from utils import load_cifar10, preprocess_data, plot_sample_images, plot_training_history

class CIFAR10Classifier:
    def __init__(self):
        self.model = None
        self.trainer = None
        self.class_names = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
    
    def load_data(self, validation_split=0.1):
        (train_data, train_labels), (test_data, test_labels), self.class_names = load_cifar10()
        
        self.X_train, self.y_train, self.X_val, self.y_val = preprocess_data(
            train_data, train_labels, validation_split=validation_split)
        self.X_test, self.y_test = preprocess_data(test_data, test_labels, validation_split=0)
        
        return True
    
    def create_model(self, hidden_size=128, activation='relu', random_seed=42):
        if self.X_train is None:
            raise ValueError("Load data first")
        
        input_size = self.X_train.shape[1]
        output_size = len(self.class_names)
        
        self.model = ThreeLayerNN(input_size=input_size, hidden_size=hidden_size,
                                 output_size=output_size, activation=activation, random_seed=random_seed)
        return True
    
    def train_model(self, learning_rate=0.01, reg_lambda=0.001, epochs=100, 
                   batch_size=64, patience=20, model_save_path='models/best_model.pkl'):
        if self.model is None:
            raise ValueError("Create model first")
        
        optimizer = SGDOptimizer(learning_rate=learning_rate, decay_rate=0.95, decay_steps=1000)
        self.trainer = Trainer(model=self.model, optimizer=optimizer, reg_lambda=reg_lambda)
        
        history = self.trainer.train(X_train=self.X_train, y_train=self.y_train, X_val=self.X_val, y_val=self.y_val,
                                    epochs=epochs, batch_size=batch_size, model_save_path=model_save_path,
                                    verbose=True, patience=patience)
        return history
    
    def test_model(self, model_path='models/best_model.pkl', show_details=False):
        try:
            test_accuracy, class_accuracies, confusion_matrix = load_and_test_model(
                model_path=model_path, show_samples=show_details, show_weights=show_details)
            return test_accuracy
        except Exception as e:
            print(f"Error testing model: {e}")
            return None
    
    def search_hyperparameters(self, method='random', trials=20):
        if self.X_train is None:
            self.load_data()
        
        if method == 'grid':
            best_params, searcher = run_grid_search()
        elif method == 'random':
            best_params, searcher = run_random_search()
        else:
            raise ValueError("Method must be 'grid' or 'random'")
        
        return best_params
    
    def show_data_samples(self, num_samples=10):
        if self.X_train is None:
            self.load_data()
        
        train_images = self.X_train[:num_samples].reshape(-1, 32, 32, 3)
        train_labels = np.argmax(self.y_train[:num_samples], axis=1)
        
        plot_sample_images(train_images, train_labels, self.class_names, 
                          num_samples=num_samples, save_path='results/data_samples.png')

def main():
    parser = argparse.ArgumentParser(description='CIFAR-10 Three-Layer Neural Network Classifier')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'search', 'demo', 'all'], default='all')
    
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'sigmoid', 'tanh'])
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--reg_lambda', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--patience', type=int, default=20)
    
    parser.add_argument('--model_path', type=str, default='models/best_model.pkl')
    parser.add_argument('--search_method', type=str, default='random', choices=['grid', 'random'])
    parser.add_argument('--search_trials', type=int, default=20)
    parser.add_argument('--show_details', action='store_true')
    parser.add_argument('--random_seed', type=int, default=42)
    
    args = parser.parse_args()
    
    np.random.seed(args.random_seed)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    classifier = CIFAR10Classifier()
    
    if args.mode == 'train':
        classifier.load_data()
        classifier.create_model(hidden_size=args.hidden_size, activation=args.activation, random_seed=args.random_seed)
        
        history = classifier.train_model(learning_rate=args.learning_rate, reg_lambda=args.reg_lambda,
                                        epochs=args.epochs, batch_size=args.batch_size, 
                                        patience=args.patience, model_save_path=args.model_path)
        
        plot_training_history(history, save_path='results/training_history.png')
        
    elif args.mode == 'test':
        test_random_baseline()
        test_accuracy = classifier.test_model(model_path=args.model_path, show_details=args.show_details)
        
    elif args.mode == 'search':
        best_params = classifier.search_hyperparameters(method=args.search_method, trials=args.search_trials)
        
        print(f"\nRecommended parameters:")
        for key, value in best_params.items():
            print(f"  --{key} {value}")
        
    elif args.mode == 'demo':
        classifier.load_data()
        classifier.show_data_samples()
        
        classifier.create_model(hidden_size=64, activation='relu', random_seed=args.random_seed)
        
        history = classifier.train_model(learning_rate=0.01, reg_lambda=0.001, epochs=20, 
                                        batch_size=64, patience=5, model_save_path='models/demo_model.pkl')
        
        # 生成训练历史曲线
        plot_training_history(history, save_path='results/training_history.png')
        
        test_accuracy = classifier.test_model(model_path='models/demo_model.pkl', show_details=True)
        print(f"\nDemo completed! Model accuracy: {test_accuracy:.4f}")
        
    elif args.mode == 'all':
        print("1. Hyperparameter search...")
        best_params = classifier.search_hyperparameters(method='random', trials=10)
        
        print("\n2. Training with best parameters...")
        classifier.load_data()
        classifier.create_model(hidden_size=best_params['hidden_size'], activation=best_params['activation'],
                               random_seed=args.random_seed)
        
        history = classifier.train_model(learning_rate=best_params['learning_rate'], reg_lambda=best_params['reg_lambda'],
                                        epochs=args.epochs, batch_size=best_params['batch_size'], 
                                        patience=args.patience, model_save_path=args.model_path)
        
        # 生成训练历史曲线
        plot_training_history(history, save_path='results/training_history.png')
        
        print("\n3. Testing final model...")
        test_accuracy = classifier.test_model(model_path=args.model_path, show_details=True)
        print(f"\nFinal model accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    main() 