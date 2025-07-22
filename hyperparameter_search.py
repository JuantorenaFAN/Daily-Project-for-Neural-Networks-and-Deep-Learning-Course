import numpy as np
import itertools
import json
import os
from neural_network import ThreeLayerNN, SGDOptimizer
from train import Trainer
from utils import load_cifar10, preprocess_data
import matplotlib.pyplot as plt

def convert_numpy_types(obj):
    """递归转换NumPy数据类型为Python原生类型"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

class HyperparameterSearch:
    def __init__(self, X_train, y_train, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.input_size = X_train.shape[1]
        self.output_size = y_train.shape[1]
        self.results = []
    
    def grid_search(self, param_grid, max_epochs=50, patience=10, verbose=True):
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(itertools.product(*param_values))
        
        total_combinations = len(param_combinations)
        print(f"Grid search: {total_combinations} combinations")
        
        for i, param_combo in enumerate(param_combinations):
            params = dict(zip(param_names, param_combo))
            
            try:
                val_accuracy, train_history = self._train_with_params(params, max_epochs, patience, verbose=False)
                
                result = {'params': params.copy(), 'val_accuracy': val_accuracy, 'train_history': train_history}
                self.results.append(result)
                
                print(f"Combination {i+1}/{total_combinations}: Val Acc = {val_accuracy:.4f}")
                
            except Exception as e:
                result = {'params': params.copy(), 'val_accuracy': 0.0, 'train_history': None, 'error': str(e)}
                self.results.append(result)
        
        self.results.sort(key=lambda x: x['val_accuracy'], reverse=True)
        return self.results
    
    def _train_with_params(self, params, max_epochs, patience, verbose=False):
        model = ThreeLayerNN(input_size=self.input_size, hidden_size=params['hidden_size'],
                            output_size=self.output_size, activation=params['activation'], random_seed=42)
        
        optimizer = SGDOptimizer(learning_rate=params['learning_rate'],
                               decay_rate=params.get('decay_rate', 0.95),
                               decay_steps=params.get('decay_steps', 1000))
        
        trainer = Trainer(model=model, optimizer=optimizer, reg_lambda=params['reg_lambda'])
        
        history = trainer.train(X_train=self.X_train, y_train=self.y_train, X_val=self.X_val, y_val=self.y_val,
                               epochs=max_epochs, batch_size=params.get('batch_size', 64),
                               model_save_path=f'models/temp_model.pkl', verbose=verbose, patience=patience)
        
        return trainer.best_val_accuracy, history
    
    def random_search(self, param_distributions, n_trials=20, max_epochs=50, patience=10):
        print(f"Random search: {n_trials} trials")
        
        for trial in range(n_trials):
            params = {}
            for param_name, distribution in param_distributions.items():
                if distribution['type'] == 'choice':
                    params[param_name] = np.random.choice(distribution['values'])
                elif distribution['type'] == 'uniform':
                    params[param_name] = np.random.uniform(distribution['low'], distribution['high'])
                elif distribution['type'] == 'log_uniform':
                    log_low = np.log10(distribution['low'])
                    log_high = np.log10(distribution['high'])
                    params[param_name] = 10 ** np.random.uniform(log_low, log_high)
                elif distribution['type'] == 'int_uniform':
                    params[param_name] = np.random.randint(distribution['low'], distribution['high'] + 1)
            
            try:
                val_accuracy, train_history = self._train_with_params(params, max_epochs, patience, verbose=False)
                
                result = {'params': params.copy(), 'val_accuracy': val_accuracy, 'train_history': train_history}
                self.results.append(result)
                
                print(f"Trial {trial+1}/{n_trials}: Val Acc = {val_accuracy:.4f}")
                
            except Exception as e:
                pass
        
        self.results.sort(key=lambda x: x['val_accuracy'], reverse=True)
        return self.results
    
    def get_best_params(self, top_k=5):
        if not self.results:
            return None
        
        print(f"Top {min(top_k, len(self.results))} best parameters:")
        
        for i in range(min(top_k, len(self.results))):
            result = self.results[i]
            print(f"Rank {i+1} (Val Acc: {result['val_accuracy']:.4f}):")
            for key, value in result['params'].items():
                print(f"  {key}: {value}")
        
        return self.results[0]['params']
    
    def save_results(self, filename='hyperparameter_search_results.json'):
        os.makedirs('results', exist_ok=True)
        filepath = os.path.join('results', filename)
        
        # 转换NumPy类型为Python原生类型
        serializable_results = []
        for result in self.results:
            serializable_result = {
                'params': convert_numpy_types(result['params']),
                'val_accuracy': float(result['val_accuracy'])
            }
            if 'error' in result:
                serializable_result['error'] = result['error']
            serializable_results.append(serializable_result)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Search results saved to: {filepath}")
    
    def plot_search_results(self, save_path=None):
        if not self.results:
            return
        
        params_to_plot = ['learning_rate', 'hidden_size', 'reg_lambda']
        
        fig, axes = plt.subplots(1, len(params_to_plot), figsize=(15, 5))
        if len(params_to_plot) == 1:
            axes = [axes]
        
        for i, param_name in enumerate(params_to_plot):
            param_values = []
            accuracies = []
            
            for result in self.results:
                if param_name in result['params']:
                    param_values.append(result['params'][param_name])
                    accuracies.append(result['val_accuracy'])
            
            if param_values:
                axes[i].scatter(param_values, accuracies, alpha=0.7)
                axes[i].set_xlabel(param_name.replace('_', ' ').title())
                axes[i].set_ylabel('Validation Accuracy')
                axes[i].set_title(f'{param_name.replace("_", " ").title()} vs Validation Accuracy')
                axes[i].grid(True, alpha=0.3)
                
                if param_name == 'learning_rate' or param_name == 'reg_lambda':
                    axes[i].set_xscale('log')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

def run_grid_search():
    param_grid = {
        'learning_rate': [0.001, 0.01, 0.1],
        'hidden_size': [64, 128, 256],
        'reg_lambda': [0.0001, 0.001, 0.01],
        'activation': ['relu', 'sigmoid', 'tanh'],
        'batch_size': [32, 64]
    }
    
    (train_data, train_labels), (test_data, test_labels), class_names = load_cifar10()
    X_train, y_train, X_val, y_val = preprocess_data(train_data, train_labels, validation_split=0.1)
    
    searcher = HyperparameterSearch(X_train, y_train, X_val, y_val)
    results = searcher.grid_search(param_grid, max_epochs=30, patience=8)
    best_params = searcher.get_best_params(top_k=5)
    
    searcher.save_results('grid_search_results.json')
    searcher.plot_search_results('results/grid_search_visualization.png')
    
    return best_params, searcher

def run_random_search():
    param_distributions = {
        'learning_rate': {'type': 'log_uniform', 'low': 0.0001, 'high': 0.1},
        'hidden_size': {'type': 'choice', 'values': [32, 64, 128, 256, 512]},
        'reg_lambda': {'type': 'log_uniform', 'low': 0.00001, 'high': 0.1},
        'activation': {'type': 'choice', 'values': ['relu', 'sigmoid', 'tanh']},
        'batch_size': {'type': 'choice', 'values': [16, 32, 64, 128]}
    }
    
    (train_data, train_labels), (test_data, test_labels), class_names = load_cifar10()
    X_train, y_train, X_val, y_val = preprocess_data(train_data, train_labels, validation_split=0.1)
    
    searcher = HyperparameterSearch(X_train, y_train, X_val, y_val)
    results = searcher.random_search(param_distributions, n_trials=30, max_epochs=30, patience=8)
    best_params = searcher.get_best_params(top_k=5)
    
    searcher.save_results('random_search_results.json')
    searcher.plot_search_results('results/random_search_visualization.png')
    
    return best_params, searcher

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Hyperparameter search')
    parser.add_argument('--method', type=str, choices=['grid', 'random', 'both'], default='random')
    parser.add_argument('--trials', type=int, default=20)
    
    args = parser.parse_args()
    np.random.seed(42)
    
    if args.method == 'grid':
        best_params, searcher = run_grid_search()
    elif args.method == 'random':
        best_params, searcher = run_random_search()
    elif args.method == 'both':
        grid_best_params, grid_searcher = run_grid_search()
        random_best_params, random_searcher = run_random_search()
        
        if grid_searcher.results[0]['val_accuracy'] > random_searcher.results[0]['val_accuracy']:
            best_params = grid_best_params
        else:
            best_params = random_best_params
    
    print(f"\nRecommended best parameters:")
    for key, value in best_params.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main() 