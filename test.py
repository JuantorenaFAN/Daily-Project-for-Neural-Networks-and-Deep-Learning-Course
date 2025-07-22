import numpy as np
import argparse
from neural_network import ThreeLayerNN
from utils import load_cifar10, preprocess_data, plot_sample_images, plot_weights_visualization
import matplotlib.pyplot as plt

def load_and_test_model(model_path, show_samples=False, show_weights=False):
    print("Loading CIFAR-10...")
    (train_data, train_labels), (test_data, test_labels), class_names = load_cifar10()
    
    X_test, y_test = preprocess_data(test_data, test_labels, validation_split=0)
    
    import pickle
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    config = model_data['config']
    
    model = ThreeLayerNN(input_size=config['input_size'], hidden_size=config['hidden_size'],
                        output_size=config['output_size'], activation=config['activation'], random_seed=42)
    
    model.load_model(model_path)
    
    test_accuracy = model.accuracy(X_test, y_test)
    y_pred = model.predict(X_test)
    y_true = np.argmax(y_test, axis=1)
    
    class_accuracies = []
    for i in range(len(class_names)):
        class_mask = y_true == i
        if np.sum(class_mask) > 0:
            class_acc = np.mean(y_pred[class_mask] == y_true[class_mask])
            class_accuracies.append(class_acc)
        else:
            class_accuracies.append(0.0)
    
    print(f"Overall test accuracy: {test_accuracy:.4f}")
    print("\nPer-class accuracy:")
    for i, (class_name, class_acc) in enumerate(zip(class_names, class_accuracies)):
        print(f"{class_name:>12}: {class_acc:.4f}")
    
    confusion_matrix = np.zeros((len(class_names), len(class_names)))
    for i in range(len(y_true)):
        confusion_matrix[y_true[i], y_pred[i]] += 1
    
    if show_samples:
        sample_indices = np.random.choice(len(X_test), 20, replace=False)
        
        fig, axes = plt.subplots(4, 5, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, idx in enumerate(sample_indices):
            img = X_test[idx].reshape(32, 32, 3)
            true_label = class_names[y_true[idx]]
            pred_label = class_names[y_pred[idx]]
            
            axes[i].imshow(img)
            color = 'green' if true_label == pred_label else 'red'
            axes[i].set_title(f'True: {true_label}\nPred: {pred_label}', color=color)
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('results/prediction_samples.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    if show_weights:
        weights = model.get_weights()
        plot_weights_visualization(weights, title="Trained Weights Distribution", 
                                 save_path='results/weights_visualization.png')
    
    return test_accuracy, class_accuracies, confusion_matrix

def test_random_baseline():
    (train_data, train_labels), (test_data, test_labels), class_names = load_cifar10()
    
    num_test = len(test_labels)
    random_predictions = np.random.randint(0, len(class_names), num_test)
    random_accuracy = np.mean(random_predictions == test_labels)
    
    print(f"Random baseline accuracy: {random_accuracy:.4f}")
    print(f"Theoretical random accuracy: {1.0/len(class_names):.4f}")
    
    return random_accuracy

def main():
    parser = argparse.ArgumentParser(description='Test trained neural network model')
    parser.add_argument('--model_path', type=str, default='models/best_model.pkl')
    parser.add_argument('--show_samples', action='store_true')
    parser.add_argument('--show_weights', action='store_true')
    parser.add_argument('--random_baseline', action='store_true')
    
    args = parser.parse_args()
    
    import os
    os.makedirs('results', exist_ok=True)
    
    if args.random_baseline:
        test_random_baseline()
        print("-" * 50)
    
    try:
        test_accuracy, class_accuracies, confusion_matrix = load_and_test_model(
            model_path=args.model_path, show_samples=args.show_samples, show_weights=args.show_weights)
        
        results_file = 'results/test_results.txt'
        with open(results_file, 'w') as f:
            f.write(f"Test Results\n")
            f.write(f"Model: {args.model_path}\n")
            f.write(f"Overall accuracy: {test_accuracy:.4f}\n\n")
            
            f.write("Per-class accuracy:\n")
            (_, _, _), (_, _), class_names = load_cifar10()
            for i, (class_name, class_acc) in enumerate(zip(class_names, class_accuracies)):
                f.write(f"{class_name:>12}: {class_acc:.4f}\n")
        
    except FileNotFoundError:
        print(f"Error: Model file {args.model_path} not found")
    except Exception as e:
        print(f"Error during testing: {e}")

if __name__ == "__main__":
    main() 