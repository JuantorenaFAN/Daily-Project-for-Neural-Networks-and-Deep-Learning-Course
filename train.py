import numpy as np
from tqdm import tqdm
import os
from neural_network import ThreeLayerNN, SGDOptimizer
from utils import load_cifar10, preprocess_data, plot_training_history

class Trainer:
    def __init__(self, model, optimizer, reg_lambda=0.01):
        self.model = model
        self.optimizer = optimizer
        self.reg_lambda = reg_lambda
        self.history = {'train_loss': [], 'val_loss': [], 'train_accuracy': [], 'val_accuracy': []}
        self.best_val_accuracy = 0
        self.best_weights = None
    
    def train_epoch(self, X_train, y_train, batch_size=32):
        num_samples = X_train.shape[0]
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        epoch_loss = 0
        indices = np.random.permutation(num_samples)
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]
            
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]
            
            y_pred = self.model.forward(X_batch)
            loss = self.model.compute_loss(y_pred, y_batch, self.reg_lambda)
            epoch_loss += loss
            
            gradients = self.model.backward(X_batch, y_batch, self.reg_lambda)
            self.model.update_parameters(gradients, self.optimizer.get_lr())
            self.optimizer.step_decay()
        
        return epoch_loss / num_batches
    
    def validate(self, X_val, y_val):
        y_pred = self.model.forward(X_val)
        val_loss = self.model.compute_loss(y_pred, y_val, self.reg_lambda)
        val_accuracy = self.model.accuracy(X_val, y_val)
        return val_loss, val_accuracy
    
    def evaluate_train_accuracy(self, X_train, y_train, batch_size=1000):
        """计算训练集准确率（分批处理以节省内存）"""
        total_correct = 0
        total_samples = X_train.shape[0]
        num_batches = (total_samples + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_samples)
            X_batch = X_train[start_idx:end_idx]
            y_batch = y_train[start_idx:end_idx]
            
            y_pred = self.model.predict(X_batch)
            if len(y_batch.shape) > 1:  # 独热编码
                y_true = np.argmax(y_batch, axis=1)
            else:
                y_true = y_batch
            
            total_correct += np.sum(y_pred == y_true)
        
        return total_correct / total_samples
    
    def save_best_model(self, model_path, val_accuracy):
        if val_accuracy > self.best_val_accuracy:
            self.best_val_accuracy = val_accuracy
            self.best_weights = self.model.get_weights()
            self.model.save_model(model_path)
            return True
        return False
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, 
              batch_size=32, model_save_path='models/best_model.pkl', 
              verbose=True, patience=10):
        
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        
        best_epoch = 0
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(X_train, y_train, batch_size)
            val_loss, val_accuracy = self.validate(X_val, y_val)
            
            # 计算训练集准确率（每5轮计算一次以节省时间）
            if epoch % 5 == 0 or epoch == epochs - 1:
                train_accuracy = self.evaluate_train_accuracy(X_train, y_train)
            else:
                # 使用上一次的训练准确率或插值估计
                if self.history['train_accuracy']:
                    train_accuracy = self.history['train_accuracy'][-1]
                else:
                    train_accuracy = 0.0
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_accuracy'].append(train_accuracy)
            self.history['val_accuracy'].append(val_accuracy)
            
            is_best = self.save_best_model(model_save_path, val_accuracy)
            if is_best:
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}/{epochs} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Train Acc: {train_accuracy:.4f} | "
                      f"Val Acc: {val_accuracy:.4f}")
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)
        
        return self.history

def main():
    np.random.seed(42)
    
    print("Loading CIFAR-10...")
    (train_data, train_labels), (test_data, test_labels), class_names = load_cifar10()
    
    X_train, y_train, X_val, y_val = preprocess_data(train_data, train_labels, validation_split=0.1)
    X_test, y_test = preprocess_data(test_data, test_labels, validation_split=0)
    
    input_size = X_train.shape[1]
    hidden_size = 128
    output_size = len(class_names)
    
    model = ThreeLayerNN(input_size=input_size, hidden_size=hidden_size, 
                        output_size=output_size, activation='relu', random_seed=42)
    
    optimizer = SGDOptimizer(learning_rate=0.01, decay_rate=0.95, decay_steps=1000)
    trainer = Trainer(model=model, optimizer=optimizer, reg_lambda=0.001)
    
    history = trainer.train(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
                           epochs=200, batch_size=64, model_save_path='models/best_model.pkl',
                           verbose=True, patience=20)
    
    plot_training_history(history, save_path='results/training_history.png')
    
    test_accuracy = model.accuracy(X_test, y_test)
    print(f"Test accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    main() 