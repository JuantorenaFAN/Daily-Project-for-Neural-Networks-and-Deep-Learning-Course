import numpy as np
import matplotlib
matplotlib.use('Agg')  # 不显示窗口，只保存图片
import matplotlib.pyplot as plt
import pickle
import os
from urllib.request import urlretrieve
import tarfile

def download_cifar10(data_dir='./data'):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = os.path.join(data_dir, "cifar-10-python.tar.gz")
    
    if not os.path.exists(filename):
        print("Downloading CIFAR-10 dataset...")
        urlretrieve(url, filename)
    
    extract_dir = os.path.join(data_dir, "cifar-10-batches-py")
    if not os.path.exists(extract_dir):
        with tarfile.open(filename, 'r:gz') as tar:
            tar.extractall(data_dir)
    
    return extract_dir

def load_cifar10_batch(file_path):
    with open(file_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    
    data = batch[b'data']
    labels = batch[b'labels']
    data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    
    return data, np.array(labels)

def load_cifar10(data_dir='./data'):
    extract_dir = download_cifar10(data_dir)
    
    train_data = []
    train_labels = []
    
    for i in range(1, 6):
        batch_file = os.path.join(extract_dir, f'data_batch_{i}')
        data, labels = load_cifar10_batch(batch_file)
        train_data.append(data)
        train_labels.append(labels)
    
    train_data = np.concatenate(train_data, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    
    test_file = os.path.join(extract_dir, 'test_batch')
    test_data, test_labels = load_cifar10_batch(test_file)
    
    meta_file = os.path.join(extract_dir, 'batches.meta')
    with open(meta_file, 'rb') as f:
        meta = pickle.load(f, encoding='bytes')
    class_names = [name.decode('utf-8') for name in meta[b'label_names']]
    
    return (train_data, train_labels), (test_data, test_labels), class_names

def preprocess_data(X, y, validation_split=0.1):
    X = X.astype(np.float32) / 255.0
    X_flat = X.reshape(X.shape[0], -1)
    num_classes = len(np.unique(y))
    y_onehot = np.eye(num_classes)[y]
    
    if validation_split > 0:
        num_val = int(len(X_flat) * validation_split)
        indices = np.random.permutation(len(X_flat))
        
        val_indices = indices[:num_val]
        train_indices = indices[num_val:]
        
        X_train = X_flat[train_indices]
        y_train = y_onehot[train_indices]
        X_val = X_flat[val_indices]
        y_val = y_onehot[val_indices]
        
        return X_train, y_train, X_val, y_val
    else:
        return X_flat, y_onehot

def plot_training_history(history, save_path=None):
    """绘制完整的训练历史：损失曲线和准确率曲线"""
    print("Generating training history plots...")
    
    # 确保保存目录存在
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 损失曲线
    axes[0].plot(history['train_loss'], label='Training Loss', color='blue', linewidth=2)
    axes[0].plot(history['val_loss'], label='Validation Loss', color='red', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 准确率曲线
    axes[1].plot(history['train_accuracy'], label='Training Accuracy', color='blue', linewidth=2)
    axes[1].plot(history['val_accuracy'], label='Validation Accuracy', color='red', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 验证集准确率单独显示（放大）
    axes[2].plot(history['val_accuracy'], label='Validation Accuracy', color='green', linewidth=2, marker='o', markersize=3)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Accuracy')
    axes[2].set_title('Validation Accuracy (Detailed)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # 在验证准确率图上标注最高点
    if history['val_accuracy']:
        max_acc = max(history['val_accuracy'])
        max_epoch = history['val_accuracy'].index(max_acc)
        axes[2].annotate(f'Max: {max_acc:.4f}', 
                        xy=(max_epoch, max_acc), 
                        xytext=(max_epoch + len(history['val_accuracy'])*0.1, max_acc),
                        arrowprops=dict(arrowstyle='->', color='red'),
                        fontsize=10, color='red')
    
    plt.tight_layout()
    
    # 总是保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}")
    else:
        plt.savefig('results/training_history.png', dpi=300, bbox_inches='tight')
        print("Training history plot saved to: results/training_history.png")
    
    plt.close()  # 关闭图形，释放内存

def plot_weights_visualization(weights, title="Weights Visualization", save_path=None):
    print("Generating weights visualization...")
    
    # 确保保存目录存在
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()
    
    for i, (layer_name, weight_matrix) in enumerate(weights.items()):
        if i >= 4:
            break
            
        ax = axes[i]
        im = ax.imshow(weight_matrix, cmap='RdBu', aspect='auto')
        ax.set_title(f'{layer_name} Weight Distribution')
        ax.set_xlabel('Input Dimension')
        ax.set_ylabel('Output Dimension')
        plt.colorbar(im, ax=ax)
    
    for i in range(len(weights), 4):
        axes[i].set_visible(False)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    # 总是保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Weights visualization saved to: {save_path}")
    else:
        plt.savefig('results/weights_visualization.png', dpi=300, bbox_inches='tight')
        print("Weights visualization saved to: results/weights_visualization.png")
    
    plt.close()  # 关闭图形，释放内存

def plot_sample_images(X, y, class_names, num_samples=10, save_path=None):
    print("Generating sample images...")
    
    # 确保保存目录存在
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()
    
    for i in range(num_samples):
        if len(X.shape) == 2:
            img = X[i].reshape(32, 32, 3)
        else:
            img = X[i]
        
        axes[i].imshow(img)
        axes[i].set_title(f'{class_names[y[i]]}')
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # 总是保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Sample images saved to: {save_path}")
    else:
        plt.savefig('results/data_samples.png', dpi=300, bbox_inches='tight')
        print("Sample images saved to: results/data_samples.png")
    
    plt.close()  # 关闭图形，释放内存 