import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, 
                     title: str = "Predictions vs Actual", 
                     save_path: str = None):
    """Plot predictions vs actual values."""
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='Actual', alpha=0.7)
    plt.plot(y_pred, label='Predicted', alpha=0.7)
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray,
                   title: str = "Residuals", save_path: str = None):
    """Plot residual analysis."""
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].scatter(y_pred, residuals, alpha=0.5)
    axes[0].axhline(y=0, color='r', linestyle='--')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title('Residual Plot')
    
    axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Residuals')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Residual Distribution')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_feature_importance(importance: dict, feature_names: list,
                             title: str = "Feature Importance", 
                             save_path: str = None):
    """Plot feature importance."""
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': [importance.get(i, 0) for i in range(len(feature_names))]
    }).sort_values('importance', ascending=True)
    
    plt.figure(figsize=(10, 8))
    plt.barh(importance_df['feature'], importance_df['importance'])
    plt.title(title)
    plt.xlabel('Importance')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_training_history(history, save_path: str = None):
    """Plot training history for neural networks."""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    if 'val_mae' in history.history:
        plt.plot(history.history['val_mae'], label='Val MAE')
    plt.title('MAE')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_comparison(models_metrics: dict, save_path: str = None):
    """Plot comparison of multiple models."""
    df = pd.DataFrame(models_metrics).T
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics = ['RMSE', 'MAE', 'R2', 'Directional_Accuracy']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        df[metric].plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()