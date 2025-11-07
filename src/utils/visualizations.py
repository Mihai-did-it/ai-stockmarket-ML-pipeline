"""
Visualization utilities.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def plot_feature_importance(importance_df: pd.DataFrame, 
                           top_n: int = 20,
                           save_path: str = None) -> None:
    """
    Plot feature importance.
    
    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        top_n: Number of top features to display
        save_path: Path to save the plot
    """
    if importance_df.empty:
        logger.warning("Empty feature importance DataFrame")
        return
    
    # Get top features
    top_features = importance_df.head(top_n)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Feature importance plot saved to {save_path}")
    
    plt.close()


def plot_confusion_matrix(cm: np.ndarray,
                         class_names: list = None,
                         save_path: str = None) -> None:
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix array
        class_names: List of class names
        save_path: Path to save the plot
    """
    if class_names is None:
        class_names = ['SELL', 'HOLD', 'BUY']
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")
    
    plt.close()


def plot_backtest_results(portfolio_history: pd.DataFrame,
                         save_path: str = None) -> None:
    """
    Plot backtesting results.
    
    Args:
        portfolio_history: DataFrame with portfolio history
        save_path: Path to save the plot
    """
    if isinstance(portfolio_history, list):
        portfolio_history = pd.DataFrame(portfolio_history)
    
    if portfolio_history.empty:
        logger.warning("Empty portfolio history")
        return
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Portfolio value over time
    axes[0].plot(portfolio_history['date'], portfolio_history['total_value'])
    axes[0].set_title('Portfolio Value Over Time')
    axes[0].set_ylabel('Value ($)')
    axes[0].grid(True, alpha=0.3)
    
    # Cash vs Holdings
    axes[1].plot(portfolio_history['date'], portfolio_history['cash'], label='Cash')
    axes[1].plot(portfolio_history['date'], portfolio_history['holdings_value'], label='Holdings')
    axes[1].set_title('Cash vs Holdings')
    axes[1].set_ylabel('Value ($)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Number of positions
    axes[2].plot(portfolio_history['date'], portfolio_history['num_positions'])
    axes[2].set_title('Number of Positions')
    axes[2].set_ylabel('Count')
    axes[2].set_xlabel('Date')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Backtest results saved to {save_path}")
    
    plt.close()


if __name__ == "__main__":
    # Example usage
    import numpy as np
    
    # Feature importance
    importance_df = pd.DataFrame({
        'feature': [f'feature_{i}' for i in range(20)],
        'importance': np.random.rand(20)
    }).sort_values('importance', ascending=False)
    
    plot_feature_importance(importance_df, save_path='test_importance.png')
    
    # Confusion matrix
    cm = np.array([[45, 10, 5], [8, 50, 12], [3, 15, 52]])
    plot_confusion_matrix(cm, save_path='test_cm.png')
    
    print("Visualizations created")
