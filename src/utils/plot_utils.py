import matplotlib.pyplot as plt
import numpy as np

def plot_loss_accuracy(train_losses, test_losses, train_accuracies, test_accuracies):
    """
    Enhanced plotting function with better visualization for convergence analysis.
    """
    epochs = range(1, len(train_losses) + 1)
    
    # Create a figure with better layout
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot losses
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2, alpha=0.8)
    ax1.plot(epochs, test_losses, 'r-', label='Test Loss', linewidth=2, alpha=0.8)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Add convergence indicators
    if len(train_losses) > 5:
        # Calculate moving average for smoother trend
        window_size = min(5, len(train_losses) // 4)
        train_ma = np.convolve(train_losses, np.ones(window_size)/window_size, mode='valid')
        test_ma = np.convolve(test_losses, np.ones(window_size)/window_size, mode='valid')
        ma_epochs = epochs[window_size-1:]
        
        ax1.plot(ma_epochs, train_ma, 'b--', alpha=0.6, linewidth=1, label='Train MA')
        ax1.plot(ma_epochs, test_ma, 'r--', alpha=0.6, linewidth=1, label='Test MA')
    
    # Plot accuracies
    ax2.plot(epochs, train_accuracies, 'b-', label='Train Acc', linewidth=2, alpha=0.8)
    ax2.plot(epochs, test_accuracies, 'r-', label='Test Acc', linewidth=2, alpha=0.8)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Add accuracy moving average
    if len(train_accuracies) > 5:
        train_acc_ma = np.convolve(train_accuracies, np.ones(window_size)/window_size, mode='valid')
        test_acc_ma = np.convolve(test_accuracies, np.ones(window_size)/window_size, mode='valid')
        
        ax2.plot(ma_epochs, train_acc_ma, 'b--', alpha=0.6, linewidth=1, label='Train MA')
        ax2.plot(ma_epochs, test_acc_ma, 'r--', alpha=0.6, linewidth=1, label='Test MA')
    
    # Add final performance text
    final_train_acc = train_accuracies[-1]
    final_test_acc = test_accuracies[-1]
    final_train_loss = train_losses[-1]
    final_test_loss = test_losses[-1]
    
    ax2.text(0.02, 0.98, f'Final Train Acc: {final_train_acc:.2f}%\nFinal Test Acc: {final_test_acc:.2f}%',
             transform=ax2.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax1.text(0.02, 0.98, f'Final Train Loss: {final_train_loss:.4f}\nFinal Test Loss: {final_test_loss:.4f}',
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'training_curves_epoch_{len(epochs)}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print convergence analysis
    print("\n" + "="*50)
    print("CONVERGENCE ANALYSIS")
    print("="*50)
    
    # Check if loss is still decreasing
    recent_epochs = min(10, len(train_losses) // 2)
    if recent_epochs > 1:
        recent_train_loss_trend = np.mean(train_losses[-recent_epochs:]) - np.mean(train_losses[-2*recent_epochs:-recent_epochs])
        recent_test_loss_trend = np.mean(test_losses[-recent_epochs:]) - np.mean(test_losses[-2*recent_epochs:-recent_epochs])
        
        print(f"Recent train loss trend: {'Decreasing' if recent_train_loss_trend < 0 else 'Increasing'} ({recent_train_loss_trend:.4f})")
        print(f"Recent test loss trend: {'Decreasing' if recent_test_loss_trend < 0 else 'Increasing'} ({recent_test_loss_trend:.4f})")
        
        # Overfitting check
        gap = final_train_acc - final_test_acc
        print(f"Train-Test accuracy gap: {gap:.2f}%")
        if gap > 10:
            print("⚠️  Potential overfitting detected!")
        elif gap < 2:
            print("✅ Good generalization!")
        
        # Convergence status
        if abs(recent_train_loss_trend) < 0.01 and abs(recent_test_loss_trend) < 0.01:
            print("✅ Model appears to have converged")
        else:
            print("📈 Model is still learning - consider more epochs")
    
    print("="*50)
