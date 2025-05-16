import matplotlib.pyplot as plt

def plot_loss_accuracy(train_losses, test_losses, train_accs, test_accs):
    epochs = range(1, len(train_losses) + 1)

    fig, ax1 = plt.subplots()

    color1 = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color1)
    ax1.plot(epochs, train_losses, label='Train Loss', color=color1, linestyle='--')
    ax1.plot(epochs, test_losses, label='Test Loss', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Accuracy (%)', color=color2)
    ax2.plot(epochs, train_accs, label='Train Acc', color=color2, linestyle='--')
    ax2.plot(epochs, test_accs, label='Test Acc', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    fig.tight_layout()
    plt.title('Training & Validation Loss/Accuracy')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.show()
