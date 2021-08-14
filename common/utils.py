import numpy as np
import matplotlib.pyplot as plt


def plot_training_curve(n_epochs, train_epoch_loss, val_epoch_loss):
    plt.plot(np.arange(1, n_epochs+1), train_epoch_loss, 'b-', label='train')
    plt.plot(np.arange(0, n_epochs)+1, val_epoch_loss, 'r-', label='val')
    plt.ylim(0)
    plt.title('Training Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
