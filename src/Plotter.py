import matplotlib.pyplot as plt

def plot_loss(loss, label, title, xlab, ylab, color='blue', **kwargs):
    # Plot the loss evolution as a 2D Line
    plt.plot(loss, label=label, color=color, **kwargs)
    
    # Specify the title
    plt.title(title)

    # Specify the Ox label
    plt.xlabel(xlab)

    # Specify the Oy label
    plt.ylabel(ylab)

    # Specify the legend of the plot
    plt.legend()