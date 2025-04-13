import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_data
from matplotlib import cm
from matplotlib.lines import Line2D
from datasets import unnormalize_image


#######FOR TRAINING GRAPH#########
def compute_dev_plot(n_epochs, data, name, legend=None, title=""):
    """
    Plots mean accuracy and standard deviation across multiple runs over epochs.
    
    Parameters:
    - n_epochs (int): Number of epochs.
    - data (numpy.ndarray): 2D array (runs x epochs) of accuracy values in percentage.
    - name (str): Output file name for saving the plot.
    - legend (list, optional): List of labels for the plot. Defaults to ["Mean Accuracy", "Standard Deviation"].
    - title (str, optional): Title of the plot.
    """
       
    data = data * 100
    epochs = np.arange(5, n_epochs + 1, 5)

    # Compute mean and standard deviation across runs
    mean_acc = np.mean(data, axis=0)
    std_acc = np.std(data, axis=0)

    # Plot mean accuracy curve
    mean_label = legend[0] if legend and len(legend) > 0 else "Mean Accuracy"
    plt.plot(epochs, mean_acc, color="blue", label=mean_label)
    
    # Plot standard deviation fill with label
    std_label = legend[2] if legend and len(legend) > 2 else "Standard Deviation"
    plt.fill_between(epochs, mean_acc - std_acc, mean_acc + std_acc, color="blue", alpha=0.2, label=std_label)

    # Customize the plot
    plt.xlabel("Epoch", fontsize = 18)
    plt.ylabel("Accuracy (%)",fontsize = 18)
    min_acc = np.floor(np.min([min(run) for run in data]) / 3) * 3
    max_acc = np.ceil(np.max([max(run) for run in data]) / 3) * 3
    plt.yticks(np.arange(min_acc, max_acc + 1, 1))
    plt.title(title, fontsize = 20)
    #plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(name)
    plt.close()

def result():
    # cifar 10 conv    
    results = [
        [80.52, 84.23, 86.57, 86.63, 86.93, 86.78, 86.90, 86.88],
        [80.21, 84.05, 86.51, 87.17, 87.01, 87.03, 87.29, 87.09],
        [80.92, 82.97, 86.55, 87.09, 87.28, 87.13, 87.30, 87.16],
        [80.21, 84.26, 86.15, 86.51, 86.49, 86.56, 86.74, 86.67],
        [80.27, 84.62, 86.25, 86.65, 86.84, 86.89, 86.82, 87.04]
    ]
    compute_dev_plot(40, results, title= 'Testing Accuracy for CIFAR10 with CNN', name = "CIFAR10conv")

    #mnist conv
    results = [
        [98.95, 98.94, 99.17, 99.08, 99.17],
        [98.78, 98.86, 99.19, 99.19, 99.20],
        [98.77, 98.96, 99.18, 99.21, 99.23],
        [98.86, 99.13, 99.24, 99.24, 99.27],
        [98.95, 99.05, 99.15, 99.14, 99.13]
    ]
    compute_dev_plot(25, results, title= 'Testing Accuracy for MNIST with CNN', name = "MNISTconv")

    #mnist fc
    results = [
        [97.70, 97.76, 98.29, 98.23, 98.25],  
        [97.55, 97.57, 98.41, 98.46, 98.41],          
        [97.19, 97.85, 98.29, 98.37, 98.37],          
        [97.69, 97.73, 98.34, 98.30, 98.45],          
        [97.59, 97.71, 98.35, 98.38, 98.33]     
    ]
    compute_dev_plot(25, results, title= 'Testing Accuracy for MNIST with FCN', name = "MNISTfc")

    #fmnist conv
    results = [
        [88.83, 89.12, 89.97, 89.95, 90.08],  
        [88.04, 89.54, 90.02, 90.22, 90.14],         
        [88.47, 89.41, 90.10, 90.11, 90.18],          
        [88.88, 89.20, 90.05, 90.20, 90.24],          
        [88.61, 88.67, 89.74, 89.62, 89.60]    
    ]
    compute_dev_plot(25, results, title= 'Testing Accuracy for FMNIST with CNN', name = "FMNISTconv")

    # #svnh conv
    results = [
        [94.74, 95.45, 96.16, 96.27, 96.30, 96.18, 96.26, 96.26],  
        [94.23, 95.68, 96.13, 96.22, 96.32, 96.31, 96.25, 96.27],        
        [94.57, 95.32, 96.37, 96.29, 96.34, 96.40, 96.38, 96.34],         
        [94.63, 95.64, 96.25, 96.24, 96.24, 96.17, 96.20, 96.25],          
        [94.61, 95.66, 96.40, 96.39, 96.37, 96.33, 96.41, 96.41]     
    ]
    compute_dev_plot(40, results, title= 'Testing Accuracy for SVHN with CNN', name = "SVHNconv")

    #cifar10 resnet
    results = [
        [85.84, 87.31, 92.35, 92.15, 92.25, 92.29, 92.36, 92.36],
        [87.74, 89.32, 92.37, 92.41, 92.49, 92.45, 92.78, 92.74],
        [89.01, 88.10, 92.35, 91.91, 92.41, 92.38, 92.28, 92.39],
        [86.94, 88.36, 92.05, 92.15, 92.38, 92.46, 92.41, 92.48],
        [86.99, 83.48, 92.43, 92.40, 92.44, 92.58, 92.63, 92.57]
    ]
    compute_dev_plot(40, results, title= 'Testing Accuracy for CIFAR10 with ResNet', name = "CIFAR10renset")

    # #mnist resnet
    results = [
        [99.08, 99.22, 99.60, 99.61, 99.60],  
        [99.18, 99.20, 99.64, 99.64, 99.66],  
        [99.16, 99.20, 99.61, 99.60, 99.64],  
        [99.06, 99.04, 99.53, 99.56, 99.63],  
        [98.95, 99.34, 99.62, 99.62, 99.63]   
    ]

    compute_dev_plot(25, results, title= 'Testing Accuracy for MNIST with ResNet', name = "MNISTrenset")

##########FOR DATASET PLOT##################
def plot_samples(dataset, rows=2, cols=10, save_path='sample_images.png'):
    trainloader, _, _, classes = load_data(dataset, batch_size_train=rows * cols)
    
    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    
    fig, axes = plt.subplots(rows, cols, figsize=((cols * 2), (rows * 2) + 1))
    fig.suptitle(f"Sample Images from {dataset}", fontsize=18)
    
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx >= len(images):
                continue
            ax = axes[i, j]
            img = images[idx]
            img = unnormalize_image(img, dataset)  
            label = labels[idx].item()
            
            if img.shape[0] == 1:  # Grayscale image
                ax.imshow(img.squeeze(), cmap='gray')
            else:
                ax.imshow(np.transpose(img.numpy(), (1, 2, 0)))
            
            ax.set_title(classes[label], fontsize=18)
            ax.axis('off')

    plt.savefig(save_path) 
    print(f"Saved sample images to {save_path}")



############## FOR ANALYSIS######################
def plot_with_dev(x, y_avg, y_dev, k=0.8, alpha=0.3, line_style="-", fig_size=(12, 7.5),
                  y_lim=None, x_ticks=None, legend=None, title=None, x_label=None, y_label=None, output_path='plot.png'):
    """
    Unified function to plot data with deviation bands and save results, with optional second y_avg and y_dev
    
    Parameters:
    x : list of arrays        - X-axis values for each series
    y_avg : list of arrays    - Average values for each series
    y_dev : list of arrays    - Standard deviations for each series
    y_avg2 : list of arrays   - (Optional) Second set of average values for each series
    y_dev2 : list of arrays   - (Optional) Second set of standard deviations for each series
    k : float                 - Standard deviation multiplier (default: 1)
    alpha : float             - Transparency for error bands (0-1)
    line_style : str          - Line style for averages
    fig_size : tuple          - Figure dimensions in inches
    y_lim : list              - Y-axis limits [min, max]
    x_ticks : list            - Custom X-axis tick labels
    legend : list             - Legend labels for each series
    title : str               - Plot title
    x_label : str             - X-axis label
    y_label : str             - Y-axis label
    output_path : str         - Path to save the figure
    """

    # --- Figure Setup ---
    plt.figure(figsize=fig_size)
    ax = plt.gca()  # Get current axis

    # --- Color Setup ---
    cols = cm.get_cmap("Set2")

    # --- Main Plotting ---
    for i in range(len(y_avg)):
        # Plot average line
        ax.plot(x[i], y_avg[i], line_style,
                color=cols.colors[i+2],
                label=legend[i] if legend else None,
                linewidth=2)

        # Add error band
        ax.fill_between(x[i],
                        y_avg[i] - k * y_dev[i],
                        y_avg[i] + k * y_dev[i],
                        alpha=alpha,
                        color=cols.colors[i+2])

    # --- Reference Line ---
    if y_lim is not None:
        ax.axhline(y_lim[1] // 2, color='black', linestyle='--', linewidth=2, zorder=0)

    # --- Axis Configuration ---
    if y_lim:
        ax.set_ylim(y_lim)
    if x_label:
        ax.set_xlabel(x_label, fontsize=16)
    if y_label:
        ax.set_ylabel(y_label, fontsize=18)
    if title:
        ax.set_title(title, fontsize=18)

    # Custom X-ticks handling
    if x_ticks:
        ax.set_xticks(np.arange(1, len(x_ticks) + 1))
        ax.set_xticklabels(x_ticks)
        ax.tick_params(axis='both', length=7, width=2, direction='in')
        plt.xticks(rotation=90, fontsize = 12)

    # --- Legend ---
    if legend:
        leg = ax.legend(loc='lower left', fontsize=18, framealpha=0.9)
        # Make legend lines thicker
        for legobj in leg.legend_handles:
            legobj.set_linewidth(4.0)

    # --- Save & Cleanup ---
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_with_dev_subplot(x, y_avg, y_dev, k= 1, alpha=0.3, line="-", second_axis=None, y_lim=None, x_ticks=None, name='noname', legend=None, title=None, f_size=None):
    """
    This function generates subplots with deviation bands (shaded regions) for multiple series of data.
    
    Parameters:
    - x : list of arrays        - X-axis values for each series
    - y_avg : list of arrays    - Average values for each series
    - y_dev : list of arrays    - Standard deviations for each series
    - k : float                 - Standard deviation multiplier (default: 1)
    - alpha : float             - Transparency of the error bands (default: 0.3)
    - line : str                - Line style for the plot (default: "-")
    - second_axis : bool       - (Optional) If true, plots a second y-axis
    - y_lim : list              - Y-axis limits [min, max]
    - x_ticks : list            - Custom X-axis tick labels
    - name : str                - Output file name for saving the plot
    - legend : list             - List of legend labels
    - title : str               - Title of the plot
    - f_size : tuple            - Figure size (width, height)
    """
    if f_size is not None:
        fig, ax = plt.subplots(len(y_avg)//2, figsize=f_size, sharex=True, sharey=True)
    else:
        fig, ax = plt.subplots(len(y_avg)//2, sharex=True, sharey=True)

    if title is not None:
        fig.suptitle(title, fontsize=22)

    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axes
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)

    plt.xlabel("Layers", fontsize=20) 
    plt.ylabel("No. of neighbors", fontsize=20)

    handles = []
    cols = cm.get_cmap("tab20")
    for i in range(len(y_avg)):
        # Check if legend is provided and has the right length
        if legend is not None and len(legend) > i:
            ax[i//2].plot(x[i], y_avg[i], line, label=legend[i], color=cols.colors[i+6])
        else:
            ax[i//2].plot(x[i], y_avg[i], line, color=cols.colors[i])
        
        handles.append(Line2D([0], [0], color=cols.colors[i+6], lw=2))
        
        ax[i//2].fill_between(x[i], (y_avg[i] - k * y_dev[i]), (y_avg[i] + k * y_dev[i]), alpha=alpha, color=cols.colors[i+6])

        ax[i//2].set_xticks(np.arange(1, len(x_ticks) + 1))

        ax[i//2].set_yticks([])
        fig.legend(handles, legend, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=len(legend) //2, fontsize=19, frameon=True)

    # Save plot and close the figure
    plt.savefig(name, bbox_inches="tight")
    plt.close(fig)

def plot_projection(output_path, x_ticks ,projs, color_dataset = False):
    """
    This function generates plots of projected images at different layers.
    
    Parameters:
    - output_path : str        - Path to save the generated plot
    - x_ticks : list           - Custom X-axis tick labels
    - projs : numpy.ndarray    - The projections to display (shape: [layers, samples, pixels])
    - color_dataset : bool     - If True, color the images; otherwise, display in grayscale
    """
    
    for i in range(projs.shape[1]):
        fig, ax = plt.subplots(1, len(x_ticks), figsize=(2*len(x_ticks), 2.2))
        for j in range(len(x_ticks)):
            if color_dataset:
                ax[j].imshow(np.moveaxis(projs[j, i, :].reshape(3, 32, 32), 0, -1))
            else:
                ax[j].imshow(projs[j, i, :].reshape(28, 28), cmap='gray')

            ax[j].set_xlabel(x_ticks[j])
            ax[j].set_xticks([])
            ax[j].set_yticks([])
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close(fig)