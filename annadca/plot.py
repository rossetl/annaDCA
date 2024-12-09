import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def _plot_hist(ax, data1, data2, color, dim, orientation='vertical'):    
    ax.hist(data1[:, dim], bins=30, color=color, histtype='step', zorder=0, density=True, orientation=orientation, lw=1)
    ax.hist(data2[:, dim], bins=30, color=color, histtype='bar', zorder=1, density=True, orientation=orientation, alpha=0.3)
    ax.axis('off')
    

def plot_PCA(
    fig: plt.figure,
    data1: np.array,
    labels1: np.ndarray,
    data2: np.array,
    labels2: np.ndarray,
    label_to_color: dict,
    dim1: int,
    dim2: int,
    title: str,
    split_labels: bool = False,
) -> plt.figure:
    """Plot the scatter plot of data1 and data2 in the space defined by the (dim1, dim2) principal components.
    Also plots the histograms projected along the x and y axes.
    
    Args:
        fig (plt.figure): Matplotlib figure.
        data1 (np.array): Data array.
        labels1 (np.ndarray): Labels array.
        data2 (np.array): Data array.
        labels2 (np.ndarray): Labels array.
        label_to_color (dict): Dictionary of label to color.
        dim1 (int): First principal component.
        dim2 (int): Second principal component.
        title (str): Title of the plot.
        split_labels (bool, optional): If True, the histograms are split by label. Defaults to False.
    """
    gs = GridSpec(4, 4)

    ax_scatter = fig.add_subplot(gs[1:4, 0:3])
    ax_hist_x = fig.add_subplot(gs[0, 0:3])
    ax_hist_y = fig.add_subplot(gs[1:4, 3])
    
    unique_labels = np.unique(labels1)
    for label in unique_labels:
        mask1 = labels1 == label
        mask2 = labels2 == label
        ax_scatter.scatter(data1[mask1, dim1], data1[mask1, dim2], color=label_to_color[label], s=50, zorder=0, alpha=0.3, label=label)
        if split_labels:
            _plot_hist(ax_hist_x, data1[mask1], data2[mask2], color=label_to_color[label], dim=dim1, orientation='vertical')
            _plot_hist(ax_hist_y,data1[mask1], data2[mask2], color=label_to_color[label], dim=dim2, orientation='horizontal')
    
    for label in unique_labels:
        mask = labels2 == label
        ax_scatter.scatter(data2[mask, dim1], data2[mask, dim2], color=label_to_color[label], s=20, zorder=2, edgecolor='black', marker='o', alpha=1, linewidth=0.4)
    
    if not split_labels:
        _plot_hist(ax_hist_x, data1, data2, 'red', dim1)
        _plot_hist(ax_hist_y, data1, data2, 'red', dim2, orientation='horizontal')
    
    ax_scatter.set_xlabel(f"PC {dim1 + 1}")
    ax_scatter.set_ylabel(f"PC {dim2 + 1}")
    
    fig.suptitle(title)
    return fig