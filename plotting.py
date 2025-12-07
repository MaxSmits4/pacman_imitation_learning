"""
Visualization utilities for training monitoring.
"""

import matplotlib.pyplot as plt


def setup_training_plots(num_plots):
    """
    Initialize matplotlib figures for loss and accuracy plots.

    Args:
        num_plots: Number of loss subplots to create

    Returns:
        tuple: (fig, axes, ax_acc) - Figure, loss axes list, accuracy axis
    """
    fig = plt.figure(figsize=(5 * num_plots + 8, 4))

    # Create grid: loss plots on left, accuracy on right
    gs = fig.add_gridspec(
        1, num_plots + 1, width_ratios=[5] * num_plots + [8]
    )

    # Loss subplot axes
    axes = [fig.add_subplot(gs[0, i]) for i in range(num_plots)]

    # Accuracy subplot axis
    ax_acc = fig.add_subplot(gs[0, num_plots])
    ax_acc.set_title("Test Accuracy vs Epoch")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.grid()

    # Bring matplotlib window to front
    fig.canvas.manager.show()
    fig.canvas.flush_events()

    return fig, axes, ax_acc


def update_training_plots(
    axes, ax_acc, plot_idx, losses, epoch_numbers,
    test_accuracies, epoch, fig, is_final=False
):
    """
    Update loss and accuracy plots.

    Args:
        axes: List of loss plot axes
        ax_acc: Accuracy plot axis
        plot_idx: Current loss plot index
        losses: List of all loss values
        epoch_numbers: List of epoch numbers
        test_accuracies: List of accuracy percentages
        epoch: Current epoch number
        fig: Figure object
        is_final: Whether this is the final update
    """
    # Update loss plot
    axes[plot_idx].plot(losses)
    axes[plot_idx].set_title(f"Epoch {epoch}" if not is_final else "Final")
    axes[plot_idx].set_ylabel("Loss")
    axes[plot_idx].set_xlabel("Steps")
    axes[plot_idx].set_xscale("log")
    axes[plot_idx].grid()

    # Update accuracy plot
    ax_acc.clear()
    ax_acc.plot(
        epoch_numbers, test_accuracies, 'b-',
        linewidth=2, marker='o'
    )
    title_suffix = " (Final)" if is_final else ""
    ax_acc.set_title(f"Test Accuracy vs Epoch{title_suffix}")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.grid()
    ax_acc.set_ylim([0, 100])

    # Refresh display
    plt.figure(fig.number)
    plt.tight_layout()
    if is_final:
        plt.draw()
        plt.pause(0.1)
    else:
        plt.pause(0.01)
