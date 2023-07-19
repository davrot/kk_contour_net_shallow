import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

mpl.rcParams["text.usetex"] = True
mpl.rcParams["font.family"] = "serif"


def plot_intermediate(
    train_accuracy: list[float],
    test_accuracy: list[float],
    train_losses: list[float],
    test_losses: list[float],
    save_name: str,
    reduction_factor: int = 1,
) -> None:
    assert len(train_accuracy) == len(test_accuracy)
    assert len(train_accuracy) == len(train_losses)
    assert len(train_accuracy) == len(test_losses)

    max_epochs: int = len(train_accuracy)
    # set stepsize
    x = np.arange(1, max_epochs + 1)

    stepsize = max_epochs // reduction_factor

    # accuracies
    plt.figure(figsize=[12, 7])
    plt.subplot(2, 1, 1)

    plt.plot(x, np.array(train_accuracy), label="Train")
    plt.plot(x, np.array(test_accuracy), label="Test")
    plt.title("Training and Testing Accuracy", fontsize=18)
    plt.xlabel("Epoch", fontsize=18)
    plt.ylabel("Accuracy (\\%)", fontsize=18)
    plt.legend(fontsize=14)
    plt.xticks(
        np.concatenate((np.array([1]), np.arange(stepsize, max_epochs + 1, stepsize))),
        np.concatenate((np.array([1]), np.arange(stepsize, max_epochs + 1, stepsize))),
    )

    # Increase tick label font size
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True)

    # losses
    plt.subplot(2, 1, 2)
    plt.plot(x, np.array(train_losses), label="Train")
    plt.plot(x, np.array(test_losses), label="Test")
    plt.title("Training and Testing Losses", fontsize=18)
    plt.xlabel("Epoch", fontsize=18)
    plt.ylabel("Loss", fontsize=18)
    plt.legend(fontsize=14)
    plt.xticks(
        np.concatenate((np.array([1]), np.arange(stepsize, max_epochs + 1, stepsize))),
        np.concatenate((np.array([1]), np.arange(stepsize, max_epochs + 1, stepsize))),
    )

    # Increase tick label font size
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True)

    plt.tight_layout()
    os.makedirs("performance_plots", exist_ok=True)
    plt.savefig(
        os.path.join(
            "performance_plots",
            f"performance_{save_name}.pdf",
        ),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()
