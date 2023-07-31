import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import glob
from natsort import natsorted


mpl.rcParams["text.usetex"] = True
mpl.rcParams["font.family"] = "serif"


def plot_performance_across_channels(
    filename_list: list[str], channel_idx: int, saveplot: bool
) -> None:
    """
    y-axis: accuracies
    x-axis: number of output channels in first layer
    """

    train_accuracy: list = []
    test_accuracy: list = []
    output_channels: list = []

    for file in filename_list:
        data = np.load(file)
        output_channels.append(data["output_channels"])
        train_accuracy.append(data["train_accuracy"])
        test_accuracy.append(data["test_accuracy"])

    # get only first output channel:
    out_channel_size = [out[channel_idx] for out in output_channels]

    # get max accuracy of trained NNs
    max_train_acc = [train.max() for train in train_accuracy]
    max_test_acc = [test.max() for test in test_accuracy]

    plt.figure(figsize=[12, 7])
    plt.plot(out_channel_size, np.array(max_train_acc), label="Train")
    plt.plot(out_channel_size, np.array(max_test_acc), label="Test")
    plt.title("Training and Testing Accuracy", fontsize=18)
    plt.xlabel(
        f"Number of features in convolutional layer {channel_idx+1}", fontsize=18
    )
    plt.ylabel("Max. accuracy (\\%)", fontsize=18)
    plt.legend(fontsize=14)

    # Increase tick label font size
    plt.xticks(out_channel_size, fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True)

    plt.tight_layout()
    if saveplot:
        os.makedirs("performance_plots", exist_ok=True)
        plt.savefig(
            os.path.join(
                "performance_plots",
                f"feature_perf_{output_channels}.pdf",
            ),
            dpi=300,
            bbox_inches="tight",
        )
    plt.show()


if __name__ == "__main__":
    path: str = "/home/kk/Documents/Semester4/code/Classic_contour_net_shallow/performance_data/"
    filename_list = natsorted(glob.glob(os.path.join(path, "performances_*.npz")))
    plot_performance_across_channels(
        filename_list=filename_list, channel_idx=0, saveplot=True
    )
