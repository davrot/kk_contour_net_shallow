import torch
import sys
import os
import matplotlib.pyplot as plt  # noqa
import matplotlib as mpl

mpl.rcParams["text.usetex"] = True
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.size"] = 14

# import files from parent dir
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from plot_as_grid import plot_in_grid
from functions.make_cnn import make_cnn  # noqa


# load on cpu
device = torch.device("cpu")

# path to NN
nn = "ArghCNN_numConvLayers3_outChannels[8, 8, 8]_kernelSize[7, 15]_leaky relu_stride1_trainFirstConvLayerTrue_seed293051_Natural_249Epoch_1308-1145"
PATH = f"D:/Katha/Neuroscience/Semester 4/newCode/kk_contour_net_shallow-main/corner888/{nn}.pt"
SAVE_PATH = "20 cnns weights/corner 888/seed293051_Natural_249Epoch_1308-1145"

# load and evaluate model
model = torch.load(PATH).to(device)
model.eval()
print("Full network:")
print(model)
print("")
# enter index to plot:
idx = int(input("Please select layer: "))
print(f"Selected layer: {idx, model[idx]}")

# bias
bias_input = input("Plot bias (y/n): ")
plot_bias: bool = False
if bias_input == "y":
    plot_bias = True
    bias = model[idx]._parameters["bias"].data
    print(bias)
else:
    bias = None

# show last layer's weights.
if idx == len(model) - 1:
    linear_weights = model[idx].weight.cpu().detach().clone().numpy()

    weights = linear_weights.reshape(2, 8, 74, 74)
    plot_in_grid(
        weights,
        fig_size=(10, 7),
        savetitle=f"{SAVE_PATH}_final_layer",
        colorbar=True,
        swap_channels=True,
        bias=bias,
        plot_bias=plot_bias,
    )

# visualize weights:
elif idx > 0:
    weights = model[idx].weight.cpu().detach().clone().numpy()

    if idx == 5:
        swap_channels = False
        layer = 3
    else:
        swap_channels = True
        layer = 2

    # plot weights
    plot_in_grid(
        weights,
        fig_size=(11, 7),
        savetitle=f"{SAVE_PATH}_conv{layer}",
        colorbar=True,
        swap_channels=swap_channels,
        bias=bias,
        plot_bias=plot_bias,
    )
else:
    first_weights = model[idx].weight.cpu().detach().clone().numpy()

    # reshape first layer weights:
    reshape_input = input("Reshape weights to 4rows 8 cols (y/n): ")
    if reshape_input == "y":
        weights = first_weights.reshape(
            8, 4, first_weights.shape[-2], first_weights.shape[-1]
        )
    else:
        weights = first_weights
    plot_in_grid(
        weights,
        fig_size=(17, 17),
        savetitle=f"{SAVE_PATH}_conv1",
        colorbar=True,
        bias=bias,
        plot_bias=plot_bias,
    )
