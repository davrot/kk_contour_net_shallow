import torch
from plot_as_grid import plot_in_grid

import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from functions.make_cnn import make_cnn  # noqa


# load on cpu
device = torch.device("cpu")

# path to NN
nn = "ArghCNN_numConvLayers3_outChannels[6, 8, 8]_kernelSize[7, 15]_leaky relu_stride1_trainFirstConvLayerTrue_seed287302_Coignless_801Epoch_2807-0857.pt"
PATH = f"../trained_models/{nn}"

# load and evaluate model
model = torch.load(PATH).to(device)
model.eval()
print("Full network:")
print(model)
print("")

# enter index to plot:
idx = int(input("Please select layer: "))
print(f"Selected layer {idx}:")
print(model[idx])

# bias
bias_input = input("Plot bias (y/n): ")
plot_bias: bool = False
if bias_input == "y":
    plot_bias = True
    bias = model[idx]._parameters["bias"].data
    print(bias)
else:
    bias = None

# visualize weights:
if idx > 0:
    weights = model[idx].weight.cpu().detach().clone().numpy()
    plot_in_grid(
        weights, colorbar=True, swap_channels=True, bias=bias, plot_bias=plot_bias
    )
else:
    weights = model[idx].weight.cpu().detach().clone().numpy()
    plot_in_grid(weights, colorbar=True, bias=bias, plot_bias=plot_bias)
