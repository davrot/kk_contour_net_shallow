import torch
import torchvision as tv
import matplotlib.pyplot as plt
import os
import glob
from natsort import natsorted
import sys

# import numpy as np

layer_id: int = int(sys.argv[1])
scale_each_inner: bool = True
scale_each_outer: bool = True

model_path: str = "trained_models"
filename_list: list = natsorted(glob.glob(os.path.join(model_path, str("*.pt"))))
assert len(filename_list) > 0
model_filename: str = filename_list[-1]
print(f"Load filename: {model_filename}")

model = torch.load(model_filename, map_location=torch.device("cpu"))
assert layer_id < len(model)

print("Full network:")
print(model)
print("")
print(f"Selected layer {layer_id}:")
print(model[layer_id])

# ---
weights = model[layer_id]._parameters["weight"].data
bias = model[layer_id]._parameters["bias"].data

weight_grid = tv.utils.make_grid(
    weights.movedim(0, 1),
    nrow=8,
    padding=2,
    scale_each=scale_each_inner,
    pad_value=float("NaN"),
)
weight_grid = tv.utils.make_grid(
    weight_grid.unsqueeze(1), nrow=4, padding=2, scale_each=scale_each_outer
)


v_max_abs = torch.abs(weight_grid[0, ...]).max()

plt.subplot(3, 1, (1, 2))
plt.imshow(
    weight_grid[0, ...],
#    vmin=-v_max_abs,
#    vmax=v_max_abs,
    cmap="hot",
)
plt.axis("off")
#plt.colorbar()
plt.title("Weights")

plt.subplot(3, 1, 3)
plt.plot(bias)
plt.title("Bias")

plt.show()
