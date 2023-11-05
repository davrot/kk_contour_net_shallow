# %%
import torch
import random
import re
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import matplotlib as mpl

mpl.rcParams["text.usetex"] = True
mpl.rcParams["font.family"] = "serif"

import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from functions.analyse_network import analyse_network
from functions.set_seed import set_seed

# define parameters
num_iterations: int = 100000
learning_rate: float = 0.1
apply_input_mask: bool = True
mark_region_in_plot: bool = True
sheduler_patience: int = 500
sheduler_factor: float = 0.9
sheduler_eps = 1e-08
target_image_active: float = 1e4
random_seed = random.randint(0, 100)
save_final: bool = True
model_str: str = "CORNER_888"

# set seet
set_seed(random_seed)
print(f"Random seed: {random_seed}")

# path to NN
condition: str = "corner_888_poster"
pattern = r"seed\d+_Natural_\d+Epoch"
nn = "ArghCNN_numConvLayers3_outChannels[8, 8, 8]_kernelSize[7, 15]_leaky relu_stride1_trainFirstConvLayerTrue_seed291857_Natural_1351Epoch_3107-2121.pt"
PATH = f"./trained_models/{nn}"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
# load and eval model
model = torch.load(PATH).to(device)
model.eval()
print("Full network:")
print(model)
print("")


# enter index to plot:
idx = int(input("Please select layer: "))
print(f"Selected layer: {model[idx]}")
assert idx < len(model)
model = model[: idx + 1]

# random input
input_img = torch.rand(1, 200, 200).to(device)
input_img = input_img.unsqueeze(0)
input_img.requires_grad_(True)  # type: ignore
print(input_img.min(), input_img.max())

input_shape = input_img.shape
assert input_shape[-2] == input_shape[-1]
coordinate_list, layer_type_list, pixel_used = analyse_network(
    model=model, input_shape=int(input_shape[-1])
)


output_shape = model(input_img).shape


target_image = torch.zeros(
    (*output_shape,), dtype=input_img.dtype, device=input_img.device
)


# image to parameter (2B optimized)
input_parameter = torch.nn.Parameter(input_img)


if len(target_image.shape) == 2:
    print((f"Available max positions: f:{target_image.shape[1] - 1} "))

    # select neuron and plot for all feature maps (?)
    neuron_f = int(input("Please select neuron_f: "))
    print(f"Selected neuron {neuron_f}")
    target_image[0, neuron_f] = 1e4
else:
    print(
        (
            f"Available max positions: f:{target_image.shape[1] - 1} "
            f"x:{target_image.shape[2]} y:{target_image.shape[3]}"
        )
    )

    # select neuron and plot for all feature maps (?)
    neuron_f = int(input("Please select neuron_f: "))
    neuron_x = target_image.shape[2] // 2
    neuron_y = target_image.shape[3] // 2
    print(f"Selected neuron {neuron_f}, {neuron_x}, {neuron_y}")
    target_image[0, neuron_f, neuron_x, neuron_y] = target_image_active

    # Input mask ->
    active_input_x = coordinate_list[-1][:, neuron_x].clone()
    active_input_y = coordinate_list[-1][:, neuron_y].clone()

    input_mask: torch.Tensor = torch.zeros_like(input_img)

    input_mask[
        :,
        :,
        active_input_x.type(torch.int64).unsqueeze(-1),
        active_input_y.type(torch.int64).unsqueeze(0),
    ] = 1

    rect_x = [int(active_input_x.min()), int(active_input_x.max())]
    rect_y = [int(active_input_y.min()), int(active_input_y.max())]
    # <- Input mask

    if apply_input_mask:
        with torch.no_grad():
            input_img *= input_mask


optimizer = torch.optim.Adam([{"params": input_parameter}], lr=learning_rate)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    patience=sheduler_patience,
    factor=sheduler_factor,
    eps=sheduler_eps * 0.1,
)


counter: int = 0
while (optimizer.param_groups[0]["lr"] > sheduler_eps) and (counter < num_iterations):
    optimizer.zero_grad()

    output = model(input_parameter)

    loss = torch.nn.functional.mse_loss(output, target_image)
    loss.backward()

    if counter % 1000 == 0:
        print(
            f"{counter} : loss={float(loss):.3e} lr={optimizer.param_groups[0]['lr']:.3e}"
        )

    optimizer.step()

    if apply_input_mask and len(target_image.shape) != 2:
        with torch.no_grad():
            input_parameter.data[torch.where(input_mask == 0)] = 0.0

    with torch.no_grad():
        max_data = torch.abs(input_parameter.data).max()
        if max_data > 1.0:
            input_parameter.data /= max_data

    if (
        torch.isfinite(input_parameter.data).sum().cpu()
        != torch.tensor(input_parameter.data.size()).prod()
    ):
        print(f"Found NaN in step: {counter}, use a smaller initial lr")
        exit()

    scheduler.step(float(loss))
    counter += 1

# save image
if save_final:
    # get short model name:
    matches = re.findall(pattern, nn)
    model_short = "".join(["".join(match) for match in matches])
    save_name = (
        f"optimal_model{model_short}_layer{idx}_feature{neuron_f}_seed{random_seed}.pt"
    )

    # filepath:
    folderpath = f"./other_{condition}_optimal"
    os.makedirs(folderpath, exist_ok=True)
    torch.save(input_img.squeeze().detach().cpu(), os.path.join(folderpath, save_name))

# plot image:
_, ax = plt.subplots()

ax.imshow(input_img.squeeze().detach().cpu().numpy(), cmap="gray")

plt.yticks(fontsize=15)
plt.xticks(fontsize=15)


if len(target_image.shape) != 2 and mark_region_in_plot:
    edgecolor = "sienna"
    kernel = patch.Rectangle(
        (rect_y[0], rect_x[0]),
        int(rect_y[1] - rect_y[0]),
        int(rect_x[1] - rect_x[0]),
        linewidth=1.2,
        edgecolor=edgecolor,
        facecolor="none",
    )
    ax.add_patch(kernel)

figure_path = f"./other_{condition}_optimal"
os.makedirs(figure_path, exist_ok=True)
plt.savefig(
    os.path.join(
        figure_path,
        f"{save_name}_{model_str}.pdf",
    ),
    dpi=300,
    bbox_inches="tight",
)

plt.show(block=True)
