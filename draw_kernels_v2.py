# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import matplotlib as mpl
import torch
from cycler import cycler

mpl.rcParams["text.usetex"] = True
mpl.rcParams["font.family"] = "serif"


def extract_kernel_stride(model: torch.nn.Sequential) -> list[dict]:
    result = []
    for idx, m in enumerate(model.modules()):
        if isinstance(m, (torch.nn.Conv2d, torch.nn.MaxPool2d)):
            result.append(
                {
                    "layer_index": idx,
                    "layer_type": type(m).__name__,
                    "kernel_size": m.kernel_size,
                    "stride": m.stride,
                }
            )
    return result


def calculate_kernel_size(
    kernel: np.ndarray, stride: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    df: np.ndarray = np.cumprod(
        (
            np.concatenate(
                (np.array(1)[np.newaxis], stride.astype(dtype=np.int64)[:-1]), axis=0
            )
        )
    )
    f = 1 + np.cumsum((kernel.astype(dtype=np.int64) - 1) * df)

    print(f"Receptive field sizes: {f} ")
    return f, df


def draw_kernel(
    image: np.ndarray, model: torch.nn.Sequential, ignore_output_conv_layer: bool
) -> None:
    """
    Call function after creating the model-to-be-trained.
    """
    assert image.shape[0] == 200
    assert image.shape[1] == 200

    # list of colors to choose from:
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    edge_color_cycler = iter(
        cycler(color=["sienna", "orange", "gold", "bisque"] + colors)
    )
    kernel_sizes: list[int] = []
    stride_sizes: list[int] = []
    layer_type: list[str] = []

    # extract kernel and stride information
    model_info: list[dict] = extract_kernel_stride(model)

    # iterate over kernels to plot on image
    for layer in model_info:
        kernel_sizes.append(layer["kernel_size"])
        stride_sizes.append(layer["stride"])
        layer_type.append(layer["layer_type"])

    # change tuples to list items:
    kernel_array: np.ndarray = np.array([k[0] if isinstance(k, tuple) else k for k in kernel_sizes])  # type: ignore
    stride_array: np.ndarray = np.array([s[0] if isinstance(s, tuple) else s for s in stride_sizes])  # type: ignore

    # calculate values
    kernels, strides = calculate_kernel_size(kernel_array, stride_array)

    # position first kernel
    start_x: int = 4
    start_y: int = 15

    # general plot structure:
    plt.ion()
    _, ax = plt.subplots()
    ax.imshow(image, cmap="gray")
    ax.tick_params(axis="both", which="major", labelsize=15)

    if ignore_output_conv_layer:
        number_of_layers: int = len(kernels) - 1
    else:
        number_of_layers = len(kernels)

    for i in range(0, number_of_layers):
        edgecolor = next(edge_color_cycler)["color"]
        # draw kernel
        kernel = patch.Rectangle(
            (start_x, start_y),
            kernels[i],
            kernels[i],
            linewidth=1.2,
            edgecolor=edgecolor,
            facecolor="none",
            label=layer_type[i],
        )
        ax.add_patch(kernel)

        # draw stride
        stride = patch.Rectangle(
            (start_x + strides[i], start_y + strides[i]),
            kernels[i],
            kernels[i],
            linewidth=1.2,
            edgecolor=edgecolor,
            facecolor="none",
            linestyle="dashed",
        )
        ax.add_patch(stride)

        # add distance of next drawing
        start_x += 14
        start_y += 10

    # final plot
    plt.tight_layout()
    plt.legend(loc="upper right", fontsize=11)
    plt.show(block=True)


# %%
if __name__ == "__main__":
    import os
    import sys
    import json
    from jsmin import jsmin

    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(parent_dir)
    from functions.alicorn_data_loader import alicorn_data_loader
    from functions.make_cnn_v2 import make_cnn
    from functions.create_logger import create_logger

    ignore_output_conv_layer: bool = True
    network_config_filename = "network_0.json"
    config_filenname = "config_v2.json"
    with open(config_filenname, "r") as file_handle:
        config = json.loads(jsmin(file_handle.read()))

    logger = create_logger(
        save_logging_messages=False,
        display_logging_messages=False,
    )

    # test image:
    data_test = alicorn_data_loader(
        num_pfinkel=[0],
        load_stimuli_per_pfinkel=10,
        condition=str(config["condition"]),
        data_path=str(config["data_path"]),
        logger=logger,
    )

    assert data_test.__len__() > 0
    input_shape = data_test.__getitem__(0)[1].shape

    model = make_cnn(
        network_config_filename=network_config_filename,
        logger=logger,
        input_shape=input_shape,
    )
    print(model)

    # test_image = torch.zeros((1, *input_shape), dtype=torch.float32)

    image = data_test.__getitem__(6)[1].squeeze(0)

    # call function:
    draw_kernel(
        image=image.numpy(),
        model=model,
        ignore_output_conv_layer=ignore_output_conv_layer,
    )
