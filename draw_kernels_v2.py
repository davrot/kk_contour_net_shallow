# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import matplotlib as mpl
import torch
from cycler import cycler

mpl.rcParams["text.usetex"] = True
mpl.rcParams["font.family"] = "serif"


def draw_kernel(
    image: np.ndarray,
    coordinate_list: list,
    layer_type_list: list,
    ignore_output_conv_layer: bool,
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

    # position first kernel
    start_x: int = 4
    start_y: int = 15

    # general plot structure:
    plt.ion()
    _, ax = plt.subplots()
    ax.imshow(image, cmap="gray")
    ax.tick_params(axis="both", which="major", labelsize=15)

    if ignore_output_conv_layer:
        number_of_layers: int = len(layer_type_list) - 1
    else:
        number_of_layers = len(layer_type_list)

    for i in range(0, number_of_layers):
        if layer_type_list[i] is not None:
            kernels = int(coordinate_list[i].shape[0])
            edgecolor = next(edge_color_cycler)["color"]
            # draw kernel
            kernel = patch.Rectangle(
                (start_x, start_y),
                kernels,
                kernels,
                linewidth=1.2,
                edgecolor=edgecolor,
                facecolor="none",
                label=layer_type_list[i],
            )
            ax.add_patch(kernel)

            if coordinate_list[i].shape[1] > 1:
                strides = int(coordinate_list[i][0, 1]) - int(coordinate_list[i][0, 0])

                # draw stride
                stride = patch.Rectangle(
                    (start_x + strides, start_y + strides),
                    kernels,
                    kernels,
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


def unfold(
    layer: torch.nn.Conv2d | torch.nn.MaxPool2d | torch.nn.AvgPool2d, size: int
) -> torch.Tensor:
    if isinstance(layer.kernel_size, tuple):
        assert layer.kernel_size[0] == layer.kernel_size[1]
        kernel_size: int = int(layer.kernel_size[0])
    else:
        kernel_size = int(layer.kernel_size)

    if isinstance(layer.dilation, tuple):
        assert layer.dilation[0] == layer.dilation[1]
        dilation: int = int(layer.dilation[0])
    else:
        dilation = int(layer.dilation)  # type: ignore

    if isinstance(layer.padding, tuple):
        assert layer.padding[0] == layer.padding[1]
        padding: int = int(layer.padding[0])
    else:
        padding = int(layer.padding)

    if isinstance(layer.stride, tuple):
        assert layer.stride[0] == layer.stride[1]
        stride: int = int(layer.stride[0])
    else:
        stride = int(layer.stride)

    out = (
        torch.nn.functional.unfold(
            torch.arange(0, size, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(-1),
            kernel_size=(kernel_size, 1),
            dilation=(dilation, 1),
            padding=(padding, 0),
            stride=(stride, 1),
        )
        .squeeze(0)
        .type(torch.int64)
    )

    return out


def analyse_network(
    model: torch.nn.Sequential, input_shape: int
) -> tuple[list, list, list]:
    combined_list: list = []
    coordinate_list: list = []
    layer_type_list: list = []
    pixel_used: list[int] = []

    size: int = int(input_shape)

    for layer_id in range(0, len(model)):
        if isinstance(
            model[layer_id], (torch.nn.Conv2d, torch.nn.MaxPool2d, torch.nn.AvgPool2d)
        ):
            out = unfold(layer=model[layer_id], size=size)
            coordinate_list.append(out)
            layer_type_list.append(
                str(type(model[layer_id])).split(".")[-1].split("'")[0]
            )
            size = int(out.shape[-1])
        else:
            coordinate_list.append(None)
            layer_type_list.append(None)

    assert coordinate_list[0] is not None
    combined_list.append(coordinate_list[0])

    for i in range(1, len(coordinate_list)):
        if coordinate_list[i] is None:
            combined_list.append(combined_list[i - 1])
        else:
            for pos in range(0, coordinate_list[i].shape[-1]):
                idx_shape: int | None = None

                idx = torch.unique(
                    torch.flatten(combined_list[i - 1][:, coordinate_list[i][:, pos]])
                )
                if idx_shape is None:
                    idx_shape = idx.shape[0]
                assert idx_shape == idx.shape[0]

            assert idx_shape is not None

            temp = torch.zeros((idx_shape, coordinate_list[i].shape[-1]))
            for pos in range(0, coordinate_list[i].shape[-1]):
                idx = torch.unique(
                    torch.flatten(combined_list[i - 1][:, coordinate_list[i][:, pos]])
                )
                temp[:, pos] = idx
            combined_list.append(temp)

    for i in range(0, len(combined_list)):
        pixel_used.append(int(torch.unique(torch.flatten(combined_list[i])).shape[0]))

    return combined_list, layer_type_list, pixel_used


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

    assert input_shape[-2] == input_shape[-1]
    coordinate_list, layer_type_list, pixel_used = analyse_network(
        model=model, input_shape=int(input_shape[-1])
    )

    for i in range(0, len(coordinate_list)):
        print(
            (
                f"Layer: {i}, Positions: {coordinate_list[i].shape[1]}, "
                f"Pixel per Positions: {coordinate_list[i].shape[0]}, "
                f"Type: {layer_type_list[i]}, Number of pixel used: {pixel_used[i]}"
            )
        )

    image = data_test.__getitem__(6)[1].squeeze(0)

    # call function:
    draw_kernel(
        image=image.numpy(),
        coordinate_list=coordinate_list,
        layer_type_list=layer_type_list,
        ignore_output_conv_layer=ignore_output_conv_layer,
    )
