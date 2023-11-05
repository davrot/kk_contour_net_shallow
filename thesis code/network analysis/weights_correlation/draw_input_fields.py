# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import matplotlib as mpl
from cycler import cycler
from functions.analyse_network import analyse_network

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

    # get current path:
    cwd = os.path.dirname(os.path.realpath(__file__)).replace(os.sep, "/")

    network_config_filename = f"{cwd}/network_0.json"
    config_filenname = f"{cwd}/config_v2.json"
    with open(config_filenname, "r") as file_handle:
        config = json.loads(jsmin(file_handle.read()))

    logger = create_logger(
        save_logging_messages=False, display_logging_messages=False, model_name=None
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

    # call function for plotting input fields into image:
    draw_kernel(
        image=image.numpy(),
        coordinate_list=coordinate_list,
        layer_type_list=layer_type_list,
        ignore_output_conv_layer=ignore_output_conv_layer,
    )
