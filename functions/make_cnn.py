import torch

# import numpy as np
from functions.SoftmaxPower import SoftmaxPower


def make_cnn(
    conv_out_channels_list: list[int],
    conv_kernel_size: list[int],
    conv_stride_size: int,
    conv_activation_function: str,
    train_conv_0: bool,
    logger,
    conv_0_kernel_size: int,
    mp_1_kernel_size: int,
    mp_1_stride: int,
    pooling_type: str,
    conv_0_enable_softmax: bool,
    conv_0_power_softmax: float,
    conv_0_meanmode_softmax: bool,
    conv_0_no_input_mode_softmax: bool,
    l_relu_negative_slope: float,
    input_shape: torch.Size,
) -> torch.nn.Sequential:
    assert len(conv_out_channels_list) >= 1
    assert len(conv_out_channels_list) == len(conv_kernel_size) + 1

    cnn = torch.nn.Sequential()

    temp_image: torch.Tensor = torch.zeros(
        (1, *input_shape), dtype=torch.float32, device=torch.device("cpu")
    )
    logger.info(
        (
            f"Input shape: {int(temp_image.shape[1])}, "
            f"{int(temp_image.shape[2])}, "
            f"{int(temp_image.shape[3])}"
        )
    )
    layer_counter: int = 0

    # Fixed structure
    cnn.append(
        torch.nn.Conv2d(
            in_channels=int(temp_image.shape[0]),
            out_channels=conv_out_channels_list[0] if train_conv_0 else 32,
            kernel_size=conv_0_kernel_size,
            stride=1,
            bias=train_conv_0,
        )
    )
    temp_image = cnn[layer_counter](temp_image)
    logger.info(
        (
            f"After layer {layer_counter}: {int(temp_image.shape[1])}, "
            f"{int(temp_image.shape[2])}, "
            f"{int(temp_image.shape[3])}"
        )
    )
    layer_counter += 1

    setting_understood: bool = False
    if conv_activation_function.upper() == str("relu").upper():
        cnn.append(torch.nn.ReLU())
        setting_understood = True
    elif conv_activation_function.upper() == str("leaky relu").upper():
        cnn.append(torch.nn.LeakyReLU(negative_slope=l_relu_negative_slope))
        setting_understood = True
    elif conv_activation_function.upper() == str("tanh").upper():
        cnn.append(torch.nn.Tanh())
        setting_understood = True
    elif conv_activation_function.upper() == str("none").upper():
        setting_understood = True
    assert setting_understood
    temp_image = cnn[layer_counter](temp_image)
    logger.info(
        (
            f"After layer {layer_counter}: {int(temp_image.shape[1])}, "
            f"{int(temp_image.shape[2])}, "
            f"{int(temp_image.shape[3])}"
        )
    )
    layer_counter += 1

    setting_understood = False
    if pooling_type.upper() == str("max").upper():
        cnn.append(torch.nn.MaxPool2d(kernel_size=mp_1_kernel_size, stride=mp_1_stride))
        setting_understood = True
    elif pooling_type.upper() == str("average").upper():
        cnn.append(torch.nn.AvgPool2d(kernel_size=mp_1_kernel_size, stride=mp_1_stride))
        setting_understood = True
    elif pooling_type.upper() == str("none").upper():
        setting_understood = True
    assert setting_understood
    temp_image = cnn[layer_counter](temp_image)
    logger.info(
        (
            f"After layer {layer_counter}: {int(temp_image.shape[1])}, "
            f"{int(temp_image.shape[2])}, "
            f"{int(temp_image.shape[3])}"
        )
    )
    layer_counter += 1

    if conv_0_enable_softmax:
        cnn.append(
            SoftmaxPower(
                dim=1,
                power=conv_0_power_softmax,
                mean_mode=conv_0_meanmode_softmax,
                no_input_mode=conv_0_no_input_mode_softmax,
            )
        )
        temp_image = cnn[layer_counter](temp_image)
        logger.info(
            (
                f"After layer {layer_counter}: {int(temp_image.shape[1])}, "
                f"{int(temp_image.shape[2])}, "
                f"{int(temp_image.shape[3])}"
            )
        )
        layer_counter += 1

    # Changing structure
    for i in range(1, len(conv_out_channels_list)):
        if i == 1 and not train_conv_0:
            in_channels = 32
        else:
            in_channels = conv_out_channels_list[i - 1]
        cnn.append(
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=conv_out_channels_list[i],
                kernel_size=conv_kernel_size[i - 1],
                stride=conv_stride_size,
                bias=True,
            )
        )
        temp_image = cnn[layer_counter](temp_image)
        logger.info(
            (
                f"After layer {layer_counter}: {int(temp_image.shape[1])}, "
                f"{int(temp_image.shape[2])}, "
                f"{int(temp_image.shape[3])}"
            )
        )
        layer_counter += 1

        setting_understood = False
        if conv_activation_function.upper() == str("relu").upper():
            cnn.append(torch.nn.ReLU())
            setting_understood = True
        elif conv_activation_function.upper() == str("leaky relu").upper():
            cnn.append(torch.nn.LeakyReLU(negative_slope=l_relu_negative_slope))
            setting_understood = True
        elif conv_activation_function.upper() == str("tanh").upper():
            cnn.append(torch.nn.Tanh())
            setting_understood = True
        elif conv_activation_function.upper() == str("none").upper():
            setting_understood = True

        assert setting_understood
        temp_image = cnn[layer_counter](temp_image)
        logger.info(
            (
                f"After layer {layer_counter}: {int(temp_image.shape[1])}, "
                f"{int(temp_image.shape[2])}, "
                f"{int(temp_image.shape[3])}"
            )
        )
        layer_counter += 1

    # Output layer
    cnn.append(
        torch.nn.Conv2d(
            in_channels=int(temp_image.shape[1]),
            out_channels=2,
            kernel_size=(int(temp_image.shape[2]), int(temp_image.shape[3])),
            stride=1,
            bias=True,
        )
    )
    temp_image = cnn[layer_counter](temp_image)
    logger.info(
        (
            f"After layer {layer_counter}: {int(temp_image.shape[1])}, "
            f"{int(temp_image.shape[2])}, "
            f"{int(temp_image.shape[3])}"
        )
    )
    layer_counter += 1

    # Need to repair loading data
    assert train_conv_0 is True

    # # if conv1 not trained:
    # filename_load_weight_0: str | None = None
    # if train_conv_0 is False and cnn[0]._parameters["weight"].shape[0] == 32:
    #     filename_load_weight_0 = "weights_radius10.npy"
    # if train_conv_0 is False and cnn[0]._parameters["weight"].shape[0] == 16:
    #     filename_load_weight_0 = "8orient_2phase_weights.npy"

    # if filename_load_weight_0 is not None:
    #     logger.info(f"Replace weights in CNN 0 with {filename_load_weight_0}")
    #     cnn[0]._parameters["weight"] = torch.tensor(
    #         np.load(filename_load_weight_0),
    #         dtype=cnn[0]._parameters["weight"].dtype,
    #         requires_grad=False,
    #         device=cnn[0]._parameters["weight"].device,
    #     )

    return cnn
