import torch
import numpy as np


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
    l_relu_negative_slope: float,
) -> torch.nn.Sequential:
    assert len(conv_out_channels_list) >= 1
    assert len(conv_out_channels_list) == len(conv_kernel_size) + 1

    cnn = torch.nn.Sequential()

    # Fixed structure
    cnn.append(
        torch.nn.Conv2d(
            in_channels=1,
            out_channels=conv_out_channels_list[0] if train_conv_0 else 32,
            kernel_size=conv_0_kernel_size,
            stride=1,
            bias=train_conv_0,
        )
    )

    if conv_0_enable_softmax:
        cnn.append(torch.nn.Softmax(dim=1))

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

    # Fixed structure
    # define fully connected layer:
    cnn.append(torch.nn.Flatten(start_dim=1))
    cnn.append(torch.nn.LazyLinear(2, bias=True))

    # if conv1 not trained:
    filename_load_weight_0: str | None = None
    if train_conv_0 is False and cnn[0]._parameters["weight"].shape[0] == 32:
        filename_load_weight_0 = "weights_radius10_norm.npy"
    if train_conv_0 is False and cnn[0]._parameters["weight"].shape[0] == 16:
        filename_load_weight_0 = "8orient_2phase_weights.npy"

    if filename_load_weight_0 is not None:
        logger.info(f"Replace weights in CNN 0 with {filename_load_weight_0}")
        cnn[0]._parameters["weight"] = torch.tensor(
            np.load(filename_load_weight_0),
            dtype=cnn[0]._parameters["weight"].dtype,
            requires_grad=False,
            device=cnn[0]._parameters["weight"].device,
        )

    return cnn
