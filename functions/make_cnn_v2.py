import torch
import os
import numpy as np
import json
from jsmin import jsmin

from functions.SoftmaxPower import SoftmaxPower


def make_cnn(
    network_config_filename: str,
    logger,
    input_shape: torch.Size,
) -> torch.nn.Sequential:
    with open(network_config_filename, "r") as file_handle:
        network_config = json.loads(jsmin(file_handle.read()))

    # Convolution layer
    conv_out_channel: list[int] = network_config["conv_out_channel"]
    conv_kernel_size: list[int] = network_config["conv_kernel_size"]
    conv_stride_size: list[int] = network_config["conv_stride_size"]
    conv_bias: list[bool] = network_config["conv_bias"]
    conv_padding: list[int | str] = network_config["conv_padding"]

    # Activation function
    activation_function: str = str(network_config["activation_function"])
    l_relu_negative_slope: float = float(network_config["l_relu_negative_slope"])

    # Pooling layer
    pooling_kernel_size: list[int] = network_config["pooling_kernel_size"]
    pooling_stride: list[int] = network_config["pooling_stride"]
    pooling_type: str = str(network_config["pooling_type"])

    # Softmax layer
    softmax_enable: list[bool] = network_config["softmax_enable"]
    softmax_power: float = network_config["softmax_power"]
    softmax_meanmode: bool = network_config["softmax_meanmode"]
    softmax_no_input_mode: bool = network_config["softmax_no_input_mode"]

    # Load pre-trained weights and biases
    path_pretrained_weights_bias: str = str(
        network_config["path_pretrained_weights_bias"]
    )
    train_weights: list[bool] = network_config["train_weights"]
    train_bias: list[bool] = network_config["train_bias"]

    # Output layer
    number_of_classes: int = int(network_config["number_of_classes"])

    assert len(conv_out_channel) == len(conv_kernel_size)
    assert len(conv_out_channel) == len(conv_stride_size)
    assert len(conv_out_channel) + 1 == len(conv_bias)
    assert len(conv_out_channel) == len(conv_padding)
    assert len(conv_out_channel) == len(pooling_kernel_size)
    assert len(conv_out_channel) == len(pooling_stride)
    assert len(conv_out_channel) + 1 == len(train_weights)
    assert len(conv_out_channel) + 1 == len(train_bias)
    assert len(conv_out_channel) == len(softmax_enable)
    assert number_of_classes > 1

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

    # Changing structure
    for i in range(0, len(conv_out_channel)):
        # Conv Layer
        if i == 0:
            in_channels: int = int(temp_image.shape[0])
        else:
            in_channels = conv_out_channel[i - 1]
        cnn.append(
            torch.nn.Conv2d(
                in_channels=int(in_channels),
                out_channels=int(conv_out_channel[i]),
                kernel_size=int(conv_kernel_size[i]),
                stride=int(conv_stride_size[i]),
                bias=bool(conv_bias[i]),
                padding=conv_padding[i],
            )
        )

        # Load bias
        if bool(conv_bias[i]) and bool(train_bias[i]) is False:
            filename_load = os.path.join(
                path_pretrained_weights_bias, str(f"bias_{i}.npy")
            )
            logger.info(f"Replace bias in conv2d {i} with {filename_load}")
            temp = np.load(filename_load)
            assert torch.equal(
                torch.tensor(temp.shape, dtype=torch.int),
                torch.tensor(cnn[-1]._parameters["bias"].data.shape, dtype=torch.int),
            )
            cnn[-1]._parameters["bias"] = torch.tensor(
                temp,
                dtype=cnn[-1]._parameters["bias"].dtype,
                requires_grad=False,
                device=cnn[-1]._parameters["bias"].device,
            )

        # Load weights
        if train_weights[i] is False:
            filename_load = os.path.join(
                path_pretrained_weights_bias, str(f"weights_{i}.npy")
            )
            logger.info(f"Replace weights in conv2d {i} with {filename_load}")
            temp = np.load(filename_load)
            assert torch.equal(
                torch.tensor(temp.shape, dtype=torch.int),
                torch.tensor(cnn[-1]._parameters["weight"].data.shape, dtype=torch.int),
            )
            cnn[-1]._parameters["weight"] = torch.tensor(
                temp,
                dtype=cnn[-1]._parameters["weight"].dtype,
                requires_grad=False,
                device=cnn[-1]._parameters["weight"].device,
            )

        cnn[-1].train_bias = True
        if bool(conv_bias[i]) and bool(train_bias[i]) is False:
            cnn[-1].train_bias = False

        cnn[-1].train_weights = train_weights[i]

        temp_image = cnn[layer_counter](temp_image)
        logger.info(
            (
                f"After layer {layer_counter} (Convolution 2D layer): {int(temp_image.shape[1])}, "
                f"{int(temp_image.shape[2])}, "
                f"{int(temp_image.shape[3])}, "
                f"train bias: {cnn[-1].train_bias}, "
                f"train weights: {cnn[-1].train_weights}, "
            )
        )
        layer_counter += 1

        # Activation Function
        setting_understood = False
        if activation_function.upper() == str("relu").upper():
            cnn.append(torch.nn.ReLU())
            setting_understood = True
        elif activation_function.upper() == str("leaky relu").upper():
            cnn.append(torch.nn.LeakyReLU(negative_slope=l_relu_negative_slope))
            setting_understood = True
        elif activation_function.upper() == str("tanh").upper():
            cnn.append(torch.nn.Tanh())
            setting_understood = True
        elif activation_function.upper() == str("none").upper():
            setting_understood = True

        assert setting_understood

        cnn[-1].train_bias = False
        cnn[-1].train_weights = False
        temp_image = cnn[layer_counter](temp_image)
        logger.info(
            (
                f"After layer {layer_counter} (Activation function): {int(temp_image.shape[1])}, "
                f"{int(temp_image.shape[2])}, "
                f"{int(temp_image.shape[3])}, "
                f"train bias: {cnn[-1].train_bias}, "
                f"train weights: {cnn[-1].train_weights} "
            )
        )
        layer_counter += 1

        if softmax_enable[i]:
            cnn.append(
                SoftmaxPower(
                    dim=1,
                    power=float(softmax_power),
                    mean_mode=bool(softmax_meanmode),
                    no_input_mode=bool(softmax_no_input_mode),
                )
            )

            cnn[-1].train_bias = False
            cnn[-1].train_weights = False

            temp_image = cnn[layer_counter](temp_image)
            logger.info(
                (
                    f"After layer {layer_counter} (Softmax Power Layer): {int(temp_image.shape[1])}, "
                    f"{int(temp_image.shape[2])}, "
                    f"{int(temp_image.shape[3])}, "
                    f"train bias: {cnn[-1].train_bias}, "
                    f"train weights: {cnn[-1].train_weights} "
                )
            )
            layer_counter += 1

        if (pooling_kernel_size[i] > 0) and (pooling_stride[i] > 0):
            setting_understood = False
            if pooling_type.upper() == str("max").upper():
                cnn.append(
                    torch.nn.MaxPool2d(
                        kernel_size=pooling_kernel_size[i], stride=pooling_stride[i]
                    )
                )
                setting_understood = True
            elif pooling_type.upper() == str("average").upper():
                cnn.append(
                    torch.nn.AvgPool2d(
                        kernel_size=pooling_kernel_size[i], stride=pooling_stride[i]
                    )
                )
                setting_understood = True
            elif pooling_type.upper() == str("none").upper():
                setting_understood = True
            assert setting_understood

            cnn[-1].train_bias = False
            cnn[-1].train_weights = False

            temp_image = cnn[layer_counter](temp_image)
            logger.info(
                (
                    f"After layer {layer_counter} (Pooling layer): {int(temp_image.shape[1])}, "
                    f"{int(temp_image.shape[2])}, "
                    f"{int(temp_image.shape[3])}, "
                    f"train bias: {cnn[-1].train_bias}, "
                    f"train weights: {cnn[-1].train_weights} "
                )
            )
            layer_counter += 1


    # Output layer
    cnn.append(
        torch.nn.Conv2d(
            in_channels=int(temp_image.shape[1]),
            out_channels=int(number_of_classes),
            kernel_size=(int(temp_image.shape[2]), int(temp_image.shape[3])),
            stride=1,
            bias=bool(conv_bias[-1]),
        )
    )

    # Load bias
    if bool(conv_bias[-1]) and bool(train_bias[-1]) is False:
        filename_load = os.path.join(
            path_pretrained_weights_bias, str(f"bias_{len(conv_out_channel)}.npy")
        )
        logger.info(
            f"Replace bias in conv2d {len(conv_out_channel)} with {filename_load}"
        )
        temp = np.load(filename_load)
        assert torch.equal(
            torch.Tensor(temp.shape),
            torch.Tensor(cnn[-1]._parameters["bias"].shape),
        )
        cnn[-1]._parameters["bias"] = torch.tensor(
            temp,
            dtype=cnn[-1]._parameters["bias"].dtype,
            requires_grad=False,
            device=cnn[-1]._parameters["bias"].device,
        )

    # Load weights
    if train_weights[-1] is False:
        filename_load = os.path.join(
            path_pretrained_weights_bias, str(f"weights_{len(conv_out_channel)}.npy")
        )
        logger.info(
            f"Replace weights in conv2d {len(conv_out_channel)} with {filename_load}"
        )
        temp = np.load(filename_load)
        assert torch.equal(
            torch.Tensor(temp.shape),
            torch.Tensor(cnn[-1]._parameters["weight"].shape),
        )
        cnn[-1]._parameters["weight"] = torch.tensor(
            temp,
            dtype=cnn[-1]._parameters["weight"].dtype,
            requires_grad=False,
            device=cnn[-1]._parameters["weight"].device,
        )

    cnn[-1].train_bias = True
    if bool(conv_bias[i]) and bool(train_bias[i]) is False:
        cnn[-1].train_bias = False

    cnn[-1].train_weights = train_weights[i]

    temp_image = cnn[layer_counter](temp_image)
    logger.info(
        (
            f"After layer {layer_counter} (Output layer): {int(temp_image.shape[1])}, "
            f"{int(temp_image.shape[2])}, "
            f"{int(temp_image.shape[3])}, "
            f"train bias: {cnn[-1].train_bias}, "
            f"train weights: {cnn[-1].train_weights} "
        )
    )
    layer_counter += 1

    # Output layer needs to have a x and y dim of 1
    assert int(temp_image.shape[2]) == 1
    assert int(temp_image.shape[3]) == 1

    return cnn
