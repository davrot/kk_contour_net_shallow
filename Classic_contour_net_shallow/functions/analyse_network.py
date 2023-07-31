import torch


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
