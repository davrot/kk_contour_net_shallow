import torch


def calc_free_params(from_loaded_model: bool, model_name: str | None):
    """
    * Calculates the number of free parameters of a CNN
    * either from trained model or by entering the respective parameters
      over command line
    """

    if from_loaded_model:
        # path to NN
        PATH = f"D:/Katha/Neuroscience/Semester 4/newCode/kk_contour_net_shallow-main/trained_models/{model_name}"

        # load and evaluate model
        model = torch.load(PATH).to("cpu")
        model.eval()
        print(model)

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total number of free parameters: {total_params}")
    else:
        print("\n##########################")
        input_out_channel_size = input(
            "Enter output channel size (comma seperated, including output layer): "
        )
        out_channel_size = [1] + [int(x) for x in input_out_channel_size.split(",")]

        input_kernel_sizes = input(
            "Enter kernel sizes of respective layers (comma seperated, including output layer): "
        )
        kernel_sizes = [int(x) for x in input_kernel_sizes.split(",")]

        total_params = 0
        for i in range(1, len(out_channel_size)):
            input_size = out_channel_size[i - 1]
            out_size = out_channel_size[i]
            kernel = kernel_sizes[i - 1]
            bias = out_channel_size[i]
            num_free_params = input_size * kernel * kernel * out_size + bias
            total_params += num_free_params
        print(f"Total number of free parameters: {total_params}")


if __name__ == "__main__":
    # model name
    nn = "ArghCNN_numConvLayers3_outChannels[8, 8, 8]_kernelSize[7, 15]_leaky relu_stride1_trainFirstConvLayerTrue_seed291857_Natural_1351Epoch_3107-2121.pt"

    calc_free_params(from_loaded_model=False, model_name=nn)
