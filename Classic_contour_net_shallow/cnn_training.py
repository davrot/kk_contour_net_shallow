import torch
import numpy as np
import datetime
import argh
import time
import os
import json
from jsmin import jsmin

from functions.alicorn_data_loader import alicorn_data_loader
from functions.train import train
from functions.test import test
from functions.make_cnn import make_cnn
from functions.set_seed import set_seed
from functions.plot_intermediate import plot_intermediate
from functions.create_logger import create_logger


# to disable logging output from Tensorflow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from torch.utils.tensorboard import SummaryWriter


def main(
    idx_conv_out_channels_list: int = 0,
    idx_conv_kernel_sizes: int = 0,
    idx_conv_stride_sizes: int = 0,
    seed_counter: int = 0,
) -> None:
    config_filenname = "config.json"
    with open(config_filenname, "r") as file_handle:
        file_contents = file_handle.read()
        f_contents = jsmin(file_contents)
        print(f_contents)
        config = json.loads(f_contents)
        # config = json.loads(jsmin(file_handle.read()))

    # get model information:
    output_channels = config["conv_out_channels_list"][idx_conv_out_channels_list]

    logger = create_logger(
        save_logging_messages=bool(config["save_logging_messages"]),
        display_logging_messages=bool(config["display_logging_messages"]),
        model_name=str(output_channels),
    )

    # network settings:
    conv_out_channels_list: list[list[int]] = config["conv_out_channels_list"]
    conv_kernel_sizes: list[list[int]] = config["conv_kernel_sizes"]
    conv_stride_sizes: list[int] = config["conv_stride_sizes"]

    num_pfinkel: list = np.arange(
        int(config["num_pfinkel_start"]),
        int(config["num_pfinkel_stop"]),
        int(config["num_pfinkel_step"]),
    ).tolist()

    run_network(
        out_channels=conv_out_channels_list[int(idx_conv_out_channels_list)],
        kernel_size=conv_kernel_sizes[int(idx_conv_kernel_sizes)],
        stride=conv_stride_sizes[int(idx_conv_stride_sizes)],
        activation_function=str(config["activation_function"]),
        train_first_layer=bool(config["train_first_layer"]),
        seed_counter=seed_counter,
        minimum_learning_rate=float(config["minimum_learning_rate"]),
        conv_0_kernel_size=int(config["conv_0_kernel_size"]),
        mp_1_kernel_size=int(config["mp_1_kernel_size"]),
        mp_1_stride=int(config["mp_1_stride"]),
        batch_size_train=int(config["batch_size_train"]),
        batch_size_test=int(config["batch_size_test"]),
        learning_rate=float(config["learning_rate"]),
        max_epochs=int(config["max_epochs"]),
        save_model=bool(config["save_model"]),
        stimuli_per_pfinkel=int(config["stimuli_per_pfinkel"]),
        num_pfinkel=num_pfinkel,
        logger=logger,
        save_ever_x_epochs=int(config["save_ever_x_epochs"]),
        scheduler_patience=int(config["scheduler_patience"]),
        condition=str(config["condition"]),
        data_path=str(config["data_path"]),
        pooling_type=str(config["pooling_type"]),
        conv_0_enable_softmax=bool(config["conv_0_enable_softmax"]),
        scale_data=int(config["scale_data"]),
        use_scheduler=bool(config["use_scheduler"]),
        use_adam=bool(config["use_adam"]),
        use_plot_intermediate=bool(config["use_plot_intermediate"]),
        leak_relu_negative_slope=float(config["leak_relu_negative_slope"]),
        switch_leakyR_to_relu=bool(config["switch_leakyR_to_relu"]),
        scheduler_verbose=bool(config["scheduler_verbose"]),
        scheduler_factor=float(config["scheduler_factor"]),
        precision_100_percent=int(config["precision_100_percent"]),
        scheduler_threshold=float(config["scheduler_threshold"]),
    )


def run_network(
    out_channels: list[int],
    kernel_size: list[int],
    num_pfinkel: list,
    logger,
    stride: int,
    activation_function: str,
    train_first_layer: bool,
    seed_counter: int,
    minimum_learning_rate: float,
    conv_0_kernel_size: int,
    mp_1_kernel_size: int,
    mp_1_stride: int,
    scheduler_patience: int,
    batch_size_train: int,
    batch_size_test: int,
    learning_rate: float,
    max_epochs: int,
    save_model: bool,
    stimuli_per_pfinkel: int,
    save_ever_x_epochs: int,
    condition: str,
    data_path: str,
    pooling_type: str,
    conv_0_enable_softmax: bool,
    scale_data: float,
    use_scheduler: bool,
    use_adam: bool,
    use_plot_intermediate: bool,
    leak_relu_negative_slope: float,
    switch_leakyR_to_relu: bool,
    scheduler_verbose: bool,
    scheduler_factor: float,
    precision_100_percent: int,
    scheduler_threshold: float,
) -> None:
    # define device:
    device_str: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using {device_str} device")
    device: torch.device = torch.device(device_str)
    torch.set_default_dtype(torch.float32)

    # switch to relu if using leaky relu (not switched yet)
    switched_to_relu: bool = False

    # get initial leaky slope:
    leaky_slope = leak_relu_negative_slope

    # -------------------------------------------------------------------
    logger.info("-==- START -==-")

    train_accuracy: list[float] = []
    train_losses: list[float] = []
    train_loss: list[float] = []
    test_accuracy: list[float] = []
    test_losses: list[float] = []

    # prepare data:

    logger.info(num_pfinkel)
    logger.info(condition)

    logger.info("Loading training data")
    data_train = alicorn_data_loader(
        num_pfinkel=num_pfinkel,
        load_stimuli_per_pfinkel=stimuli_per_pfinkel,
        condition=condition,
        logger=logger,
        data_path=data_path,
    )

    logger.info("Loading test data")
    data_test = alicorn_data_loader(
        num_pfinkel=num_pfinkel,
        load_stimuli_per_pfinkel=stimuli_per_pfinkel,
        condition=condition,
        logger=logger,
        data_path=data_path,
    )

    logger.info("Loading done!")

    # data loader
    loader_train = torch.utils.data.DataLoader(
        data_train, shuffle=True, batch_size=batch_size_train
    )
    loader_test = torch.utils.data.DataLoader(
        data_test, shuffle=False, batch_size=batch_size_test
    )

    previous_test_acc: float = -1

    # set seed for reproducibility
    set_seed(seed=int(seed_counter), logger=logger)

    # number conv layer:
    if train_first_layer:
        num_conv_layers = len(out_channels)
    else:
        num_conv_layers = len(out_channels) if len(out_channels) >= 2 else 1

    # determine num conv layers
    model_name = (
        f"ArghCNN_numConvLayers{num_conv_layers}"
        f"_outChannels{out_channels}_kernelSize{kernel_size}_"
        f"{activation_function}_stride{stride}_"
        f"trainFirstConvLayer{train_first_layer}_"
        f"seed{seed_counter}_{condition}"
    )
    current = datetime.datetime.now().strftime("%d%m-%H%M")

    # new tb session
    os.makedirs("tb_runs", exist_ok=True)
    path: str = os.path.join("tb_runs", f"{model_name}")
    tb = SummaryWriter(path)

    # --------------------------------------------------------------------------

    # print network configuration:
    logger.info("----------------------------------------------------")
    logger.info(f"Number conv layers: {num_conv_layers}")
    logger.info(f"Output channels: {out_channels}")
    logger.info(f"Kernel sizes: {kernel_size}")
    logger.info(f"Stride: {stride}")
    logger.info(f"Activation function: {activation_function}")
    logger.info(f"Training conv 0: {train_first_layer}")
    logger.info(f"Seed: {seed_counter}")
    logger.info(f"LR-scheduler patience: {scheduler_patience}")
    logger.info(f"Pooling layer kernel: {mp_1_kernel_size}, stride: {mp_1_stride}")

    # define model:
    model = make_cnn(
        conv_out_channels_list=out_channels,
        conv_kernel_size=kernel_size,
        conv_stride_size=stride,
        conv_activation_function=activation_function,
        train_conv_0=train_first_layer,
        conv_0_kernel_size=conv_0_kernel_size,
        mp_1_kernel_size=mp_1_kernel_size,
        mp_1_stride=mp_1_stride,
        logger=logger,
        pooling_type=pooling_type,
        conv_0_enable_softmax=conv_0_enable_softmax,
        l_relu_negative_slope=leak_relu_negative_slope,
    ).to(device)

    logger.info(model)

    old_params: dict = {}
    for name, param in model.named_parameters():
        old_params[name] = param.data.detach().cpu().clone()

    # pararmeters for training:
    param_list: list = []

    for i in range(0, len(model)):
        if (not train_first_layer) and (i == 0):
            pass
        else:
            for name, param in model[i].named_parameters():
                logger.info(f"Learning parameter: layer: {i} name: {name}")
                param_list.append(param)

    for name, param in model.named_parameters():
        assert (
            torch.isfinite(param.data).sum().cpu()
            == torch.tensor(param.data.size()).prod()
        ), name

    # optimizer and learning rate scheduler
    if use_adam:
        optimizer = torch.optim.Adam(param_list, lr=learning_rate)
    else:
        optimizer = torch.optim.SGD(param_list, lr=learning_rate)  # type: ignore

    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=scheduler_patience,
            eps=minimum_learning_rate / 10,
            verbose=scheduler_verbose,
            factor=scheduler_factor,
            threshold=scheduler_threshold,
        )

    # training loop:
    logger.info("-==- Data and network loader: Done -==-")
    t_dis0 = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        # train
        logger.info("-==- Training... -==-")
        running_loss = train(
            model=model,
            loader=loader_train,
            optimizer=optimizer,
            epoch=epoch,
            device=device,
            tb=tb,
            test_acc=previous_test_acc,
            logger=logger,
            train_accuracy=train_accuracy,
            train_losses=train_losses,
            train_loss=train_loss,
            scale_data=scale_data,
        )

        # logging:
        logger.info("")

        logger.info("Check for changes in the weights:")
        for name, param in model.named_parameters():
            if isinstance(old_params[name], torch.Tensor) and isinstance(
                param.data, torch.Tensor
            ):
                temp_torch = param.data.detach().cpu().clone()
                if old_params[name].ndim == temp_torch.ndim:
                    if old_params[name].size() == temp_torch.size():
                        abs_diff = torch.abs(old_params[name] - temp_torch).max()
                        logger.info(f"Parameter {name}: {abs_diff:.3e}")

            old_params[name] = temp_torch

        logger.info("")

        logger.info("-==- Testing... -==-")
        previous_test_acc = test(  # type: ignore
            model=model,
            loader=loader_test,
            device=device,
            tb=tb,
            epoch=epoch,
            logger=logger,
            test_accuracy=test_accuracy,
            test_losses=test_losses,
            scale_data=scale_data,
        )

        logger.info(f"Time required: {time.perf_counter()-t_dis0:.2e} sec")

        # save model after every 100th epoch:
        if save_model and (epoch % save_ever_x_epochs == 0):
            pt_filename: str = f"{model_name}_{epoch}Epoch_{current}.pt"
            logger.info("")
            logger.info(f"Saved model: {pt_filename}")
            os.makedirs("trained_models", exist_ok=True)
            torch.save(
                model,
                os.path.join(
                    "trained_models",
                    pt_filename,
                ),
            )

        # check nan
        for name, param in model.named_parameters():
            assert (
                torch.isfinite(param.data).sum().cpu()
                == torch.tensor(param.data.size()).prod()
            ), name

        # update scheduler
        if use_scheduler:
            if scheduler_verbose and isinstance(scheduler.best, float):
                logger.info(
                    "Step LR scheduler: "
                    f"Loss: {running_loss:.2e} "
                    f"Best: {scheduler.best:.2e} "
                    f"Delta: {running_loss-scheduler.best:.2e} "
                    f"Threshold: {scheduler.threshold:.2e} "
                    f"Number of bad epochs: {scheduler.num_bad_epochs} "
                    f"Patience: {scheduler.patience} "
                )
            scheduler.step(running_loss)

        # stop learning: lr too small
        if optimizer.param_groups[0]["lr"] <= minimum_learning_rate:
            logger.info("Learning rate is too small. Stop training.")
            break

        # stop learning: done
        if round(previous_test_acc, precision_100_percent) == 100.0:
            if activation_function == "leaky relu":
                if switch_leakyR_to_relu and not switched_to_relu:
                    leaky_slope /= 10
                    logger.info(
                        f"100% test performance reached. Decreasing LeakyReLU slope to {leaky_slope}."
                    )
                    for name, module in model.named_children():
                        if isinstance(module, torch.nn.LeakyReLU):
                            module.negative_slope = leaky_slope
                    logger.info(model)

                    if leaky_slope <= 1e-5:
                        switched_to_relu = True
                        activation_function = "relu"
            else:
                logger.info("100% test performance reached. Stop training.")
                break

    if use_plot_intermediate:
        plot_intermediate(
            train_accuracy=train_accuracy,
            test_accuracy=test_accuracy,
            train_losses=train_losses,
            test_losses=test_losses,
            save_name=model_name,
        )

    os.makedirs("performance_data", exist_ok=True)
    np.savez(
        os.path.join("performance_data", f"performances_{model_name}.npz"),
        output_channels=np.array(out_channels),
        train_accuracy=np.array(train_accuracy),
        test_accuracy=np.array(test_accuracy),
        train_losses=np.array(train_losses),
        test_losses=np.array(test_losses),
    )

    # end TB session:
    tb.close()

    # print model name:
    logger.info("")
    logger.info(f"Saved model: {model_name}_{epoch}Epoch_{current}")
    if save_model:
        os.makedirs("trained_models", exist_ok=True)
        torch.save(
            model,
            os.path.join(
                "trained_models",
                f"{model_name}_{epoch}Epoch_{current}.pt",
            ),
        )


if __name__ == "__main__":
    argh.dispatch_command(main)
