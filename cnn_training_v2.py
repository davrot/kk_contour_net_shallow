import torch
import numpy as np
import datetime
import argh
import time
import os
import json
import glob
from jsmin import jsmin
from natsort import natsorted

from functions.alicorn_data_loader import alicorn_data_loader
from functions.train import train
from functions.test import test
from functions.make_cnn_v2 import make_cnn
from functions.set_seed import set_seed
from functions.plot_intermediate import plot_intermediate
from functions.create_logger import create_logger


# to disable logging output from Tensorflow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from torch.utils.tensorboard import SummaryWriter


def main(
    network_config_filename: str = "network_0.json",
    seed_counter: int = 0,
) -> None:
    config_filenname = "config_v2.json"
    with open(config_filenname, "r") as file_handle:
        config = json.loads(jsmin(file_handle.read()))

    logger = create_logger(
        save_logging_messages=bool(config["save_logging_messages"]),
        display_logging_messages=bool(config["display_logging_messages"]),
    )

    # network settings:

    num_pfinkel: list = np.arange(
        int(config["num_pfinkel_start"]),
        int(config["num_pfinkel_stop"]),
        int(config["num_pfinkel_step"]),
    ).tolist()

    run_network(
        seed_counter=seed_counter,
        minimum_learning_rate=float(config["minimum_learning_rate"]),
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
        scale_data=int(config["scale_data"]),
        use_scheduler=bool(config["use_scheduler"]),
        use_adam=bool(config["use_adam"]),
        use_plot_intermediate=bool(config["use_plot_intermediate"]),
        scheduler_verbose=bool(config["scheduler_verbose"]),
        scheduler_factor=float(config["scheduler_factor"]),
        precision_100_percent=int(config["precision_100_percent"]),
        scheduler_threshold=float(config["scheduler_threshold"]),
        model_continue=bool(config["model_continue"]),
        initial_model_path=str(config["initial_model_path"]),
        tb_runs_path=str(config["tb_runs_path"]),
        trained_models_path=str(config["trained_models_path"]),
        performance_data_path=str(config["performance_data_path"]),
        network_config_filename=network_config_filename,
    )


def run_network(
    num_pfinkel: list,
    logger,
    seed_counter: int,
    minimum_learning_rate: float,
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
    scale_data: float,
    use_scheduler: bool,
    use_adam: bool,
    use_plot_intermediate: bool,
    scheduler_verbose: bool,
    scheduler_factor: float,
    precision_100_percent: int,
    scheduler_threshold: float,
    model_continue: bool,
    initial_model_path: str,
    tb_runs_path: str,
    trained_models_path: str,
    performance_data_path: str,
    network_config_filename: str,
) -> None:
    # define device:
    device_str: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using {device_str} device")
    device: torch.device = torch.device(device_str)
    torch.set_default_dtype(torch.float32)

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
    assert data_train.__len__() > 0
    input_shape = data_train.__getitem__(0)[1].shape

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

    # determine num conv layers
    model_name = (
        f"{str(network_config_filename).replace('.json','')}_"
        f"seed{seed_counter}_{condition}"
    )
    current = datetime.datetime.now().strftime("%d%m-%H%M")

    # new tb session
    os.makedirs(tb_runs_path, exist_ok=True)
    path: str = os.path.join(tb_runs_path, f"{model_name}")
    tb = SummaryWriter(path)

    # --------------------------------------------------------------------------

    # print network configuration:
    logger.info("----------------------------------------------------")
    logger.info(f"Seed: {seed_counter}")

    # define model:
    if model_continue:
        filename_list: list = natsorted(
            glob.glob(os.path.join(initial_model_path, str("*.pt")))
        )
        assert len(filename_list) > 0
        model_filename: str = filename_list[-1]
        logger.info(f"Load filename: {model_filename}")
        model = torch.load(model_filename, map_location=device)
    else:
        model = make_cnn(
            network_config_filename=network_config_filename,
            logger=logger,
            input_shape=input_shape,
        ).to(device)
    logger.info("----------------------------------------------------")
    logger.info(model)
    logger.info("----------------------------------------------------")
    old_params: dict = {}
    for name, param in model.named_parameters():
        old_params[name] = param.data.detach().cpu().clone()

    # pararmeters for training:
    param_list: list = []

    for i in range(0, len(model)):
        if model[i].train_bias or model[i].train_weights:
            for name, param in model[i].named_parameters():
                if (name == "weight") and model[i].train_weights:
                    logger.info(f"Learning parameter: layer: {i} name: {name}")
                    param_list.append(param)

                if (name == "bias") and model[i].train_bias:
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

            os.makedirs(trained_models_path, exist_ok=True)
            torch.save(
                model,
                os.path.join(
                    trained_models_path,
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

    os.makedirs(performance_data_path, exist_ok=True)
    np.savez(
        os.path.join(performance_data_path, f"performances_{model_name}.npz"),
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
        os.makedirs(trained_models_path, exist_ok=True)
        torch.save(
            model,
            os.path.join(
                trained_models_path,
                f"{model_name}_{epoch}Epoch_{current}.pt",
            ),
        )


if __name__ == "__main__":
    argh.dispatch_command(main)
