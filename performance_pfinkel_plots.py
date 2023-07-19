import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import datetime
import re
import glob
from natsort import natsorted

mpl.rcParams["text.usetex"] = True
mpl.rcParams["font.family"] = "serif"

from functions.alicorn_data_loader import alicorn_data_loader
from functions.create_logger import create_logger


def performance_pfinkel_plot(
    performances_list: list[dict], labels: list[str], save_name: str, logger
) -> None:
    figure_path: str = "performance_pfinkel"
    assert len(performances_list) == len(labels)

    plt.figure(figsize=[14, 10])
    # plot accuracy
    plt.subplot(2, 1, 1)
    for id in range(0, len(labels)):
        x_values = np.zeros((len(performances_list[id].keys())))
        y_values = np.zeros((len(performances_list[id].keys())))

        counter = 0
        for id_key in performances_list[id].keys():
            x_values[counter] = performances_list[id][id_key]["pfinkel"]
            y_values[counter] = performances_list[id][id_key]["test_accuracy"]
            counter += 1

        plt.plot(x_values, y_values, label=labels[id])
    plt.xticks(x_values)
    plt.title("Average accuracy", fontsize=18)
    plt.xlabel("Path angle (in °)", fontsize=17)
    plt.ylabel("Accuracy (\\%)", fontsize=17)
    plt.legend(fontsize=14)

    # Increase tick label font size
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True)

    # plot loss
    plt.subplot(2, 1, 2)
    for id in range(0, len(labels)):
        x_values = np.zeros((len(performances_list[id].keys())))
        y_values = np.zeros((len(performances_list[id].keys())))

        counter = 0
        for id_key in performances_list[id].keys():
            x_values[counter] = performances_list[id][id_key]["pfinkel"]
            y_values[counter] = performances_list[id][id_key]["test_losses"]
            counter += 1

        plt.plot(x_values, y_values, label=labels[id])

    plt.xticks(x_values)
    plt.title("Average loss", fontsize=18)
    plt.xlabel("Path angle (in °)", fontsize=17)
    plt.ylabel("Loss", fontsize=17)
    plt.legend(fontsize=14)

    # Increase tick label font size
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True)

    plt.tight_layout()
    logger.info("")
    logger.info("Saved in:")

    os.makedirs(figure_path, exist_ok=True)
    print(
        os.path.join(
            figure_path,
            f"PerformancePfinkel_{save_name}_{current}.pdf",
        )
    )
    plt.savefig(
        os.path.join(
            figure_path,
            f"PerformancePfinkel_{save_name}_{current}.pdf",
        ),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


if __name__ == "__main__":
    model_path: str = "trained_models"
    data_path: str = "/home/kk/Documents/Semester4/code/RenderStimuli/Output/"
    selection_file_id: int = 0

    # num stimuli per Pfinkel and batch size
    stim_per_pfinkel: int = 10000
    batch_size: int = 1000
    # stimulus condition:
    performances_list: list = []
    condition: list[str] = ["Coignless", "Natural", "Angular"]
    figure_label: list[str] = ["Classic", "Corner", "Bridge"]
    # load test data:
    num_pfinkel: list = np.arange(0, 100, 10).tolist()
    image_scale: float = 255.0

    # ------------------------------------------

    # create logger:
    logger = create_logger(
        save_logging_messages=False,
        display_logging_messages=True,
    )

    device_str: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using {device_str} device")
    device: torch.device = torch.device(device_str)
    torch.set_default_dtype(torch.float32)

    # current time:
    current = datetime.datetime.now().strftime("%d%m-%H%M")

    # path to NN
    list_filenames: list[str] = natsorted(
        list(glob.glob(os.path.join(model_path, "*.pt")))
    )
    assert selection_file_id < len(list_filenames)
    model_filename: str = str(list_filenames[selection_file_id])
    logger.info(f"Using model file: {model_filename}")

    # shorter saving name:
    pattern = r"(outChannels\[.*?\])|(kernelSize\[.*?\])|(_relu)|(_seed\d+)"
    matches = re.findall(pattern, model_filename)
    save_name = "".join(["".join(match) for match in matches])

    # load and evaluate model
    model = torch.load(model_filename, map_location=device)

    # Set the model to evaluation mode
    model.eval()

    for selected_condition in condition:
        # save performances:
        logger.info(f"Condition: {selected_condition}")
        performances: dict = {}
        for pfinkel in num_pfinkel:
            test_loss: float = 0.0
            correct: int = 0
            pattern_count: int = 0

            data_test = alicorn_data_loader(
                num_pfinkel=[pfinkel],
                load_stimuli_per_pfinkel=stim_per_pfinkel,
                condition=selected_condition,
                logger=logger,
                data_path=data_path,
            )
            loader = torch.utils.data.DataLoader(
                data_test, shuffle=False, batch_size=batch_size
            )

            # start testing network on new stimuli:
            logger.info("")
            logger.info(f"-==- Start {selected_condition} " f"Pfinkel {pfinkel}°  -==-")
            with torch.no_grad():
                for batch_num, data in enumerate(loader):
                    label = data[0].to(device)
                    image = data[1].type(dtype=torch.float32).to(device)
                    image /= image_scale

                    # compute prediction error;
                    output = model(image)

                    # Label Typecast:
                    label = label.to(device)

                    # loss and optimization
                    loss = torch.nn.functional.cross_entropy(
                        output, label, reduction="sum"
                    )
                    pattern_count += int(label.shape[0])
                    test_loss += float(loss)
                    prediction = output.argmax(dim=1)
                    correct += prediction.eq(label).sum().item()

                    total_number_of_pattern: int = int(len(loader)) * int(
                        label.shape[0]
                    )

                    # logging:
                    logger.info(
                        (
                            f"{selected_condition},{pfinkel}° "
                            "Pfinkel: "
                            f"[{int(pattern_count)}/{total_number_of_pattern} ({100.0 * pattern_count / total_number_of_pattern:.2f}%)],"
                            f" Average loss: {test_loss / pattern_count:.3e}, "
                            "Accuracy: "
                            f"{100.0 * correct / pattern_count:.2f}% "
                        )
                    )

            performances[pfinkel] = {
                "pfinkel": pfinkel,
                "test_accuracy": 100 * correct / pattern_count,
                "test_losses": float(loss) / pattern_count,
            }

        performances_list.append(performances)

    performance_pfinkel_plot(
        performances_list=performances_list,
        labels=figure_label,
        save_name=save_name,
        logger=logger,
    )
    logger.info("-==- DONE -==-")
