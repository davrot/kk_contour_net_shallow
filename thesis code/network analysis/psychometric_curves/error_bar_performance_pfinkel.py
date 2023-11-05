import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import datetime

# import re
# import glob
# from natsort import natsorted

mpl.rcParams["text.usetex"] = True
mpl.rcParams["font.family"] = "serif"

from functions.alicorn_data_loader import alicorn_data_loader
from functions.create_logger import create_logger


def performance_pfinkel_plot(
    performances_list: list[dict],
    all_performances: dict,
    labels: list[str],
    save_name: str,
    logger,
) -> None:
    figure_path: str = "rerun_errorbar_performance_pfinkel"
    os.makedirs(figure_path, exist_ok=True)

    plt.figure(figsize=[10, 10])
    with open(f"./{figure_path}/performances_{save_name}_{current}.txt", "w") as f:
        for id, selected_condition in enumerate(condition):
            f.write(
                f"Condition:{selected_condition}  Path angle (in °), Mean accuracy (\\%), Standard deviation (\\%)\n"
            )

            x_values = np.array(num_pfinkel)
            y_values = np.array(
                [
                    np.mean(all_performances[selected_condition][pfinkel])
                    for pfinkel in num_pfinkel
                ]
            )
            yerr_values = np.array(
                [
                    np.std(all_performances[selected_condition][pfinkel])
                    for pfinkel in num_pfinkel
                ]
            )

            for x, y, yerr in zip(x_values, y_values, yerr_values):
                f.write(f"{x}, {y/100.0:.3f}, {yerr/100.0:.3f}\n")
                f.write(f"{x}, {y}, {yerr}\n")

            plt.errorbar(
                x_values,
                y_values / 100.0,
                yerr=yerr_values / 100.0,
                fmt="o",
                capsize=5,
                label=labels[id],
            )
    plt.xticks(x_values)
    plt.title("Average accuracy", fontsize=19)
    plt.xlabel("Path angle (in °)", fontsize=18)
    plt.ylabel("Accuracy (\\%)", fontsize=18)
    plt.ylim(0.5, 1.0)
    plt.legend(fontsize=15)

    # Increase tick label font size
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.grid(True)
    plt.tight_layout()
    logger.info("")
    logger.info("Saved in:")

    print(
        os.path.join(
            figure_path,
            f"ylim_ErrorBarPerformancePfinkel_{save_name}_{current}.pdf",
        )
    )
    plt.savefig(
        os.path.join(
            figure_path,
            f"ylim_ErrorBarPerformancePfinkel_{save_name}_{current}.pdf",
        ),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


if __name__ == "__main__":
    model_path: str = "classic_3288_fest"
    print(model_path)
    data_path: str = "/home/kk/Documents/Semester4/code/RenderStimuli/Output/"

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
        model_name=model_path,
    )

    device_str: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using {device_str} device")
    device: torch.device = torch.device(device_str)
    torch.set_default_dtype(torch.float32)

    # current time:
    current = datetime.datetime.now().strftime("%d%m-%H%M")

    all_performances: dict = {
        condition_name: {pfinkel: [] for pfinkel in num_pfinkel}
        for condition_name in condition
    }

    for filename in os.listdir(model_path):
        if filename.endswith(".pt"):
            model_filename = os.path.join(model_path, filename)
            model = torch.load(model_filename, map_location=device)
            model.eval()
            print(model_filename)

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
                    logger.info(
                        f"-==- Start {selected_condition} " f"Pfinkel {pfinkel}°  -==-"
                    )
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
                    all_performances[selected_condition][pfinkel].append(
                        100 * correct / pattern_count
                    )

                performances_list.append(performances)
        else:
            print("No files found!")
            break

    performance_pfinkel_plot(
        performances_list=performances_list,
        all_performances=all_performances,
        labels=figure_label,
        save_name=model_path,
        logger=logger,
    )
    logger.info("-==- DONE -==-")
