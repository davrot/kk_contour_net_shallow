import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import datetime
import re

# import glob
# from natsort import natsorted

mpl.rcParams["text.usetex"] = True
mpl.rcParams["font.family"] = "serif"

from functions.alicorn_data_loader import alicorn_data_loader
from functions.create_logger import create_logger


def sort_and_plot(
    extracted_params,
    save: bool,
    plot_for_each_condition: bool,
    name: str,
    sort_by="params",
):
    figure_path: str = "performance_pfinkel_0210"
    os.makedirs(figure_path, exist_ok=True)

    architecture_params = extracted_params.copy()
    if sort_by == "params":
        architecture_params.sort(key=lambda x: x[1])
    elif sort_by == "accuracy":
        architecture_params.sort(key=lambda x: x[-1])

    sorted_architectures, sorted_params, test_conditions, sorted_performances = zip(
        *architecture_params
    )
    final_labels = [
        f"{arch[1:-1]} - {params}"
        for arch, params in zip(sorted_architectures, sorted_params)
    ]

    plt.figure(figsize=(18, 9))

    # performance for each condition
    if plot_for_each_condition:
        conditions = ["Coignless", "Natural", "Angular"]
        labels = ["Classic", "Corner", "Bridge"]
        shift_amounts = [-0.05, 0, 0.05]
        save_name = name + "_each_condition"
        for i, condition in enumerate(conditions):
            # x_vals = range(len(sorted_performances))
            jittered_x = np.arange(len(sorted_performances)) + shift_amounts[i]
            y_vals = [perf[condition] for perf in test_conditions]
            plt.errorbar(
                jittered_x,
                y_vals,
                fmt="D",
                markerfacecolor="none",
                markeredgewidth=1.5,
                label=labels[i],
            )
    else:
        save_name = name + "_mean"
        plt.plot(range(len(sorted_performances)), sorted_performances, marker="o")

    plt.ylabel("Accuracy (in \\%)", fontsize=17)
    plt.xticks(range(len(sorted_performances)), final_labels, rotation=90, fontsize=15)
    plt.yticks(fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.legend(fontsize=15)

    if save:
        plt.savefig(
            os.path.join(
                figure_path,
                f"minimalCNN_64sorted_{sort_by}_{save_name}.pdf",
            ),
            dpi=300,
            bbox_inches="tight",
        )
    plt.show()


if __name__ == "__main__":
    training_con: str = "classic"
    model_path: str = "./trained_classic"
    print(model_path)
    data_path: str = "/home/kk/Documents/Semester4/code/RenderStimuli/Output/"

    # num stimuli per Pfinkel and batch size
    stim_per_pfinkel: int = 10000
    batch_size: int = 1000

    # stimulus condition:
    performances_list: list = []
    condition: list[str] = ["Coignless", "Natural", "Angular"]

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

    # save data
    cnn_data: list = []
    cnn_counter: int = 0

    for filename in os.listdir(model_path):
        if filename.endswith(".pt"):
            model_filename = os.path.join(model_path, filename)
            model = torch.load(model_filename, map_location=device)
            model.eval()
            print(f"CNN {cnn_counter+1} :{model_filename}")

            # number free parameters for current CNN
            num_free_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )

            # save
            all_performances: dict = {
                condition_name: {pfinkel: [] for pfinkel in num_pfinkel}
                for condition_name in condition
            }

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

            # store num free params + performances
            avg_performance_per_condition = {
                cond: np.mean([np.mean(perfs) for perfs in pfinkel_dict.values()])
                for cond, pfinkel_dict in all_performances.items()
            }
            avg_performance_overall = np.mean(
                list(avg_performance_per_condition.values())
            )

            # extract CNN config:
            match = re.search(r"_outChannels\[(\d+), (\d+), (\d+)\]_", filename)
            if match:
                out_channels = (
                    [1] + [int(match.group(i)) for i in range(1, 3 + 1)] + [2]
                )

            # number of free parameters and performances
            cnn_data.append(
                (
                    out_channels,
                    num_free_params,
                    avg_performance_per_condition,
                    avg_performance_overall,
                )
            )

        else:
            print("No files found!")
            break

    # save all 64 performances
    torch.save(
        cnn_data,
        f"{model_path}.pt",
    )

    # plot
    sort_and_plot(
        cnn_data,
        save=True,
        plot_for_each_condition=True,
        name=training_con,
        sort_by="params",
    )
    sort_and_plot(
        cnn_data,
        save=True,
        plot_for_each_condition=False,
        name=training_con,
        sort_by="params",
    )
    sort_and_plot(
        cnn_data,
        save=True,
        plot_for_each_condition=True,
        name=training_con,
        sort_by="accuracy",
    )
    sort_and_plot(
        cnn_data,
        save=True,
        plot_for_each_condition=False,
        name=training_con,
        sort_by="accuracy",
    )

    logger.info("-==- DONE -==-")
