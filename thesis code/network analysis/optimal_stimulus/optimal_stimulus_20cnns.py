# %%
import torch
import numpy as np
import random
import re
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import matplotlib as mpl

mpl.rcParams["text.usetex"] = True
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.size"] = 15

import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from functions.analyse_network import analyse_network
from functions.set_seed import set_seed

# set seet
random_seed = random.randint(0, 100)
set_seed(random_seed)
print(f"Random seed: {random_seed}")


def get_file_list_all_cnns(dir: str) -> list:
    all_results: list = []
    for filename in os.listdir(dir):
        if filename.endswith(".pt"):
            print(os.path.join(dir, filename))
            all_results.append(os.path.join(dir, filename))

    return all_results


def show_single_optimal_stimulus(model_list, save: bool = False, cnn: str = "CORNER"):
    first_run: bool = True
    chosen_layer_idx: int
    chosen_neuron_f_idx: int
    chosen_neuron_x_idx: int
    chosen_neuron_y_idx: int
    mean_opt_stim_list: list = []
    fig, axs = plt.subplots(4, 5, figsize=(15, 15))
    for i, load_model in enumerate(model_list):
        print(f"\nModel: {i} ")
        num_iterations: int = 100000
        learning_rate: float = 0.1
        apply_input_mask: bool = True
        mark_region_in_plot: bool = True
        sheduler_patience: int = 500
        sheduler_factor: float = 0.9
        sheduler_eps = 1e-08
        target_image_active: float = 1e4
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # load model
        model = torch.load(load_model).to(device)
        model.eval()

        if first_run:
            print("Full network:")
            print(model)
            print("")

            # enter index to plot:
            idx = int(input("Please select layer: "))
            assert idx < len(model)
            chosen_layer_idx = idx

        print(f"Selected layer: {model[chosen_layer_idx]}")
        model = model[: chosen_layer_idx + 1]

        # prepare random input image
        input_img = torch.rand(1, 200, 200).to(device)
        input_img = input_img.unsqueeze(0)
        input_img.requires_grad_(True)  # type: ignore

        input_shape = input_img.shape
        assert input_shape[-2] == input_shape[-1]
        coordinate_list, layer_type_list, pixel_used = analyse_network(
            model=model, input_shape=int(input_shape[-1])
        )

        output_shape = model(input_img).shape

        target_image = torch.zeros(
            (*output_shape,), dtype=input_img.dtype, device=input_img.device
        )

        # image to parameter (2B optimized)
        input_parameter = torch.nn.Parameter(input_img)

        # back to first run:
        if first_run:
            if len(target_image.shape) == 2:
                print((f"Available max positions: f:{target_image.shape[1] - 1} "))

                # select neuron and plot for all feature maps (?)
                neuron_f = int(input("Please select neuron_f: "))
                print(f"Selected neuron {neuron_f}")
                chosen_neuron_f_idx = neuron_f
            else:
                print(
                    (
                        f"Available max positions: f:{target_image.shape[1] - 1} "
                        f"x:{target_image.shape[2]} y:{target_image.shape[3]}"
                    )
                )

                # select neuron and plot for all feature maps (?)
                neuron_f = int(input("Please select neuron_f: "))
                neuron_x = target_image.shape[2] // 2
                neuron_y = target_image.shape[3] // 2
                chosen_neuron_f_idx = neuron_f
                chosen_neuron_x_idx = neuron_x
                chosen_neuron_y_idx = neuron_y
                print(
                    f"Selected neuron {chosen_neuron_f_idx}, {chosen_neuron_x_idx}, {chosen_neuron_y_idx}"
                )

            # keep settings for further runs:
            first_run = False

        # keep input values for all cnns
        if len(target_image.shape) == 2:
            target_image[0, chosen_neuron_f_idx] = 1e4
        else:
            target_image[
                0, chosen_neuron_f_idx, chosen_neuron_x_idx, chosen_neuron_y_idx
            ] = target_image_active

            # Input mask ->
            active_input_x = coordinate_list[-1][:, neuron_x].clone()
            active_input_y = coordinate_list[-1][:, neuron_y].clone()

            input_mask: torch.Tensor = torch.zeros_like(input_img)

            input_mask[
                :,
                :,
                active_input_x.type(torch.int64).unsqueeze(-1),
                active_input_y.type(torch.int64).unsqueeze(0),
            ] = 1

            rect_x = [int(active_input_x.min()), int(active_input_x.max())]
            rect_y = [int(active_input_y.min()), int(active_input_y.max())]
            # <- Input mask

            if apply_input_mask:
                with torch.no_grad():
                    input_img *= input_mask

        # start optimization:
        optimizer = torch.optim.Adam([{"params": input_parameter}], lr=learning_rate)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=sheduler_patience,
            factor=sheduler_factor,
            eps=sheduler_eps * 0.1,
        )

        counter: int = 0
        while (optimizer.param_groups[0]["lr"] > sheduler_eps) and (
            counter < num_iterations
        ):
            optimizer.zero_grad()

            output = model(input_parameter)

            loss = torch.nn.functional.mse_loss(output, target_image)
            loss.backward()

            if counter % 1000 == 0:
                print(
                    f"{counter} : loss={float(loss):.3e} lr={optimizer.param_groups[0]['lr']:.3e}"
                )

            optimizer.step()

            if apply_input_mask and len(target_image.shape) != 2:
                with torch.no_grad():
                    input_parameter.data[torch.where(input_mask == 0)] = 0.0

            with torch.no_grad():
                max_data = torch.abs(input_parameter.data).max()
                if max_data > 1.0:
                    input_parameter.data /= max_data

            if (
                torch.isfinite(input_parameter.data).sum().cpu()
                != torch.tensor(input_parameter.data.size()).prod()
            ):
                print(f"Found NaN in step: {counter}, use a smaller initial lr")
                exit()

            scheduler.step(float(loss))
            counter += 1
        mean_opt_stim_list.append(input_img.squeeze().detach().cpu().numpy())

        # plot image:
        ax = axs[i // 5, i % 5]
        im = ax.imshow(input_img.squeeze().detach().cpu().numpy(), cmap="gray")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"Model {i+1}", fontsize=13)
        cbar.ax.tick_params(labelsize=12)

        if len(target_image.shape) != 2 and mark_region_in_plot:
            edgecolor = "sienna"
            kernel = patch.Rectangle(
                (rect_y[0], rect_x[0]),
                int(rect_y[1] - rect_y[0]),
                int(rect_x[1] - rect_x[0]),
                linewidth=1.2,
                edgecolor=edgecolor,
                facecolor="none",
            )
            ax.add_patch(kernel)

    plt.tight_layout()
    # save image
    if save:
        save_name = f"single_optimal_stimulus_{cnn}_layer{chosen_layer_idx}_feature{chosen_neuron_f_idx}"
        folderpath = "./all20_optimal_stimuli"
        os.makedirs(folderpath, exist_ok=True)
        torch.save(
            input_img.squeeze().detach().cpu(),
            os.path.join(folderpath, save_name) + ".pt",
        )
        plt.savefig(
            f"{os.path.join(folderpath, save_name)}.pdf",
            dpi=300,
            bbox_inches="tight",
        )

    plt.show(block=True)

    if len(target_image.shape) == 2:
        return mean_opt_stim_list, chosen_neuron_f_idx, chosen_layer_idx
    else:
        return (
            mean_opt_stim_list,
            (chosen_layer_idx, chosen_neuron_f_idx),
            (chosen_neuron_x_idx, chosen_neuron_y_idx),
        )


def plot_mean_optimal_stimulus(
    overall_optimal_stimuli,
    chosen_layer_idx: int,
    chosen_neuron_f_idx: int,
    save: bool = False,
    cnn: str = "CORNER",
):
    fig, axs = plt.subplots(figsize=(15, 15))
    mean_optimal_stimulus = np.mean(overall_optimal_stimuli, axis=0)
    im = axs.imshow(mean_optimal_stimulus, cmap="gray")
    cbar = fig.colorbar(im, ax=axs, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=15)

    plt.tight_layout()
    # save image
    if save:
        save_name = f"overall_mean_optimal_stimulus_{cnn}_layer{chosen_layer_idx}_feature{chosen_neuron_f_idx}"
        folderpath = "./mean_optimal_stimulus"
        os.makedirs(folderpath, exist_ok=True)
        torch.save(mean_optimal_stimulus, os.path.join(folderpath, save_name) + ".pt")
        plt.savefig(
            f"{os.path.join(folderpath, save_name)}.pdf",
            dpi=300,
        )

    plt.show(block=True)


if __name__ == "__main__":
    # path to NN
    PATH_corner = "./classic_3288_fest"
    all_cnns_corner = get_file_list_all_cnns(dir=PATH_corner)
    opt_stim_list, feature_idx, layer_idx = show_single_optimal_stimulus(
        all_cnns_corner, save=True, cnn="CLASSIC_3288_fest"
    )

    # average optimal stimulus:
    # plot_mean_optimal_stimulus(
    #     opt_stim_list,
    #     save=True,
    #     cnn="CORNER_3288_fest",
    #     chosen_layer_idx=layer_idx,
    #     chosen_neuron_f_idx=feature_idx,
    # )
