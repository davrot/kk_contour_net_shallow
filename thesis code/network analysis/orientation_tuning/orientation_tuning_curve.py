import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

mpl.rcParams["text.usetex"] = True
mpl.rcParams["font.family"] = "serif"

import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from functions.make_cnn import make_cnn  # noqa


def plot_single_tuning_curve(mean_syn_input, mean_relu_response, theta):
    plt.figure()
    plt.plot(theta, mean_syn_input, label="Before ReLU")
    plt.plot(theta, mean_relu_response, label="After ReLU")
    plt.xlabel("orientation (degs)")
    plt.ylabel("activity")
    plt.legend()
    plt.grid(True)
    plt.gca().set_xticks(theta)

    plt.show()


def plot_single_phase_orientation(syn_input, relu_response, theta, phi, j):
    plt.figure()
    plt.subplot(1, 2, 1, aspect="equal")
    plt.imshow(
        syn_input.T,  # type: ignore
        cmap="viridis",
        aspect="auto",
        extent=[theta[0], theta[-1], phi[0], phi[-1]],
    )
    plt.xlabel("orientation (degs)")
    plt.ylabel("phase (degs)")
    plt.colorbar(label="activity")
    plt.title(f"Weight {j}", fontsize=16)

    plt.subplot(1, 2, 2, aspect="equal")
    plt.imshow(
        relu_response.T,  # type: ignore
        cmap="viridis",
        aspect="auto",
        extent=[theta[0], theta[-1], phi[0], phi[-1]],
    )
    plt.xlabel("orientation (degs)")
    plt.ylabel("phase (degs)")
    plt.colorbar(label="activity")
    plt.title(f"Weight {j}", fontsize=16)

    plt.show()


def plot_all_tuning_curves(response_array, theta, orientations=32, phases=4):
    # plot tuning curves
    plt.figure(figsize=(12, 15))
    for i in range(response_array.shape[0]):
        # synaptic input
        in_neuron = response_array[i].reshape(orientations, phases)
        mean_syn_in = in_neuron.mean(axis=1)

        # after non linearity
        out_relu = torch.nn.functional.leaky_relu(torch.tensor(response_array[i]))
        out_relu = out_relu.numpy().reshape(orientations, phases)
        mean_out_relu = out_relu.mean(axis=1)  # type: ignore

        plt.subplot(8, 4, i + 1)
        plt.plot(theta, mean_syn_in)
        plt.plot(theta, mean_out_relu)
        plt.xlabel("Theta (degs)")
        plt.ylabel("Activity")

    plt.tight_layout()
    plt.show()


def calculate_responses(weights, plot_single_responses):
    # load Gabor filter
    orientations = 32
    phases = 4
    filename: str = "gabor_dict_32o_4p.npy"
    filepath: str = os.path.join(
        "D:/Katha/Neuroscience/Semester 4/newCode/kk_contour_net_shallow-main/investigate",
        filename,
    )
    gabor_dict = np.load(filepath)

    # collect data
    all_responses: list = []
    after_relu = np.zeros((weights.shape[0], orientations))
    for j in range(weights.shape[0]):
        w0 = weights[j, 0].detach().cpu()  # .numpy()

        response: list = []
        for i in range(gabor_dict.shape[0]):
            gabor = gabor_dict[i, 0]
            if w0.shape[0] != gabor.shape[0]:
                # TODO: for later layers
                # get number to pad
                pad = (gabor.shape[0] - w0.shape[0]) // 2

                # pad:
                w_pad = torch.nn.functional.pad(
                    w0, (pad, pad, pad, pad), mode="constant", value=0
                )
                w_pad = w_pad.numpy()

            else:
                w_pad = w0.numpy()

            dot = np.sum(gabor * w_pad)
            response.append(dot)

        # axis for plotting:
        theta = np.rad2deg(np.arange(orientations) * np.pi / orientations)
        phi = np.rad2deg(np.arange(phases) * 2 * np.pi / phases)

        # to array + mean
        syn_input = np.array(response)
        syn_input = syn_input.reshape(orientations, phases)
        mean_response_orient = syn_input.mean(axis=1)

        # leaky relu:
        relu_response = torch.nn.functional.leaky_relu(
            torch.tensor(response), negative_slope=0.1
        )
        relu_response = relu_response.numpy().reshape(orientations, phases)
        mean_relu_orient = relu_response.mean(axis=1)  # type: ignore

        # append to save:
        after_relu[j] = mean_relu_orient

        # plot 2D:
        if plot_single_responses:
            plot_single_phase_orientation(
                syn_input=syn_input,
                relu_response=relu_response,
                theta=theta,
                phi=phi,
                j=j,
            )

            # plot tuning curve
            plot_single_tuning_curve(
                mean_syn_input=mean_response_orient,
                mean_relu_response=mean_relu_orient,
                theta=theta,
            )

        # collect response for each weight
        all_responses.append(response)

    # to array:
    response_array = np.array(all_responses)

    return response_array, after_relu, theta


def plot_mean_resp_after_relu(mean_response, theta):
    # plot tuning curves
    plt.figure(figsize=(12, 15))
    for i in range(mean_response.shape[0]):
        plt.subplot(8, 4, i + 1)
        plt.plot(theta, mean_response[i])
        plt.xlabel("Theta (degs)")
        plt.ylabel("Activity")

    plt.tight_layout()
    plt.show()


def load_data_from_cnn(
    cnn_name: str,
    plot_responses: bool,
    do_stats: bool,
    plot_single_responses: bool = False,
):
    # path to NN

    if do_stats:
        PATH = cnn_name
    else:
        PATH = f"D:/Katha/Neuroscience/Semester 4/newCode/kk_contour_net_shallow-main/trained_models/{cnn_name}"

    # load and evaluate model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(PATH).to(device)

    # Set the model to evaluation mode
    model.eval()
    print(model)

    # load NNs conv1 weights:
    weights = model[0]._parameters["weight"].data

    # call
    response_array, mean_response_after_relu, theta = calculate_responses(
        weights=weights, plot_single_responses=plot_single_responses
    )

    # plot
    if plot_responses:
        plot_all_tuning_curves(response_array=response_array, theta=theta)
        plot_mean_resp_after_relu(mean_response=mean_response_after_relu, theta=theta)

    return np.array(mean_response_after_relu)


if __name__ == "__main__":
    # path to NN
    nn = "ArghCNN_numConvLayers3_outChannels[32, 8, 8]_kernelSize[7, 15]_leaky relu_stride1_trainFirstConvLayerTrue_seed291853_Natural_314Epoch_0908-1206.pt"
    _ = load_data_from_cnn(cnn_name=nn, plot_responses=True, do_stats=False)
    exit()

    PATH = f"D:/Katha/Neuroscience/Semester 4/newCode/kk_contour_net_shallow-main/trained_models/{nn}"

    # load and evaluate model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(PATH).to(device)

    # Set the model to evaluation mode
    model.eval()
    print(model)

    # load NNs conv1 weights:
    weights = model[0]._parameters["weight"].data

    # plot?
    plot_single_responses: bool = False

    # call
    response_array, mean_response_after_relu, theta = calculate_responses(
        weights=weights, plot_single_responses=plot_single_responses
    )

    # plot
    plot_all_tuning_curves(response_array=response_array, theta=theta)
    plot_mean_resp_after_relu(mean_response=mean_response_after_relu, theta=theta)
    print()
