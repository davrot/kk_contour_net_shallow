# %%
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.optimize as sop
import orientation_tuning_curve  # import load_data_from_cnn
import warnings
import matplotlib as mpl

mpl.rcParams["text.usetex"] = True
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.size"] = 15

# suppress warnings
warnings.filterwarnings("ignore")


def mises(orientation, a, mean, variance):
    k = 1 / variance**2
    return a * np.exp(k * np.cos(orientation - mean)) / np.exp(k)


def biemlich_mieses_karma(orientation, a1, mean1, variance1, a2, mean2, variance2):
    m1 = mises(orientation, a1, mean1, variance1)
    m2 = mises(orientation, a2, mean2, variance2)
    return m1 + m2


def triemlich_mieses_karma(
    orientation, a1, mean1, variance1, a2, mean2, variance2, a3, mean3, variance3
):
    m1 = mises(orientation, a1, mean1, variance1)
    m2 = mises(orientation, a2, mean2, variance2)
    m3 = mises(orientation, a3, mean3, variance3)
    return m1 + m2 + m3


def plot_reshaped(tune, fits, theta, save_name: str | None, save_plot: bool = False):
    """
    Plot shows the original tuning with the best fits
    """

    num_rows = tune.shape[0] // 4
    num_cols = tune.shape[0] // num_rows
    # plt.figure(figsize=(12, 15))
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 7))

    # plot the respective y-lims:
    overall_min = np.min(tune)
    overall_max = np.max(tune)

    for i_tune in range(tune.shape[0]):
        ax = axs[i_tune // num_cols, i_tune % num_cols]
        ax.plot(np.rad2deg(theta), tune[i_tune], label="Original")

        x_center = (np.rad2deg(theta).min() + np.rad2deg(theta).max()) / 2
        y_center = (tune[i_tune].min() + tune[i_tune].max()) / 2

        fit = next((fit for key, fit in fits if key == i_tune))
        if fit["fitted_curve"] is not None:
            ax.plot(
                np.rad2deg(theta),
                fit["fitted_curve"] * fit["scaling_factor"],
                label="Fit",
            )
            ax.text(
                x_center,
                y_center,
                str(fit["curve"]),
                ha="center",
                va="center",
                size="xx-large",
                color="gray",
            )

            # update again if there's a fit
            overall_min = min(
                overall_min, (fit["fitted_curve"] * fit["scaling_factor"]).min()
            )
            overall_max = max(
                overall_max, (fit["fitted_curve"] * fit["scaling_factor"]).max()
            )
        else:
            # plt.plot(np.rad2deg(theta), fit[i_tune], "--")
            ax.text(
                x_center,
                y_center,
                "*",
                ha="center",
                va="center",
                size="xx-large",
                color="gray",
            )
        # specified y lims: of no fit: min and max of tune
        ax.set_ylim([overall_min, overall_max + 0.05])

        # x-ticks from 0°-360°
        ax.set_xticks(range(0, 361, 90))

        # label them from 0° to  180°
        ax.set_xticklabels(range(0, 181, 45), fontsize=15)
        ax.set_xlabel("(in deg)", fontsize=16)

    plt.yticks(fontsize=15)

    plt.tight_layout()
    if save_plot:
        plt.savefig(
            f"additional thesis plots/saved_plots/fitkarotte/{save_name}.pdf",
            dpi=300,
            bbox_inches="tight",
        )

    plt.show(block=True)


def plot_fit(tune, fits, theta, save_name: str | None, save_plot: bool = False):
    """
    Plot shows the original tuning with the best fits
    """

    if tune.shape[0] >= 8:
        num_rows = tune.shape[0] // 8
        num_cols = tune.shape[0] // num_rows
    else:
        num_rows = 2
        num_cols = tune.shape[0] // num_rows
    # plt.figure(figsize=(12, 15))
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 7))

    # plot the respective y-lims:
    overall_min = np.min(tune)
    overall_max = np.max(tune)

    for i_tune in range(tune.shape[0]):
        if axs.ndim == 1:
            ax = axs[i_tune]
        else:
            ax = axs[i_tune // num_cols, i_tune % num_cols]
        ax.plot(np.rad2deg(theta), tune[i_tune], label="Original")

        x_center = (np.rad2deg(theta).min() + np.rad2deg(theta).max()) / 2
        y_center = (tune[i_tune].min() + tune[i_tune].max()) / 2

        # fit = next((fit for key, fit in fits if key == i_tune), None)
        fit = next((fit for key, fit in fits if key == i_tune))
        if fit["fitted_curve"] is not None:
            ax.plot(
                np.rad2deg(theta),
                fit["fitted_curve"] * fit["scaling_factor"],
                label="Fit",
            )
            ax.text(
                x_center,
                y_center,
                str(fit["curve"]),
                ha="center",
                va="center",
                size="xx-large",
                color="gray",
            )

            # update again if there's a fit
            overall_min = min(
                overall_min, (fit["fitted_curve"] * fit["scaling_factor"]).min()
            )
            overall_max = max(
                overall_max, (fit["fitted_curve"] * fit["scaling_factor"]).max()
            )
        else:
            ax.text(
                x_center,
                y_center,
                "*",
                ha="center",
                va="center",
                size="xx-large",
                color="gray",
            )
        # specified y lims: of no fit: min and max of tune
        ax.set_ylim([overall_min, overall_max + 0.05])

        # x-ticks from 0°-360°
        ax.set_xticks(range(0, 361, 90))

        # label them from 0° to  180°
        ax.set_xticklabels(range(0, 181, 45), fontsize=15)
        ax.set_xlabel("(in deg)", fontsize=16)

    plt.yticks(fontsize=15)

    plt.tight_layout()
    if save_plot:
        plt.savefig(
            f"additional thesis plots/saved_plots/fitkarotte/{save_name}.pdf", dpi=300
        )

    plt.show(block=True)


def fit_curves(tune, theta):
    # save all fits:
    save_fits: list = []
    scaling_factor: list = []
    for curve in range(1, 4):
        fit_possible: int = 0
        fit_impossible: int = 0
        for i_tune in range(tune.shape[0]):
            to_tune = tune[i_tune].copy()
            scale_fact = np.max(to_tune)
            scaling_factor.append(scale_fact)
            to_tune /= scale_fact

            p10 = theta[np.argmax(to_tune)]
            a10 = 1
            s10 = 0.5

            if curve == 1:
                function = mises
                p0 = [a10, p10, s10]
            elif curve == 2:
                function = biemlich_mieses_karma  # type: ignore
                p20 = p10 + np.pi
                a20 = 1.0
                s20 = 0.4
                p0 = [a10, p10, s10, a20, p20, s20]
            else:
                function = triemlich_mieses_karma  # type: ignore
                p20 = p10 + 2 * np.pi / 3
                a20 = 0.7
                s20 = 0.3
                p30 = p10 + 4 * np.pi / 3
                a30 = 0.4
                s30 = 0.3
                p0 = [a10, p10, s10, a20, p20, s20, a30, p30, s30]

            try:
                popt = sop.curve_fit(function, theta, to_tune, p0=p0)
                fitted_curve = function(theta, *popt[0])
                quad_dist = np.sum((to_tune - fitted_curve) ** 2)

                save_fits.append(
                    {
                        "weight_idx": i_tune,
                        "curve": curve,
                        "fit_params": popt[0],
                        "fitted_curve": fitted_curve,
                        "quad_dist": quad_dist,
                        "scaling_factor": scale_fact,
                    }
                )

                # count:
                fit_possible += 1
            except:
                fit_impossible += 1
                fitted_curve = function(theta, *p0)
                quad_dist = np.sum((to_tune - fitted_curve) ** 2)
                save_fits.append(
                    {
                        "weight_idx": i_tune,
                        "curve": curve,
                        "fit_params": None,
                        "fitted_curve": None,
                        "quad_dist": quad_dist,  # quad_dist
                        "scaling_factor": scale_fact,
                    }
                )
        print(
            "\n################",
            f" {curve} Mises\tPossible fits: {fit_possible}\tImpossible fits: {fit_impossible}",
            "################\n",
        )

    return save_fits


def sort_fits(fits, thresh1: float = 0.1, thresh2: float = 0.1):  # , thresh3=0.5 | None
    filtered_fits: dict = {}

    # search fits for 1 mises:
    for fit in fits:
        w_idx = fit["weight_idx"]
        quad_dist = fit["quad_dist"]
        curve = fit["curve"]

        if curve == 1:
            if quad_dist <= thresh1:
                filtered_fits[w_idx] = fit

        if w_idx not in filtered_fits:
            if curve == 2:
                if round(quad_dist, 2) <= thresh2:
                    filtered_fits[w_idx] = fit
            elif curve == 3:
                filtered_fits[w_idx] = fit

    sorted_filtered_fits = sorted(
        filtered_fits.items(), key=lambda x: x[1]["weight_idx"]
    )
    return sorted_filtered_fits


def analyze_cnns(dir: str, thresh1: float = 0.1, thresh2: float = 0.1):
    # theta
    num_thetas = 32
    dtheta = 2 * np.pi / num_thetas
    theta = dtheta * np.arange(num_thetas)

    all_results: list = []
    for filename in os.listdir(dir):
        if filename.endswith(".pt"):
            print(os.path.join(dir, filename))
            # load
            tune = orientation_tuning_curve.load_data_from_cnn(
                cnn_name=os.path.join(dir, filename),
                plot_responses=False,
                do_stats=True,
            )

            # fit
            all_fits = fit_curves(tune=tune, theta=theta)

            # sort
            filtered = sort_fits(fits=all_fits, thresh1=thresh1, thresh2=thresh2)

            # store
            all_results.append(filtered)
    return all_results


if __name__ == "__main__":
    num_thetas = 32
    dtheta = 2 * np.pi / num_thetas
    theta = dtheta * np.arange(num_thetas)
    threshold: float = 0.1
    use_saved_tuning: bool = False

    if use_saved_tuning:
        # load from file
        tune = np.load(
            "D:/Katha/Neuroscience/Semester 4/newCode/tuning_CORNER_32o_4p.npy"
        )
    else:
        # load cnn data
        nn = "ArghCNN_numConvLayers3_outChannels[2, 6, 8]_kernelSize[7, 15]_leaky relu_stride1_trainFirstConvLayerTrue_seed299624_Natural_921Epoch_1609-2307"
        PATH = f"D:/Katha/Neuroscience/Semester 4/newCode/kk_contour_net_shallow-main/trained_64er_models/{nn}.pt"

        tune = orientation_tuning_curve.load_data_from_cnn(
            cnn_name=PATH, plot_responses=False, do_stats=True
        )

    all_fits = fit_curves(tune=tune, theta=theta)
    filtered_fits = sort_fits(fits=all_fits)
    save_name: str = "CLASSIC_888_trained_4r8c"
    save_plot: bool = False

    if tune.shape[0] == 8:
        plot_reshaped(
            tune=tune,
            fits=filtered_fits,
            theta=theta,
            save_name=save_name,
            save_plot=save_plot,
        )
    else:
        plot_fit(
            tune=tune,
            fits=filtered_fits,
            theta=theta,
            save_name=save_name,
            save_plot=save_plot,
        )
