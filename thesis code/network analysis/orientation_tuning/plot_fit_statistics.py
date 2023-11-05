import matplotlib.pyplot as plt
import numpy as np
import warnings
import matplotlib as mpl

mpl.rcParams["text.usetex"] = True
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.size"] = 15

# suppress warnings
warnings.filterwarnings("ignore")


def autolabel(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            f"{height:.2f}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(-10, 3),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=17,
        )


def plot_mean_percentile_amplit_ratio(
    ratio_classic_21: float,
    ratio_corner_21: float,
    ratio_classic_321: float,
    ratio_corner_321: float,
    ratio_classic_332: float,
    ratio_corner_332: float,
    percentile_classic21: tuple,
    percentile_classic321: tuple,
    percentile_classic_332: tuple,
    percentile_corner_21: tuple,
    percentile_corner_321: tuple,
    percentile_corner_332: tuple,
    save_name: str,
    saveplot: bool,
):
    num_von_mises = [2, 3, 3]  # X-axis ticks

    # bar setup
    bar_width = 0.35
    index = np.arange(len(num_von_mises))

    # position error bars correctly:
    lower_err_classic = [
        ratio_classic_21 - percentile_classic21[0],
        ratio_classic_332 - percentile_classic_332[0],
        ratio_classic_321 - percentile_classic321[0],
    ]
    upper_err_classic = [
        percentile_classic21[1] - ratio_classic_21,
        percentile_classic_332[1] - ratio_classic_332,
        percentile_classic321[1] - ratio_classic_321,
    ]

    lower_err_corner = [
        ratio_corner_21 - percentile_corner_21[0],
        ratio_corner_332 - percentile_corner_332[0],
        ratio_corner_321 - percentile_corner_321[0],
    ]
    upper_err_corner = [
        percentile_corner_21[1] - ratio_corner_21,
        percentile_corner_332[1] - ratio_corner_332,
        percentile_corner_321[1] - ratio_corner_321,
    ]

    yerr_classic = [lower_err_classic, upper_err_classic]
    yerr_corner = [lower_err_corner, upper_err_corner]

    # subplots
    fig, ax = plt.subplots(figsize=(7, 7))
    bars_classic = ax.bar(
        index - bar_width / 2,
        [ratio_classic_21, ratio_classic_332, ratio_classic_321],
        bar_width,
        yerr=yerr_classic,
        capsize=5,
        label="Classic",
        color="cornflowerblue",
    )
    bars_corner = ax.bar(
        index + bar_width / 2,
        [ratio_corner_21, ratio_corner_332, ratio_corner_321],
        bar_width,
        yerr=yerr_corner,
        capsize=5,
        label="Corner",
        color="coral",
    )

    autolabel(bars_classic, ax)
    autolabel(bars_corner, ax)

    ax.set_ylabel("Median ratio of amplitudes", fontsize=18)
    ax.set_xticks(index)
    ax.set_xticklabels(
        [
            "2 von Mises \n(min/max)",
            "3 von Mises \n(mid/max)",
            "3 von Mises\n(min/mid)",
        ],
        fontsize=17,
    )
    ax.legend(fontsize=17)
    ax.set_ylim(bottom=0.0)

    # plot
    plt.yticks(fontsize=17)
    plt.tight_layout()
    if saveplot:
        plt.savefig(
            f"additional thesis plots/saved_plots/fitkarotte/median_quartiles_ampli_ratio_{save_name}_corn_class.pdf",
            dpi=300,
            bbox_inches="tight",
        )
    plt.show(block=True)


def plot_means_std_corner_classic(
    means_classic: list,
    means_corner: list,
    std_classic: list,
    std_corner: list,
    saveplot: bool,
    save_name: str,
):
    num_von_mises = [1, 2, 3]  # X-axis ticks

    # bar setup
    bar_width = 0.35
    index = np.arange(len(num_von_mises))

    # subplots
    fig, ax = plt.subplots(figsize=(7, 7))
    bars_classic = ax.bar(
        index - bar_width / 2,
        means_classic,
        bar_width,
        yerr=std_classic,
        capsize=5,
        label="Classic",
        color="cornflowerblue",
    )
    bars_corner = ax.bar(
        index + bar_width / 2,
        means_corner,
        bar_width,
        yerr=std_corner,
        capsize=5,
        label="Corner",
        color="coral",
    )

    autolabel(bars_classic, ax)
    autolabel(bars_corner, ax)

    ax.set_ylabel("Average number of fits", fontsize=17)
    ax.set_xticks(index)
    ax.set_xticklabels(["1 von Mises", "2 von Mises", "3 von Mises"], fontsize=17)
    ax.legend(fontsize=16)
    ax.set_ylim(bottom=0.0)

    # plot
    plt.yticks(fontsize=17)
    plt.tight_layout()
    if saveplot:
        plt.savefig(
            f"additional thesis plots/saved_plots/fitkarotte/y_lim_mean_fits_{save_name}_corn_class.pdf",
            dpi=300,
        )
    plt.show(block=True)


def plot_mean_std_amplit_ratio(
    ratio_classic_21: float,
    std_class_21: float,
    ratio_corner_21: float,
    std_corn_21: float,
    ratio_classic_321: float,
    std_class_321: float,
    ratio_corner_321: float,
    std_corn_321: float,
    ratio_classic_332: float,
    std_class_332: float,
    ratio_corner_332: float,
    std_corn_332: float,
    save_name: str,
    saveplot: bool,
):
    num_von_mises = [2, 3, 3]  # X-axis ticks

    # bar setup
    bar_width = 0.35
    index = np.arange(len(num_von_mises))

    # subplots
    fig, ax = plt.subplots(figsize=(12, 7))
    bars_classic = ax.bar(
        index - bar_width / 2,
        [ratio_classic_21, ratio_classic_332, ratio_classic_321],
        bar_width,
        yerr=[std_class_21, std_class_332, std_class_321],
        capsize=5,
        label="Classic",
        color="cornflowerblue",
    )
    bars_corner = ax.bar(
        index + bar_width / 2,
        [ratio_corner_21, ratio_corner_332, ratio_corner_321],
        bar_width,
        yerr=[std_corn_21, std_corn_332, std_corn_321],
        capsize=5,
        label="Corner",
        color="coral",
    )

    autolabel(bars_classic, ax)
    autolabel(bars_corner, ax)

    ax.set_ylabel("Mean ratio of amplitudes", fontsize=17)
    ax.set_xticks(index)
    ax.set_xticklabels(
        [
            "2 von Mises \n(max/min)",
            "3 von Mises \n(max/mid)",
            "3 von Mises\n(mid/min)",
        ],
        fontsize=17,
    )
    ax.legend(fontsize=16)
    ax.set_ylim(bottom=0.0)

    # plot
    plt.yticks(fontsize=17)
    plt.tight_layout()
    if saveplot:
        plt.savefig(
            f"additional thesis plots/saved_plots/fitkarotte/y_lim_mean_std_ampli_ratio_{save_name}_corn_class.pdf",
            dpi=300,
            bbox_inches="tight",
        )
    plt.show(block=True)


def plot_hist(fits_per_mises, num_mises: int):
    """
    Plot to see if data has normal distribution
    """
    # get correct x-ticks
    x_ticks = np.arange(start=min(fits_per_mises), stop=max(fits_per_mises) + 1, step=1)

    # plot
    plt.hist(
        fits_per_mises,
        # bins=bins,
        alpha=0.5,
        label=f"{num_mises} von Mises function",
        align="mid",
    )
    plt.xlabel(f"Number of weights fitted with {num_mises} ")
    plt.ylabel(f"Frequency of fit with {num_mises} for 20 CNNs")
    plt.title(f"Histogram of Fits with {num_mises} von Mises Function")
    plt.xticks(x_ticks)
    plt.legend()
    plt.tight_layout()
    plt.show(block=True)
