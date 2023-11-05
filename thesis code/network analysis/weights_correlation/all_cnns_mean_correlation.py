import torch
import sys
import os
import matplotlib.pyplot as plt  # noqa
import numpy as np
import matplotlib as mpl

mpl.rcParams["text.usetex"] = True
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.size"] = 15

# import files from parent dir
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from functions.make_cnn import make_cnn  # noqa


def show_20mean_correlations(model_list, save: bool = False, cnn: str = "CORNER"):
    """
    Displays a correlation matrix for every single of the 20 CNNs
    """

    fig, axs = plt.subplots(4, 5, figsize=(15, 15))
    for i, load_model in enumerate(model_list):
        # load model
        model = torch.load(load_model).to("cpu")
        model.eval()

        # load 2nd convs weights
        weights = model[3].weight.cpu().detach().clone().numpy()
        corr_matrices = []
        for j in range(weights.shape[0]):
            w_j = weights[j]
            w = w_j.reshape(w_j.shape[0], -1)
            corr_matrix = np.corrcoef(w)
            corr_matrices.append(corr_matrix)

        mean_corr_matrix = np.mean(corr_matrices, axis=0)
        ax = axs[i // 5, i % 5]
        im = ax.matshow(mean_corr_matrix, cmap="RdBu_r")
        cbar = fig.colorbar(
            im, ax=ax, fraction=0.046, pad=0.04, ticks=np.arange(-1.1, 1.1, 0.2)
        )
        ax.set_title(f"Model {i+1}")

        # remove lower x-axis ticks
        ax.tick_params(axis="x", which="both", bottom=False)
        ax.tick_params(axis="both", which="major", labelsize=14)
        cbar.ax.tick_params(labelsize=13)

    # fig.colorbar(im, ax=axs.ravel().tolist())
    plt.tight_layout()
    if save:
        plt.savefig(
            f"additional thesis plots/saved_plots/weight plots/all20cnn_mean_corr_{cnn}.pdf",
            dpi=300,
        )
    plt.show()


def show_overall_mean_correlation(model_list, save: bool = False, cnn: str = "CORNER"):
    """
    Displays the mean correlation across all 20 CNNs
    """

    fig, ax = plt.subplots(figsize=(7, 7))
    overall_corr_matrices = []
    for i, load_model in enumerate(model_list):
        # load model
        model = torch.load(load_model).to("cpu")
        model.eval()

        # load 2nd convs weights
        weights = model[3].weight.cpu().detach().clone().numpy()
        corr_matrices = []
        for j in range(weights.shape[0]):
            w_j = weights[j]
            w = w_j.reshape(w_j.shape[0], -1)
            corr_matrix = np.corrcoef(w)
            corr_matrices.append(corr_matrix)

        mean_corr_matrix = np.mean(corr_matrices, axis=0)
        overall_corr_matrices.append(mean_corr_matrix)

    overall_mean_corr_matrix = np.mean(overall_corr_matrices, axis=0)
    im = ax.matshow(overall_mean_corr_matrix, cmap="RdBu_r")
    cbar = fig.colorbar(
        im, ax=ax, fraction=0.046, pad=0.04, ticks=np.arange(-1.1, 1.1, 0.1)
    )

    # remove lower x-axis ticks
    ax.tick_params(axis="x", which="both", bottom=False)
    ax.tick_params(axis="both", which="major", labelsize=17)
    cbar.ax.tick_params(labelsize=15)

    plt.tight_layout()
    if save:
        plt.savefig(
            f"additional thesis plots/saved_plots/weight plots/mean20cnn_mean_corr_{cnn}.pdf",
            dpi=300,
        )
    plt.show()
    return overall_mean_corr_matrix


def get_file_list_all_cnns(dir: str) -> list:
    all_results: list = []
    for filename in os.listdir(dir):
        if filename.endswith(".pt"):
            # print(os.path.join(dir, filename))
            all_results.append(os.path.join(dir, filename))

    return all_results


def test_normality(correlation_data, condition: str, alpha: float = 0.05):
    """
    Tests if data has normal distribution
    * 0-hyp: data is normally distributed
    * low p-val: data not normally distributed
    """
    from scipy import stats

    statistic, p_value = stats.normaltest(correlation_data)
    print(
        f"\nD'Agostino-Pearson Test for {condition} - p-val :",
        p_value,
    )
    print(
        f"D'Agostino-Pearson Test for {condition} - statistic :",
        statistic,
    )

    # set alpha
    if p_value < alpha:
        print("P-val < alpha. Reject 0-hypothesis. Data is not normally distributed")
    else:
        print("P-val > alpha. Keep 0-hypothesis. Data is normally distributed")

    return p_value


def two_sample_ttest(corr_classic, corr_coner, alpha: float = 0.05):
    """
    This is a test for the null hypothesis that 2 independent samples have identical average (expected) values. This test assumes that the populations have identical variances by default.
    """

    from scipy.stats import ttest_ind

    t_stat, p_value = ttest_ind(corr_classic, corr_coner)
    print(f"t-statistic: {t_stat}")

    # check if the p-value less than significance level
    if p_value < alpha:
        print(
            "There is a significant difference in the mean correlation values between the two groups."
        )
    else:
        print(
            "There is no significant difference in the mean correlation values between the two groups."
        )


def willy_is_not_whitney_test(data_classic, data_corner):
    from scipy.stats import mannwhitneyu

    """
    * Test does not assume normal distribution
    * Compares means between 2 indep groups
    """

    # call test
    statistic, p_value = mannwhitneyu(data_classic, data_corner)

    # results
    print("\nMann-Whitney U Test Statistic:", statistic)
    print("Mann-Whitney U Test p-value:", p_value)

    # check significance:
    alpha = 0.05
    if p_value < alpha:
        print("The distributions are significantly different.")
    else:
        print("The distributions are not significantly different.")

    return p_value


def visualize_differences(corr_class, corr_corn, save: bool = False):
    # calc mean, std, median
    mean_class = np.mean(corr_class)
    median_class = np.median(corr_class)
    std_class = np.std(corr_class)

    mean_corn = np.mean(corr_corn)
    median_corn = np.median(corr_corn)
    std_corn = np.std(corr_corn)

    # plot
    labels = ["Mean", "Median", "Standard Deviation"]
    condition_class = [mean_class, median_class, std_class]
    condition_corn = [mean_corn, median_corn, std_corn]

    x = np.arange(len(labels))
    width = 0.35

    _, ax = plt.subplots(figsize=(7, 7))
    rect_class = ax.bar(
        x - width / 2, condition_class, width, label="CLASSIC", color="cornflowerblue"
    )
    rect_corn = ax.bar(
        x + width / 2, condition_corn, width, label="CORNER", color="coral"
    )

    # show bar values
    for i, rect in enumerate(rect_class + rect_corn):
        height = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=15,
        )

    # ax.set_ylabel('Value')
    ax.set_title("Summary Statistics by Condition")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=17)
    ax.legend()

    plt.tight_layout()
    if save:
        plt.savefig(
            "additional thesis plots/saved_plots/weight plots/summary_stats_correlation_CLASSvsCORN.pdf",
            dpi=300,
        )
    plt.show()


if __name__ == "__main__":
    #  CLASSIC:
    directory_classic: str = "D:/Katha/Neuroscience/Semester 4/newCode/kk_contour_net_shallow-main/classic3288_fest"
    all_results_classic = get_file_list_all_cnns(dir=directory_classic)
    show_20mean_correlations(all_results_classic)
    mean_corr_classic = show_overall_mean_correlation(all_results_classic)

    #  CORNER:
    directory_corner: str = "D:/Katha/Neuroscience/Semester 4/newCode/kk_contour_net_shallow-main/corner3288_fest"
    all_results_corner = get_file_list_all_cnns(dir=directory_corner)
    show_20mean_correlations(all_results_corner)
    mean_corr_corner = show_overall_mean_correlation(all_results_corner)

    # flatten
    corr_classic = mean_corr_classic.flatten()
    corr_corner = mean_corr_corner.flatten()

    # test how data is distributed
    p_class = test_normality(correlation_data=corr_classic, condition="CLASSIC")
    p_corn = test_normality(correlation_data=corr_corner, condition="CORNER")

    # perform statistical test:
    alpha: float = 0.05

    if p_class < alpha and p_corn < alpha:
        willy_is_not_whitney_test(data_classic=corr_classic, data_corner=corr_corner)
    else:
        # do ttest:
        two_sample_ttest(corr_classic=corr_classic, corr_coner=corr_corner)

    # visualize the differences:
    visualize_differences(corr_class=corr_classic, corr_corn=corr_corner, save=True)
