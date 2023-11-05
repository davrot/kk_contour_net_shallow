import numpy as np
import fitkarotte
from orientation_tuning_curve import load_data_from_cnn  # noqa
import plot_fit_statistics
import warnings
from scipy.stats import ranksums

# suppress warnings
warnings.filterwarnings("ignore")


def get_general_data_info(data, print_mises_all_cnn: bool):
    num_cnns = len(data)
    num_weights_per_cnn = [len(cnn_results) for cnn_results in data]

    num_fits_per_cnn = {1: [0] * num_cnns, 2: [0] * num_cnns, 3: [0] * num_cnns}

    for idx, cnn_results in enumerate(data):
        for _, fit in cnn_results:
            curve = fit["curve"]
            num_fits_per_cnn[curve][idx] += 1

    print("\n\nNumber of CNNs:", num_cnns)
    print("Number of weights saved for each CNN:", num_weights_per_cnn)
    print("Number of fits with 1 von Mises function per CNN:", num_fits_per_cnn[1])
    print("Number of fits with 2 von Mises functions per CNN:", num_fits_per_cnn[2])
    print("Number of fits with 3 von Mises functions per CNN:", num_fits_per_cnn[3])

    # mean and stdev 4each type of fit
    mean_1_mises = np.mean(num_fits_per_cnn[1])
    std_1_mises = np.std(num_fits_per_cnn[1])
    mean_2_mises = np.mean(num_fits_per_cnn[2])
    std_2_mises = np.std(num_fits_per_cnn[2])
    mean_3_mises = np.mean(num_fits_per_cnn[3])
    std_3_mises = np.std(num_fits_per_cnn[3])

    print(
        f"Mean number of fits with 1 von Mises function: {mean_1_mises:.2f} (std: {std_1_mises:.2f})"
    )
    print(
        f"Mean number of fits with 2 von Mises functions: {mean_2_mises:.2f} (std: {std_2_mises:.2f})"
    )
    print(
        f"Mean number of fits with 3 von Mises functions: {mean_3_mises:.2f} (std: {std_3_mises:.2f})"
    )

    if print_mises_all_cnn:
        print("--================================--")
        for idx_cnn, (num_1_mises, num_2_mises, num_3_mises) in enumerate(
            zip(num_fits_per_cnn[1], num_fits_per_cnn[2], num_fits_per_cnn[3])
        ):
            print(
                f"CNN {idx_cnn+1}:\t# 1 Mises: {num_1_mises},\t# 2 Mises: {num_2_mises},\t# 3 Mises: {num_3_mises}"
            )

    return (
        num_fits_per_cnn,
        mean_1_mises,
        mean_2_mises,
        mean_3_mises,
        std_1_mises,
        std_2_mises,
        std_3_mises,
    )


def ratio_amplitude_two_mises(data, mean_std: bool = False):
    """
    * This function calculates the mean ratio of those weights
    of the first layer, which were fitted with 2 von Mises functions
    * It first calculates the mean ratio  for every single CNN
    (of the overall 20 CNNs)
    * It then computes the overall mean ratio for the weights
    of all 20 CNNs that were fitted with 2 von Mises functions
    """
    num_cnns = len(data)
    mean_ratio_per_cnn = [0] * num_cnns

    for idx, cnn_results in enumerate(data):
        ratio_list: list = []
        count_num_2mises: int = 0
        for _, fit in cnn_results:
            curve = fit["curve"]
            if curve == 2 and fit["fit_params"] is not None:
                count_num_2mises += 1
                first_amp = fit["fit_params"][0]
                sec_amp = fit["fit_params"][3]

                if sec_amp < first_amp:
                    ratio = sec_amp / first_amp
                else:
                    ratio = first_amp / sec_amp

                if not (ratio > 1.0 or ratio < 0):
                    ratio_list.append(ratio)
                else:
                    print(f"\nRATIO OUT OF RANGE FOR: CNN:{idx}, weight{_}\n")

        # print(f"CNN [{idx}]: num fits with 2 von mises = {count_num_2mises}")
        mean_ratio_per_cnn[idx] = np.mean(ratio_list)

    # calc mean difference over all 20 CNNs:
    if mean_std:
        mean_all_cnns = np.mean(mean_ratio_per_cnn)
        std_all_cnns = np.std(mean_ratio_per_cnn)
        print("\n-=== Mean ratio between 2 amplitudes ===-")
        print(f"Mean ratio of all {len(mean_ratio_per_cnn)} CNNs: {mean_all_cnns}")
        print(f"Stdev of ratio of all {len(mean_ratio_per_cnn)} CNNs: {std_all_cnns}")

        return mean_all_cnns, std_all_cnns

    else:  # get median and percentile
        percentiles = np.percentile(mean_ratio_per_cnn, [10, 25, 50, 75, 90])

        print("\n-=== Percentiles of ratio between 2 amplitudes ===-")
        print(f"10th Percentile: {percentiles[0]}")
        print(f"25th Percentile: {percentiles[1]}")
        print(f"Median (50th Percentile): {percentiles[2]}")
        print(f"75th Percentile: {percentiles[3]}")
        print(f"90th Percentile: {percentiles[4]}")

        # return mean_all_cnns, std_all_cnns
        return percentiles[2], (percentiles[1], percentiles[3])


def ratio_amplitude_three_mises(data, mean_std: bool = False):
    """
    * returns: mean21, std21, mean32, std32
    * This function calculates the mean ratio of those weights
    of the first layer, which were fitted with 2 von Mises functions
    * It first calculates the mean ratio  for every single CNN
    (of the overall 20 CNNs)
    * It then computes the overall mean ratio for the weights
    of all 20 CNNs that were fitted with 2 von Mises functions
    """
    num_cnns = len(data)
    mean_ratio_per_cnn21 = [0] * num_cnns
    mean_ratio_per_cnn32 = [0] * num_cnns

    for idx, cnn_results in enumerate(data):
        ratio_list21: list = []
        ratio_list32: list = []
        count_num_2mises: int = 0
        for _, fit in cnn_results:
            curve = fit["curve"]
            if curve == 3 and fit["fit_params"] is not None:
                count_num_2mises += 1
                first_amp = fit["fit_params"][0]
                sec_amp = fit["fit_params"][3]
                third_amp = fit["fit_params"][6]

                if sec_amp < first_amp:
                    ratio21 = sec_amp / first_amp
                else:
                    ratio21 = first_amp / sec_amp

                if third_amp < sec_amp:
                    ratio32 = third_amp / sec_amp
                else:
                    ratio32 = sec_amp / third_amp

                if not (ratio21 > 1.0 or ratio32 > 1.0 or ratio21 < 0 or ratio32 < 0):
                    ratio_list21.append(ratio21)
                    ratio_list32.append(ratio32)
                else:
                    print(f"\nRATIO OUT OF RANGE FOR: CNN:{idx}, weight{_}\n")

        # print(f"CNN [{idx}]: num fits with 2 von mises =
        # {count_num_2mises}")
        if len(ratio_list21) != 0:
            mean_ratio_per_cnn21[idx] = np.mean(ratio_list21)
            mean_ratio_per_cnn32[idx] = np.mean(ratio_list32)
        else:
            mean_ratio_per_cnn21[idx] = None  # type: ignore
            mean_ratio_per_cnn32[idx] = None  # type: ignore

    mean_ratio_per_cnn21 = [x for x in mean_ratio_per_cnn21 if x is not None]
    mean_ratio_per_cnn32 = [x for x in mean_ratio_per_cnn32 if x is not None]

    # calc mean difference over all 20 CNNs:

    if mean_std:
        mean_all_cnns21 = np.mean(mean_ratio_per_cnn21)
        std_all_21 = np.std(mean_ratio_per_cnn21)
        mean_all_cnns32 = np.mean(mean_ratio_per_cnn32)
        std_all_32 = np.std(mean_ratio_per_cnn32)

        print("\n-=== Mean ratio between 3 preferred orienations ===-")
        print(f"Ratio 2/1 of all {len(mean_ratio_per_cnn21)} CNNs: {mean_all_cnns21}")
        print(
            f"Stdev of ratio 2/1 of all {len(mean_ratio_per_cnn21)} CNNs: {std_all_21}"
        )
        print(f"Ratio 3/2 of all {len(mean_ratio_per_cnn32)} CNNs: {mean_all_cnns32}")
        print(
            f"Stdev of ratio 3/2 of all {len(mean_ratio_per_cnn32)} CNNs: {std_all_32}"
        )

        return mean_all_cnns21, std_all_21, mean_all_cnns32, std_all_32

    else:  # get median and percentile:
        percentiles_21 = np.percentile(mean_ratio_per_cnn32, [10, 25, 50, 75, 90])
        percentiles_32 = np.percentile(mean_ratio_per_cnn21, [10, 25, 50, 75, 90])

        # get percentile 25 and 75
        percentile25_32 = percentiles_32[1]
        percentile75_32 = percentiles_32[-2]
        percentile25_21 = percentiles_21[1]
        percentile75_21 = percentiles_21[-2]

        print("\n-=== Percentiles of ratio between 2 amplitudes ===-")
        print(f"10th Percentile 3->2: {percentiles_32[0]}")
        print(f"10th Percentile 2->1: {percentiles_21[0]}")
        print(f"25th Percentile 3->2: {percentiles_32[1]}")
        print(f"25th Percentile 2->1: {percentiles_21[1]}")
        print(f"Median (50th Percentile 3->2): {percentiles_32[2]}")
        print(f"Median (50th Percentile 2->1): {percentiles_21[2]}")
        print(f"75th Percentile 3->2: {percentiles_32[3]}")
        print(f"75th Percentile 2->1: {percentiles_21[3]}")
        print(f"90th Percentile3->2: {percentiles_32[4]}")
        print(f"90th Percentile 2->1: {percentiles_21[4]}")

        return (
            percentiles_21[2],
            (percentile25_21, percentile75_21),
            percentiles_32[2],
            (percentile25_32, percentile75_32),
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
    print("Null-hypothesis: distributions are the same.")
    alpha = 0.05
    if p_value < alpha:
        print("The distributions are significantly different.")
    else:
        print("The distributions are not significantly different.")

    return p_value


def ks(data_classic, data_corner):
    from scipy.stats import ks_2samp

    ks_statistic, p_value = ks_2samp(data_classic, data_corner)

    print("\nKolmogorov-Smirnov Test - p-value:", p_value)
    print("Kolmogorov-Smirnov Test - ks_statistic:", ks_statistic)
    alpha = 0.05
    if p_value < alpha:
        print("The distributions for von Mises functions are significantly different.")

    return p_value


def shapiro(fits_per_mises, num_mises: int, alpha: float = 0.05):
    """
    Tests if data has normal distribution
    * 0-hyp: data is normally distributed
    * low p-val: data not normally distributed
    """
    from scipy.stats import shapiro

    statistic, p_value = shapiro(fits_per_mises)
    print(f"\nShapiro-Wilk Test for {num_mises} von Mises function - p-val :", p_value)
    print(
        f"Shapiro-Wilk Test for {num_mises} von Mises function - statistic :", statistic
    )

    # set alpha
    if p_value < alpha:
        print("P-val < alpha. Reject 0-hypothesis. Data is not normally distributed")
    else:
        print("P-val > alpha. Keep 0-hypothesis. Data is normally distributed")

    return p_value


def agostino_pearson(fits_per_mises, num_mises: int, alpha: float = 0.05):
    """
    Tests if data has normal distribution
    * 0-hyp: data is normally distributed
    * low p-val: data not normally distributed
    """
    from scipy import stats

    statistic, p_value = stats.normaltest(fits_per_mises)
    print(
        f"\nD'Agostino-Pearson Test for {num_mises} von Mises function - p-val :",
        p_value,
    )
    print(
        f"D'Agostino-Pearson Test for {num_mises} von Mises function - statistic :",
        statistic,
    )

    # set alpha
    if p_value < alpha:
        print("P-val < alpha. Reject 0-hypothesis. Data is not normally distributed")
    else:
        print("P-val > alpha. Keep 0-hypothesis. Data is normally distributed")

    return p_value


if __name__ == "__main__":
    num_thetas = 32
    dtheta = 2 * np.pi / num_thetas
    theta = dtheta * np.arange(num_thetas)
    threshold: float = 0.1

    # to do statistics on corner
    directory_corner: str = "D:/Katha/Neuroscience/Semester 4/newCode/kk_contour_net_shallow-main/corner_888"
    all_results_corner = fitkarotte.analyze_cnns(dir=directory_corner)

    # analyze
    print("-=== CORNER ===-")
    # amplitude ratios
    ratio_corner_21, std_corner_21 = ratio_amplitude_two_mises(data=all_results_corner)
    (
        ratio_corner_321,
        std_corner_321,
        ratio_corner_332,
        std_corner_332,
    ) = ratio_amplitude_three_mises(data=all_results_corner)

    # general data
    (
        corner_num_fits,
        mean_corner_1,
        mean_corner_2,
        mean_corner_3,
        std_corner_1,
        std_corner_2,
        std_corner_3,
    ) = get_general_data_info(data=all_results_corner, print_mises_all_cnn=True)
    # analyze_num_curve_fits(data=all_results_corner)

    # to do statistics: CLASSIC
    directory_classic: str = "D:/Katha/Neuroscience/Semester 4/newCode/kk_contour_net_shallow-main/classic_888"
    all_results_classic = fitkarotte.analyze_cnns(dir=directory_classic)

    # analyze
    print("-=== CLASSIC ===-")
    # amplitude ratio
    ratio_classic_21, std_class_21 = ratio_amplitude_two_mises(data=all_results_classic)
    (
        ratio_classic_321,
        std_classic_321,
        ratio_classic_332,
        std_classic_332,
    ) = ratio_amplitude_three_mises(data=all_results_classic)

    # general data
    (
        classic_num_fits,
        mean_classic_1,
        mean_classic_2,
        mean_classic_3,
        std_classic_1,
        std_classic_2,
        std_classic_3,
    ) = get_general_data_info(data=all_results_classic, print_mises_all_cnn=False)
    # analyze_num_curve_fits(data=all_results_classic)

    print("################################")
    print("-==== plotting hists: compare amplitude ratios ====-")
    plot_fit_statistics.plot_mean_percentile_amplit_ratio(
        ratio_classic_21=ratio_classic_21,
        ratio_classic_321=ratio_classic_321,
        ratio_classic_332=ratio_classic_332,
        ratio_corner_21=ratio_corner_21,
        ratio_corner_321=ratio_corner_321,
        ratio_corner_332=ratio_corner_332,
        percentile_classic21=std_class_21,
        percentile_classic321=std_classic_321,
        percentile_classic_332=std_classic_332,
        percentile_corner_21=std_corner_21,
        percentile_corner_321=std_corner_321,
        percentile_corner_332=std_corner_332,
        saveplot=True,
        save_name="median_percentile_888",
    )

    # p-value < 0.05:  statistically significant difference between your two samples
    statistic21, pvalue21 = ranksums(ratio_classic_21, ratio_corner_21)
    print(
        f"Wilcoxon rank sum test 2 Mises for ratio 2->1: statistic={statistic21}, pvalue={pvalue21}"
    )

    statistic321, pvalue321 = ranksums(ratio_classic_321, ratio_corner_321)
    print(
        f"Wilcoxon rank sum test 3 Mises for ratio 2->1: statistic={statistic321}, pvalue={pvalue321}"
    )

    statistic332, pvalue332 = ranksums(ratio_classic_332, ratio_corner_332)
    print(
        f"Wilcoxon rank sum test 3 Mises for ratio 3->2: statistic={statistic332}, pvalue={pvalue332}"
    )

    print("-==== plotting hists: CORNER ====-")
    # plot histogram
    # plot_hist(corner_num_fits[1], num_mises=1)
    # plot_hist(corner_num_fits[2], num_mises=2)
    # plot_hist(corner_num_fits[3], num_mises=3)

    # test for normal distribution
    print("-== Shapiro test ==- ")
    # shapiro(corner_num_fits[1], num_mises=1)
    # shapiro(corner_num_fits[2], num_mises=2)
    # shapiro(corner_num_fits[3], num_mises=3)

    print("\n-== D'Agostino-Pearson test ==- ")
    agostino_pearson(corner_num_fits[1], num_mises=1)
    agostino_pearson(corner_num_fits[2], num_mises=2)
    agostino_pearson(corner_num_fits[3], num_mises=3)

    print("-==== plotting hists: CLASSIC ====-")
    # plot histogram
    # plot_hist(classic_num_fits[1], num_mises=1)
    # plot_hist(classic_num_fits[2], num_mises=2)
    # plot_hist(classic_num_fits[3], num_mises=3)

    # test for normal distribution
    print("-== Shapiro test ==- ")
    # shapiro(classic_num_fits[1], num_mises=1)
    # shapiro(classic_num_fits[2], num_mises=2)
    # shapiro(classic_num_fits[3], num_mises=3)

    print("\n-== D'Agostino-Pearson test ==- ")
    agostino_pearson(classic_num_fits[1], num_mises=1)
    agostino_pearson(classic_num_fits[2], num_mises=2)
    agostino_pearson(classic_num_fits[3], num_mises=3)

    # statistics for each von mises:
    print("######################")
    print(" -=== CLASSIC vs CORNER ===-")
    # 1:
    willy_is_not_whitney_test(
        data_classic=classic_num_fits[1], data_corner=corner_num_fits[1]
    )

    # 2:
    willy_is_not_whitney_test(
        data_classic=classic_num_fits[2], data_corner=corner_num_fits[2]
    )

    # 3:
    willy_is_not_whitney_test(
        data_classic=classic_num_fits[3], data_corner=corner_num_fits[3]
    )

    # visualize as bar plots:
    plot_fit_statistics.plot_means_std_corner_classic(
        means_classic=[mean_classic_1, mean_classic_2, mean_classic_3],
        means_corner=[mean_corner_1, mean_corner_2, mean_corner_3],
        std_classic=[std_classic_1, std_classic_2, std_classic_3],
        std_corner=[std_corner_1, std_corner_2, std_corner_3],
        saveplot=False,
        save_name="3288",
    )
