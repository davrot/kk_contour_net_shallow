from scipy.stats import fisher_exact


def fisher_excat_upper(
    correct_pattern_count: int, number_of_pattern: int, p_threshold: float = 5.0 / 100.0
) -> float:
    error_pattern_count = int(number_of_pattern - correct_pattern_count)

    bound = 100.0
    for u in range(0, correct_pattern_count):
        z = int(error_pattern_count + u)
        _, pvalue = fisher_exact(
            [[correct_pattern_count, error_pattern_count], [number_of_pattern - z, z]],
            alternative="greater",
        )
        if bool(pvalue > p_threshold) is False:
            bound = u * 100.0 / number_of_pattern
            break

    return bound


def fisher_excat_lower(
    correct_pattern_count: int, number_of_pattern: int, p_threshold: float = 5.0 / 100.0
) -> float:
    error_pattern_count = int(number_of_pattern - correct_pattern_count)

    bound = 0.0
    for u in range(0, error_pattern_count):
        z = int(error_pattern_count - u)
        _, pvalue = fisher_exact(
            [[correct_pattern_count, error_pattern_count], [number_of_pattern - z, z]],
            alternative="less",
        )
        if bool(pvalue > p_threshold) is False:
            bound = u * 100.0 / number_of_pattern
            break

    return bound
