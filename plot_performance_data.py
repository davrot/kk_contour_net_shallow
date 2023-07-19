import numpy as np
import glob
import os
from natsort import natsorted
from functions.fisher_exact import fisher_excat_upper, fisher_excat_lower
import matplotlib.pyplot as plt

p_threshold: float = 1.0 / 100.0
file_selection: int = 0
number_of_pattern: int = 60000
path: str = "performance_data"
data_source: str = "test_accuracy"
filename_list = natsorted(glob.glob(os.path.join(path, "performances_*.npz")))
assert file_selection < len(filename_list)

filename = filename_list[file_selection]

data = np.load(filename)

understand_parameter: bool = False
percentage: bool = True
if data_source.upper() == str("test_accuracy").upper():
    to_print = data["test_accuracy"]
    understand_parameter = True
    percentage = True
elif data_source.upper() == str("train_accuracy").upper():
    to_print = data["train_accuracy"]
    understand_parameter = True
    percentage = True
elif data_source.upper() == str("train_losses").upper():
    to_print = data["train_losses"]
    understand_parameter = True
    percentage = False
elif data_source.upper() == str("test_losses").upper():
    to_print = data["test_losses"]
    understand_parameter = True
    percentage = False
assert understand_parameter

if percentage:
    correct_count = np.round(to_print * number_of_pattern / 100.0).astype(np.int64)
    upper = np.zeros((correct_count.shape[0]))
    lower = np.zeros((correct_count.shape[0]))

    for id in range(0, correct_count.shape[0]):
        upper[id] = fisher_excat_upper(
            correct_pattern_count=correct_count[id],
            number_of_pattern=number_of_pattern,
            p_threshold=p_threshold,
        )

        lower[id] = fisher_excat_lower(
            correct_pattern_count=correct_count[id],
            number_of_pattern=number_of_pattern,
            p_threshold=p_threshold,
        )

    x = np.arange(1, to_print.shape[0] + 1)
    y = 100.0 - to_print
    plt.plot(x, y + upper, "k--")
    plt.plot(x, y - lower, "k--")
    plt.plot(x, y, "r")
    plt.ylim([0, (100.0 - to_print.min()) * 1.1])
    plt.xlim([1, to_print.shape[0]])
    plt.xlabel("Epochs")
    plt.ylabel("Error [%]")
    plt.title(data_source.replace("_", " "))
else:
    x = np.arange(1, to_print.shape[0] + 1)
    plt.plot(x, to_print)
    plt.ylim([0, to_print.max() * 1.1])
    plt.xlim([1, to_print.shape[0]])
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(data_source.replace("_", " "))

plt.show()
