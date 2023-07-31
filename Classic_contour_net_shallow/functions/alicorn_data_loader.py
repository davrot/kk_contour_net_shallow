import torch
import numpy as np
import os


@torch.no_grad()
def alicorn_data_loader(
    num_pfinkel: list[int] | None,
    load_stimuli_per_pfinkel: int,
    condition: str,
    data_path: str,
    logger=None,  
) -> torch.utils.data.TensorDataset:
    """
    - num_pfinkel: list of the angles that should be loaded (ranging from
    0-90). If None: all pfinkels loaded
    - stimuli_per_pfinkel: defines amount of stimuli per path angle but
    for label 0 and label 1 seperatly (e.g., stimuli_per_pfinkel = 1000:
    1000 stimuli = label 1, 1000 stimuli = label 0)
    """
    filename: str | None = None
    if condition == "Angular":
        filename = "angular_angle"
    elif condition == "Coignless":
        filename = "base_angle"
    elif condition == "Natural":
        filename = "corner_angle"
    else:
        filename = None
    assert filename is not None
    filepaths: str = os.path.join(data_path, f"{condition}")

    stimuli_per_pfinkel: int = 100000

    # ----------------------------

    # for angles and batches
    if num_pfinkel is None:
        angle: list[int] = np.arange(0, 100, 10).tolist()
    else:
        angle = num_pfinkel

    assert isinstance(angle, list)

    batch: list[int] = np.arange(1, 11, 1).tolist()

    if load_stimuli_per_pfinkel <= (stimuli_per_pfinkel // len(batch)):
        num_img_per_pfinkel: int = load_stimuli_per_pfinkel
        num_batches: int = 1
    else:
        # handle case where more than 10,000 stimuli per pfinkel needed
        num_batches = load_stimuli_per_pfinkel // (stimuli_per_pfinkel // len(batch))
        num_img_per_pfinkel = load_stimuli_per_pfinkel // num_batches

    if logger is not None:
        logger.info(f"{num_batches} batches")
        logger.info(f"{num_img_per_pfinkel} stimuli per pfinkel.")

    # initialize data and label tensors:
    num_stimuli: int = len(angle) * num_batches * num_img_per_pfinkel * 2
    data_tensor: torch.Tensor = torch.empty(
        (num_stimuli, 200, 200), dtype=torch.uint8, device=torch.device("cpu")
    )
    label_tensor: torch.Tensor = torch.empty(
        (num_stimuli), dtype=torch.int64, device=torch.device("cpu")
    )

    if logger is not None:
        logger.info(f"data tensor shape: {data_tensor.shape}")
        logger.info(f"label tensor shape: {label_tensor.shape}")

    # append data
    idx: int = 0
    for i in range(len(angle)):
        for j in range(num_batches):
            # load contour
            temp_filename: str = (
                f"{filename}_{angle[i]:03}_b{batch[j]:03}_n10000_RENDERED.npz"
            )
            contour_filename: str = os.path.join(filepaths, temp_filename)
            c_data = np.load(contour_filename)
            data_tensor[idx : idx + num_img_per_pfinkel, ...] = torch.tensor(
                c_data["gaborfield"][:num_img_per_pfinkel, ...],
                dtype=torch.uint8,
                device=torch.device("cpu"),
            )
            label_tensor[idx : idx + num_img_per_pfinkel] = int(1)
            idx += num_img_per_pfinkel

    # next append distractor stimuli
    for i in range(len(angle)):
        for j in range(num_batches):
            # load distractor
            temp_filename = (
                f"{filename}_{angle[i]:03}_dist_b{batch[j]:03}_n10000_RENDERED.npz"
            )
            distractor_filename: str = os.path.join(filepaths, temp_filename)
            nc_data = np.load(distractor_filename)
            data_tensor[idx : idx + num_img_per_pfinkel, ...] = torch.tensor(
                nc_data["gaborfield"][:num_img_per_pfinkel, ...],
                dtype=torch.uint8,
                device=torch.device("cpu"),
            )
            label_tensor[idx : idx + num_img_per_pfinkel] = int(0)
            idx += num_img_per_pfinkel

    return torch.utils.data.TensorDataset(label_tensor, data_tensor.unsqueeze(1))
