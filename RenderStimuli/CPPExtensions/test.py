import torch
import matplotlib.pyplot as plt
import os

from PyTCopyCPU import TCopyCPU


copyier = TCopyCPU()

# Output canvas
output_pattern: int = 10
output_x: int = 160
output_y: int = 200  # Last dim

output = torch.zeros(
    (output_pattern, output_x, output_y), device="cpu", dtype=torch.float32
)

# The "gabors"
garbor_amount: int = 3
garbor_x: int = 11
garbor_y: int = 13  # Last dim

gabor = torch.arange(
    1, garbor_amount * garbor_x * garbor_y + 1, device="cpu", dtype=torch.float32
).reshape((garbor_amount, garbor_x, garbor_y))

# The sparse control matrix

sparse_matrix = torch.zeros((3, 4), device="cpu", dtype=torch.int64)

sparse_matrix[0, 0] = 0  # pattern_id -> output dim 0
sparse_matrix[0, 1] = 2  # gabor_type -> gabor dim 0
sparse_matrix[0, 2] = 0  # gabor_x -> output dim 1 start point
sparse_matrix[0, 3] = 0  # gabor_x -> output dim 2 start point

sparse_matrix[1, 0] = 0  # pattern_id -> output dim 0
sparse_matrix[1, 1] = 1  # gabor_type -> gabor dim 0
sparse_matrix[1, 2] = 40  # gabor_x -> output dim 1 start point
sparse_matrix[1, 3] = 60  # gabor_x -> output dim 2 start point

sparse_matrix[2, 0] = 1  # pattern_id -> output dim 0
sparse_matrix[2, 1] = 0  # gabor_type -> gabor dim 0
sparse_matrix[2, 2] = 0  # gabor_x -> output dim 1 start point
sparse_matrix[2, 3] = 0  # gabor_x -> output dim 2 start point

# ###########################################

assert sparse_matrix.ndim == 2
assert int(sparse_matrix.shape[1]) == 4

assert gabor.ndim == 3
assert output.ndim == 3

number_of_cpu_processes = os.cpu_count()

copyier.process(
    sparse_matrix.data_ptr(),
    int(sparse_matrix.shape[0]),
    int(sparse_matrix.shape[1]),
    gabor.data_ptr(),
    int(gabor.shape[0]),
    int(gabor.shape[1]),
    int(gabor.shape[2]),
    output.data_ptr(),
    int(output.shape[0]),
    int(output.shape[1]),
    int(output.shape[2]),
    int(number_of_cpu_processes),
)

plt.imshow(output[0, :, :])
plt.title("Pattern 0")
plt.show()

plt.imshow(output[1, :, :])
plt.title("Pattern 1")
plt.show()
