import numpy as np
import matplotlib.pyplot as plt  # noqa


def change_base(
    x: np.ndarray, y: np.ndarray, theta: float
) -> tuple[np.ndarray, np.ndarray]:
    x_theta: np.ndarray = x.astype(dtype=np.float32) * np.cos(theta) + y.astype(
        dtype=np.float32
    ) * np.sin(theta)
    y_theta: np.ndarray = y.astype(dtype=np.float32) * np.cos(theta) - x.astype(
        dtype=np.float32
    ) * np.sin(theta)
    return x_theta, y_theta


def cos_gabor_function(
    x: np.ndarray, y: np.ndarray, theta: float, f: float, sigma: float, phi: float
) -> np.ndarray:
    r_a: np.ndarray = change_base(x, y, theta)[0]
    r_b: np.ndarray = change_base(x, y, theta)[1]
    r2 = r_a**2 + r_b**2
    gauss: np.ndarray = np.exp(-0.5 * r2 / sigma**2)
    correction = np.exp(-2 * (np.pi * sigma * f) ** 2) * np.cos(phi)
    envelope = np.cos(2 * np.pi * f * change_base(x, y, theta)[0] + phi) - correction
    patch = gauss * envelope

    return patch


def weights(num_orients, num_phase, f, sigma, diameter, delta_x):
    dx = delta_x
    n = np.ceil(diameter / 2 / dx)
    x, y = np.mgrid[
        -n : n + 1,
        -n : n + 1,
    ]

    t = np.arange(num_orients) * np.pi / num_orients
    p = np.arange(num_phase) * 2 * np.pi / num_phase

    w = np.zeros((num_orients, num_phase, x.shape[0], x.shape[0]))
    for i in range(num_orients):
        theta = t[i]
        for j in range(num_phase):
            phase = p[j]

            w[i, j] = cos_gabor_function(
                x=x * dx, y=y * dx, theta=theta, f=f, sigma=sigma, phi=phase
            ).T

    return w


if __name__ == "__main__":
    f = 0.25  # frequency = 1/lambda = 1/4
    sigma = 2.0
    diameter = 10
    num_orients = 8
    num_phase = 4
    we = weights(
        num_orients=num_orients,
        num_phase=num_phase,
        f=f,
        sigma=sigma,
        diameter=diameter,
        delta_x=1,
    )

    # comment in for plotting as matrix :
    # fig = plt.figure(figsize=(5, 5))
    # for i in range(num_orients):
    #     for j in range(num_phase):
    #         plt.subplot(num_orients, num_phase, (i * num_phase) + j + 1)
    #         plt.imshow(we[i, j], cmap="gray", vmin=we.min(), vmax=we.max())
    #         plt.axis("off")
    #         # plt.colorbar()
    # plt.tight_layout()
    # plt.show(block=True)

    weights_flatten = np.ascontiguousarray(we)
    weights_flatten = np.reshape(
        weights_flatten, (we.shape[0] * we.shape[1], 1, we.shape[-2], we.shape[-1])
    )

    # comment in for saving
    # np.save("gabor_dict_32o_8p.npy", weights_flatten)
