# %%
#
# contours.py
#
# Tools for contour integration studies
#
# Version 2.0, 28.04.2023:
#   phases can now be randomized...
#

#
# Coordinate system assumptions:
#
# for arrays:
#   [..., HEIGHT, WIDTH], origin is on TOP LEFT
#   HEIGHT indices *decrease* with increasing y-coordinates (reversed)
#   WIDTH indices *increase* with increasing x-coordinates (normal)
#
# Orientations:
#   0 is horizontal, orientation *increase* counter-clockwise
#   Corner elements, quantified by [dir_source, dir_change]:
#       - consist of two legs
#       - contour *enters* corner from *source direction* at one leg
#         and goes from border to its center...
#       - contour path changes by *direction change* and goes
#         from center to the border
#

import torch
import time
import matplotlib.pyplot as plt
import math
import scipy.io
import numpy as np
import os

torch_device = "cuda"
default_dtype = torch.float32
torch.set_default_dtype(default_dtype)
torch.device(torch_device)


#
# performs a coordinate transform (rotation with phi around origin)
# rotation is performed CLOCKWISE with increasing phi
#
# remark: rotating a mesh grid by phi and orienting an image element
# along the new x-axis is EQUIVALENT to rotating the image element
# by -phi (so this realizes a rotation COUNTER-CLOCKWISE with
# increasing phi)
#
def rotate_CW(x: torch.Tensor, y: torch.Tensor, phi: torch.float32):
    xr = +x * torch.cos(phi) + y * torch.sin(phi)
    yr = -x * torch.sin(phi) + y * torch.cos(phi)

    return xr, yr


#
# renders a Gabor with (or without) corner
#
def gaborner(
    r_gab: int,  # radius, size will be 2*r_gab+1
    dir_source: float,  # contour enters in this dir
    dir_change: float,  # contour turns around by this dir
    lambdah: float,  # wavelength of Gabor
    sigma: float,  # half-width of Gabor
    phase: float,  # phase of Gabor
    normalize: bool,  # normalize patch to zero
    torch_device: str,  # GPU or CPU...
) -> torch.Tensor:
    # incoming dir: change to outgoing dir
    dir1 = dir_source + torch.pi
    nook = dir_change - torch.pi

    # create coordinate grids
    d_gab = 2 * r_gab + 1
    x = -r_gab + torch.arange(d_gab, device=torch_device)
    yg, xg = torch.meshgrid(x, x, indexing="ij")

    # put into tensor for performing vectorized scalar products
    xyg = torch.zeros([d_gab, d_gab, 1, 2], device=torch_device)
    xyg[:, :, 0, 0] = xg
    xyg[:, :, 0, 1] = yg

    # create Gaussian hull
    gauss = torch.exp(-(xg**2 + yg**2) / 2 / sigma**2)
    gabor_corner = gauss.clone()

    if (dir_change == 0) or (dir_change == torch.pi):
        # handle special case of straight Gabor or change by 180 deg

        # vector orth to Gabor axis
        ev1_orth = torch.tensor(
            [math.cos(-dir1 + math.pi / 2), math.sin(-dir1 + math.pi / 2)],
            device=torch_device,
        )
        # project coords to orth vector to get distance
        legs = torch.cos(
            2
            * torch.pi
            * torch.matmul(xyg, ev1_orth.unsqueeze(1).unsqueeze(0).unsqueeze(0))
            / lambdah
            + phase
        )
        gabor_corner *= legs[:, :, 0, 0]

    else:
        dir2 = dir1 + nook

        # compute separation line between corner's legs
        ev1 = torch.tensor([math.cos(-dir1), math.sin(-dir1)], device=torch_device)
        ev2 = torch.tensor([math.cos(-dir2), math.sin(-dir2)], device=torch_device)
        v_towards_1 = (ev1 - ev2).unsqueeze(1).unsqueeze(0).unsqueeze(0)

        # which coords belong to which leg?
        which_side = torch.matmul(xyg, v_towards_1)[:, :, 0, 0]
        towards_1y, towards_1x = torch.where(which_side > 0)
        towards_2y, towards_2x = torch.where(which_side <= 0)

        # compute orth distance to legs
        side_sign = -1 + 2 * ((dir_change % 2 * torch.pi) > torch.pi)
        ev12 = ev1 + ev2
        v1_orth = ev12 - ev1 * torch.matmul(ev12, ev1)
        v2_orth = ev12 - ev2 * torch.matmul(ev12, ev2)
        ev1_orth = side_sign * v1_orth / torch.sqrt((v1_orth**2).sum())
        ev2_orth = side_sign * v2_orth / torch.sqrt((v2_orth**2).sum())

        leg1 = torch.cos(
            2
            * torch.pi
            * torch.matmul(xyg, ev1_orth.unsqueeze(1).unsqueeze(0).unsqueeze(0))
            / lambdah
            + phase
        )
        leg2 = torch.cos(
            2
            * torch.pi
            * torch.matmul(xyg, ev2_orth.unsqueeze(1).unsqueeze(0).unsqueeze(0))
            / lambdah
            + phase
        )
        gabor_corner[towards_1y, towards_1x] *= leg1[towards_1y, towards_1x, 0, 0]
        gabor_corner[towards_2y, towards_2x] *= leg2[towards_2y, towards_2x, 0, 0]

    # depending on phase, Gabor might not be normalized...
    if normalize:
        s = gabor_corner.sum()
        s0 = gauss.sum()
        gabor_corner -= s / s0 * gauss

    return gabor_corner


#
# creates a filter bank of Gabor corners
#
# outputs:
#   filters: [n_source, n_change, HEIGHT, WIDTH]
#   dirs_source: [n_source]
#   dirs_change: [n_change]
#
def gaborner_filterbank(
    r_gab: int,  # radius, size will be 2*r_gab+1
    n_source: int,  # number of source orientations
    n_change: int,  # number of direction changes
    lambdah: float,  # wavelength of Gabor
    sigma: float,  # half-width of Gabor
    phase: float,  # phase of Gabor
    normalize: bool,  # normalize patch to zero
    torch_device: str,  # GPU or CPU...
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    kernels = torch.zeros(
        [n_source, n_change, 2 * r_gab + 1, 2 * r_gab + 1],
        device=torch_device,
        requires_grad=False,
    )
    dirs_source = 2 * torch.pi * torch.arange(n_source, device=torch_device) / n_source
    dirs_change = 2 * torch.pi * torch.arange(n_change, device=torch_device) / n_change

    for i_source in range(n_source):
        for i_change in range(n_change):
            gabor_corner = gaborner(
                r_gab=r_gab,
                dir_source=dirs_source[i_source],
                dir_change=dirs_change[i_change],
                lambdah=lambdah,
                sigma=sigma,
                phase=phase,
                normalize=normalize,
                torch_device=torch_device,
            )
            kernels[i_source, i_change] = gabor_corner

            # check = torch.isnan(gabor_corner).sum()
            # if check > 0:
            #     print(i_source, i_change, check)
            #     kernels[i_source, i_change] = 1

    return kernels, dirs_source, dirs_change


def discretize_stimuli(
    posori,
    x_range: tuple,
    y_range: tuple,
    scale_factor: float,
    r_gab_PIX: int,
    n_source: int,
    n_change: int,
    torch_device: str,
    n_phase: int = 1,
) -> torch.Tensor:
    # check correct input size
    s = posori.shape
    assert len(s) == 2, "posori should be NDARRAY with N x 1 entries"
    assert s[1] == 1, "posori should be NDARRAY with N x 1 entries"

    # determine size of (extended) canvas
    x_canvas_PIX = torch.tensor(
        (x_range[1] - x_range[0]) * scale_factor, device=torch_device
    ).ceil()
    y_canvas_PIX = torch.tensor(
        (y_range[1] - y_range[0]) * scale_factor, device=torch_device
    ).ceil()
    x_canvas_ext_PIX = int(x_canvas_PIX + 2 * r_gab_PIX)
    y_canvas_ext_PIX = int(y_canvas_PIX + 2 * r_gab_PIX)

    # get number of contours
    n_contours = s[0]
    index_phasrcchg = []
    index_y = []
    index_x = []
    for i_contour in range(n_contours):
        x_y_src_chg = torch.asarray(posori[i_contour, 0][1:, :].copy())
        x_y_src_chg[2] += torch.pi

        # if i_contour == 0:
        #     print(x_y_src_chg[2][:3])

        # compute integer coordinates and find all visible elements
        x = ((x_y_src_chg[0] - x_range[0]) * scale_factor + r_gab_PIX).type(torch.long)
        y = y_canvas_ext_PIX - (
            (x_y_src_chg[1] - y_range[0]) * scale_factor + r_gab_PIX
        ).type(torch.long)
        i_visible = torch.where(
            (x >= 0) * (y >= 0) * (x < x_canvas_ext_PIX) * (y < y_canvas_ext_PIX)
        )[0]

        # compute integer (changes of) directions
        i_source = (
            ((((x_y_src_chg[2]) / (2 * torch.pi)) + 1 / (2 * n_source)) % 1) * n_source
        ).type(torch.long)
        i_change = (
            (((x_y_src_chg[3] / (2 * torch.pi)) + 1 / (2 * n_change)) % 1) * n_change
        ).type(torch.long)

        i_phase = torch.randint(n_phase, i_visible.size())
        index_phasrcchg.append(
            (i_phase * n_source + i_source[i_visible]) * n_change + i_change[i_visible]
        )
        # index_change.append(i_change[i_visible])
        index_y.append(y[i_visible])
        index_x.append(x[i_visible])

    return (
        index_phasrcchg,
        index_x,
        index_y,
        x_canvas_ext_PIX,
        y_canvas_ext_PIX,
    )


def render_stimulus(
    kernels, index_element, index_y, index_x, y_canvas, x_canvas, torch_device
):
    s = kernels.shape
    kx = s[-1]
    ky = s[-2]

    stimulus = torch.zeros((y_canvas + ky - 1, x_canvas + kx - 1), device=torch_device)
    n = index_element.size()[0]
    for i in torch.arange(n, device=torch_device):
        x = index_x[i]
        y = index_y[i]
        stimulus[y : y + ky, x : x + kx] += kernels[index_element[i]]

    return stimulus[ky - 1 : -(ky - 1), kx - 1 : -(kx - 1)]


if __name__ == "__main__":
    VERBOSE = True
    BENCH_CONVOLVE = True
    BENCH_GPU = True
    BENCH_CPU = True
    BENCH_DAVID = True

    print("Testing contour rendering speed:")
    print("================================")

    # load contours, multiplex coordinates to simulate a larger set of contours
    n_multiplex = 1
    mat = scipy.io.loadmat("z.mat")
    posori = np.tile(mat["z"], (n_multiplex, 1))
    n_contours = posori.shape[0]
    print(f"Processing {n_contours} contour stimuli")

    # how many contours to render simultaneously?
    n_simultaneous = 5
    n_simultaneous_chunks, n_remaining = divmod(n_contours, n_simultaneous)
    assert n_remaining == 0, "Check parameters for simultaneous contour rendering!"

    # repeat some times for speed testing
    n_repeat = 10
    t_dis = torch.zeros((n_repeat + 2), device=torch_device)
    t_con = torch.zeros((n_repeat + 2), device=torch_device)
    t_rsg = torch.zeros((n_repeat + 2), device=torch_device)
    t_rsc = torch.zeros((n_repeat + 2), device="cpu")
    t_rsd = torch.zeros((n_repeat + 2), device="cpu")

    # cutout for stimuli, and gabor parameters
    x_range = [140, 940]
    y_range = [140, 940]
    d_gab = 40
    lambdah = 12
    sigma = 8
    phase = 0.0
    normalize = True

    # scale to convert coordinates to pixel values
    scale_factor = 0.25

    # number of directions for dictionary
    n_source = 32
    n_change = 32

    # convert sizes to pixel units
    lambdah_PIX = lambdah * scale_factor
    sigma_PIX = sigma * scale_factor
    r_gab_PIX = int(d_gab * scale_factor / 2)
    d_gab_PIX = r_gab_PIX * 2 + 1

    # make filterbank
    kernels, dirs_source, dirs_change = gaborner_filterbank(
        r_gab=r_gab_PIX,
        n_source=n_source,
        n_change=n_change,
        lambdah=lambdah_PIX,
        sigma=sigma_PIX,
        phase=phase,
        normalize=normalize,
        torch_device=torch_device,
    )
    kernels = kernels.reshape([1, n_source * n_change, d_gab_PIX, d_gab_PIX])
    kernels_flip = kernels.flip(dims=(-1, -2))

    # define "network" and put to cuda
    conv = torch.nn.Conv2d(
        in_channels=n_source * n_change,
        out_channels=1,
        kernel_size=d_gab_PIX,
        stride=1,
        device=torch_device,
    )
    conv.weight.data = kernels_flip

    print("Discretizing START!!!")
    t_dis[0] = time.perf_counter()
    for i_rep in range(n_repeat):
        # discretize
        (
            index_srcchg,
            index_x,
            index_y,
            x_canvas,
            y_canvas,
        ) = discretize_stimuli(
            posori=posori,
            x_range=x_range,
            y_range=y_range,
            scale_factor=scale_factor,
            r_gab_PIX=r_gab_PIX,
            n_source=n_source,
            n_change=n_change,
            torch_device=torch_device,
        )
        t_dis[i_rep + 1] = time.perf_counter()
    t_dis[-1] = time.perf_counter()
    print("Discretizing END!!!")

    if BENCH_CONVOLVE:
        print("Allocating!")
        stimuli = torch.zeros(
            [n_simultaneous, n_source * n_change, y_canvas, x_canvas],
            device=torch_device,
            requires_grad=False,
        )

        print("Generation by CONVOLUTION start!")
        t_con[0] = time.perf_counter()
        for i_rep in torch.arange(n_repeat):
            for i_simultaneous_chunks in torch.arange(n_simultaneous_chunks):
                i_ofs = i_simultaneous_chunks * n_simultaneous

                for i_sim in torch.arange(n_simultaneous):
                    stimuli[
                        i_sim,
                        index_srcchg[i_sim + i_ofs],
                        index_y[i_sim + i_ofs],
                        index_x[i_sim + i_ofs],
                    ] = 1

                output = conv(stimuli)

                for i_sim in range(n_simultaneous):
                    stimuli[
                        i_sim,
                        index_srcchg[i_sim + i_ofs],
                        index_y[i_sim + i_ofs],
                        index_x[i_sim + i_ofs],
                    ] = 0

            t_con[i_rep + 1] = time.perf_counter()
        t_con[-1] = time.perf_counter()
        print("Generation by CONVOLUTION stop!")

    if BENCH_GPU:
        print("Generation by GPU start!")
        output_gpu = torch.zeros(
            (
                n_contours,
                y_canvas - d_gab_PIX + 1,
                x_canvas - d_gab_PIX + 1,
            ),
            device=torch_device,
        )
        t_rsg[0] = time.perf_counter()
        for i_rep in torch.arange(n_repeat):
            for i_con in torch.arange(n_contours):
                output_gpu[i_con] = render_stimulus(
                    kernels=kernels[0],
                    index_element=index_srcchg[i_con],
                    index_y=index_y[i_con],
                    index_x=index_x[i_con],
                    y_canvas=y_canvas,
                    x_canvas=x_canvas,
                    torch_device=torch_device,
                )
            # output_gpu = torch.clip(output_gpu, -1, +1)

            t_rsg[i_rep + 1] = time.perf_counter()
        t_rsg[-1] = time.perf_counter()
        print("Generation by GPU stop!")

    if BENCH_CPU:
        print("Generation by CPU start!")
        output_cpu = torch.zeros(
            (
                n_contours,
                y_canvas - d_gab_PIX + 1,
                x_canvas - d_gab_PIX + 1,
            ),
            device="cpu",
        )
        kernels_cpu = kernels.detach().cpu()
        t_rsc[0] = time.perf_counter()
        for i_rep in range(n_repeat):
            for i_con in range(n_contours):
                output_cpu[i_con] = render_stimulus(
                    kernels=kernels_cpu[0],
                    index_element=index_srcchg[i_con],
                    index_y=index_y[i_con],
                    index_x=index_x[i_con],
                    y_canvas=y_canvas,
                    x_canvas=x_canvas,
                    torch_device="cpu",
                )
            # output_cpu = torch.clip(output_cpu, -1, +1)

            t_rsc[i_rep + 1] = time.perf_counter()
        t_rsc[-1] = time.perf_counter()
        print("Generation by CPU stop!")

    if BENCH_DAVID:
        print("Generation by DAVID start!")
        from CPPExtensions.PyTCopyCPU import TCopyCPU as render_stimulus_CPP

        copyier = render_stimulus_CPP()

        number_of_cpu_processes = os.cpu_count()
        output_dav_tmp = torch.zeros(
            (
                n_contours,
                y_canvas + 2 * r_gab_PIX,
                x_canvas + 2 * r_gab_PIX,
            ),
            device="cpu",
            dtype=torch.float,
        )
        gabor = kernels[0].detach().cpu()

        # Umsort!
        n_elements_total = 0
        for i_con in range(n_contours):
            n_elements_total += len(index_x[i_con])
        sparse_matrix = torch.zeros(
            (n_elements_total, 4), device="cpu", dtype=torch.int64
        )
        i_elements_total = 0
        for i_con in range(n_contours):
            n_add = len(index_x[i_con])
            sparse_matrix[i_elements_total : i_elements_total + n_add, 0] = i_con
            sparse_matrix[
                i_elements_total : i_elements_total + n_add, 1
            ] = index_srcchg[i_con]
            sparse_matrix[i_elements_total : i_elements_total + n_add, 2] = index_y[
                i_con
            ]
            sparse_matrix[i_elements_total : i_elements_total + n_add, 3] = index_x[
                i_con
            ]
            i_elements_total += n_add
        assert i_elements_total == n_elements_total, "UNBEHAGEN macht sich breit!"

        t_dav = torch.zeros((n_repeat + 2), device="cpu")
        t_dav[0] = time.perf_counter()
        for i_rep in range(n_repeat):
            output_dav_tmp.fill_(0.0)
            copyier.process(
                sparse_matrix.data_ptr(),
                int(sparse_matrix.shape[0]),
                int(sparse_matrix.shape[1]),
                gabor.data_ptr(),
                int(gabor.shape[0]),
                int(gabor.shape[1]),
                int(gabor.shape[2]),
                output_dav_tmp.data_ptr(),
                int(output_dav_tmp.shape[0]),
                int(output_dav_tmp.shape[1]),
                int(output_dav_tmp.shape[2]),
                int(number_of_cpu_processes),
            )
            output_dav = output_dav_tmp[
                :,
                d_gab_PIX - 1 : -(d_gab_PIX - 1),
                d_gab_PIX - 1 : -(d_gab_PIX - 1),
            ].clone()
            t_dav[i_rep + 1] = time.perf_counter()
        t_dav[-1] = time.perf_counter()
        print("Generation by DAVID done!")

    if VERBOSE:  # show last stimulus
        if BENCH_CONVOLVE:
            plt.subplot(2, 2, 1)
            plt.imshow(output[-1, 0].detach().cpu(), cmap="gray", vmin=-1, vmax=+1)
            plt.title("convolve")
        if BENCH_GPU:
            plt.subplot(2, 2, 2)
            plt.imshow(output_gpu[-1].detach().cpu(), cmap="gray", vmin=-1, vmax=+1)
            plt.title("gpu")
        if BENCH_CPU:
            plt.subplot(2, 2, 3)
            plt.imshow(output_cpu[-1], cmap="gray", vmin=-1, vmax=+1)
            plt.title("cpu")
        if BENCH_DAVID:
            plt.subplot(2, 2, 4)
            plt.imshow(output_dav[-1], cmap="gray", vmin=-1, vmax=+1)
            plt.title("david")
        plt.show()

    dt_discretize = t_dis.diff() / n_contours
    plt.plot(dt_discretize.detach().cpu())
    dt_convolve = t_con.diff() / n_contours
    plt.plot(dt_convolve.detach().cpu())
    dt_gpu = t_rsg.diff() / n_contours
    plt.plot(dt_gpu.detach().cpu())
    dt_cpu = t_rsc.diff() / n_contours
    plt.plot(dt_cpu.detach().cpu())
    dt_david = t_dav.diff() / n_contours
    plt.plot(dt_david.detach().cpu())

    plt.legend(["discretize", "convolve", "gpu", "cpu", "david"])
    plt.show()
    print(
        f"Average discretize for 1k stims: {1000*dt_discretize[:-1].detach().cpu().mean()} secs."
    )
    print(
        f"Average convolve for 1k stims: {1000*dt_convolve[:-1].detach().cpu().mean()} secs."
    )
    print(f"Average gpu for 1k stims: {1000*dt_gpu[:-1].detach().cpu().mean()} secs.")
    print(f"Average cpu for 1k stims: {1000*dt_cpu[:-1].detach().cpu().mean()} secs.")
    print(
        f"Average david for 1k stims: {1000*dt_david[:-1].detach().cpu().mean()} secs."
    )

    if BENCH_GPU and BENCH_CPU and BENCH_DAVID:
        df1 = (torch.abs(output_gpu[-1].detach().cpu() - output_cpu[-1])).mean()
        df2 = (torch.abs(output_gpu[-1].detach().cpu() - output_dav[-1])).mean()
        df3 = (torch.abs(output_dav[-1].cpu() - output_cpu[-1])).mean()
        print(f"Differences: CPU-GPU:{df1}, GPU-David:{df2}, David-CPU:{df3}")

    # %%
