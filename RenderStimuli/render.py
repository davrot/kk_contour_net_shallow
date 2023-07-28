# %%

import torch
import time
import scipy
import os
import matplotlib.pyplot as plt
import numpy as np
import contours
import glob

USE_CEXT_FROM_DAVID = True
if USE_CEXT_FROM_DAVID:
    #    from CPPExtensions.PyTCopyCPU import TCopyCPU
    from CPPExtensions.PyTCopyCPU import TCopyCPU as render_stimulus_CPP


def render_gaborfield(posori, params, verbose=False):
    scale_factor = params["scale_factor"]
    n_source = params["n_source"]
    n_change = params["n_change"]
    n_phase = params["n_phase"]

    # convert sizes to pixel units
    lambda_PIX = params["lambda_gabor"] * scale_factor
    sigma_PIX = params["sigma_gabor"] * scale_factor
    r_gab_PIX = int(params["d_gabor"] * scale_factor / 2)
    d_gab_PIX = r_gab_PIX * 2 + 1

    # make filterbank
    gabors = torch.zeros(
        [n_phase, n_source, n_change, d_gab_PIX, d_gab_PIX], dtype=torch.float32
    )
    for i_phase in range(n_phase):
        phase = (torch.pi * 2 * i_phase) / n_phase
        gabors[i_phase], dirs_source, dirs_change = contours.gaborner_filterbank(
            r_gab=r_gab_PIX,
            n_source=n_source,
            n_change=n_change,
            lambdah=lambda_PIX,
            sigma=sigma_PIX,
            phase=phase,
            normalize=params["normalize_gabor"],
            torch_device="cpu",
        )
    gabors = gabors.reshape([n_phase * n_source * n_change, d_gab_PIX, d_gab_PIX])

    n_contours = posori.shape[0]

    # discretize ALL stimuli
    if verbose:
        print("Discretizing START!!!")
    t_dis0 = time.perf_counter()
    (
        index_srcchg,
        index_x,
        index_y,
        x_canvas,
        y_canvas,
    ) = contours.discretize_stimuli(
        posori=posori,
        x_range=params["x_range"],
        y_range=params["y_range"],
        scale_factor=scale_factor,
        r_gab_PIX=r_gab_PIX,
        n_source=n_source,
        n_change=n_change,
        n_phase=n_phase,
        torch_device="cpu",
    )
    t_dis1 = time.perf_counter()
    if verbose:
        print(f"Discretizing END, took {t_dis1-t_dis0} seconds.!!!")

    if verbose:
        print("Generation START!!!")
    t0 = time.perf_counter()

    if not USE_CEXT_FROM_DAVID:
        if verbose:
            print("   (using NUMPY...)")
        output = torch.zeros(
            (
                n_contours,
                y_canvas - d_gab_PIX + 1,
                x_canvas - d_gab_PIX + 1,
            ),
            device="cpu",
            dtype=torch.float32,
        )
        kernels_cpu = gabors.detach().cpu()
        for i_con in range(n_contours):
            output[i_con] = contours.render_stimulus(
                kernels=kernels_cpu,
                index_element=index_srcchg[i_con],
                index_y=index_y[i_con],
                index_x=index_x[i_con],
                y_canvas=y_canvas,
                x_canvas=x_canvas,
                torch_device="cpu",
            )
        output = torch.clip(output, -1, +1)

    else:
        if verbose:
            print("   (using C++...)")
        copyier = render_stimulus_CPP()
        number_of_cpu_processes = os.cpu_count()
        output_dav_tmp = torch.zeros(
            (
                n_contours,
                y_canvas + 2 * r_gab_PIX,
                x_canvas + 2 * r_gab_PIX,
            ),
            device="cpu",
            dtype=torch.float32,
        )

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

        # output_dav_tmp.fill_(0.0)
        copyier.process(
            sparse_matrix.data_ptr(),
            int(sparse_matrix.shape[0]),
            int(sparse_matrix.shape[1]),
            gabors.data_ptr(),
            int(gabors.shape[0]),
            int(gabors.shape[1]),
            int(gabors.shape[2]),
            output_dav_tmp.data_ptr(),
            int(output_dav_tmp.shape[0]),
            int(output_dav_tmp.shape[1]),
            int(output_dav_tmp.shape[2]),
            int(number_of_cpu_processes),
        )
        output = torch.clip(
            output_dav_tmp[
                :,
                d_gab_PIX - 1 : -(d_gab_PIX - 1),
                d_gab_PIX - 1 : -(d_gab_PIX - 1),
            ],
            -1,
            +1,
        )

    t1 = time.perf_counter()
    if verbose:
        print(f"Generating END, took {t1-t0} seconds.!!!")

    if verbose:
        print("Showing first and last stimulus generated...")
        plt.imshow(output[0], cmap="gray", vmin=-1, vmax=+1)
        plt.show()
        plt.imshow(output[-1], cmap="gray", vmin=-1, vmax=+1)
        plt.show()
        print(f"Processed {n_contours} stimuli in {t1-t_dis0} seconds!")

    return output


def render_gaborfield_frommatfiles(files, params, varname, varname_dist, altpath=None, verbose=False):
    n_total = 0
    n_files = len(files)
    print(f"Going through {n_files} contour files...")

    for i_file in range(n_files):
        # get path, basename, suffix...
        full = files[i_file]
        path, file = os.path.split(full)
        base, suffix = os.path.splitext(file)

        # load file
        mat = scipy.io.loadmat(full)
        # posori = mat[varname]
        if "dist" in full:
            posori = mat[varname_dist]
        else:
            posori = mat[varname]
        n_contours = posori.shape[0]
        n_total += n_contours
        print(f"   ...file {file} contains {n_contours} contours.")

        # process...
        gaborfield = render_gaborfield(posori, params=params, verbose=verbose)

        # save
        if altpath:
            savepath = altpath
        else:
            savepath = path
        savefull = savepath + os.sep + base + "_RENDERED.npz"
        print(f"   ...saving under {savefull}...")
        gaborfield = (torch.clip(gaborfield, -1, 1) * 127 + 128).type(torch.uint8)
        np.savez_compressed(savefull, gaborfield=gaborfield)

    return n_total


if __name__ == "__main__":
    TESTMODE = "files"  # "files" or "posori"

    # cutout for stimuli, and gabor parameters
    params = {
        "x_range": [140, 940],
        "y_range": [140, 940],
        "scale_factor": 0.25,  # scale to convert coordinates to pixel values
        "d_gabor": 40,
        "lambda_gabor": 16,
        "sigma_gabor": 8,
        "n_phase": 4,
        "normalize_gabor": True,
        # number of directions for dictionary
        "n_source": 32,
        "n_change": 32,
    }

    if TESTMODE == "files":
        num = int(9)
        path = f"/data_1/kk/StimulusGeneration/Alicorn/Coignless/Base0{num}0_n100"
        files = glob.glob(path + os.sep + "*.mat")

        t0 = time.perf_counter()
        n_total = render_gaborfield_frommatfiles(
            files=files, params=params, varname=f"Table_base_crn0{num}0", varname_dist=f"Table_base_crn0{num}0_dist", altpath="./Output100/Coignless"  #intr crn base 
        )
        t1 = time.perf_counter()
        dt = t1 - t0
        print(
            f"Rendered {n_total} contours in {dt} secs, yielding {n_total/dt} contours/sec."
        )

    if TESTMODE == "posori":
        print("Sample stimulus generation:")
        print("===========================")

        # load contours, multiplex coordinates to simulate a larger set of contours
        n_multiplex = 500
        mat = scipy.io.loadmat("z.mat")
        posori = np.tile(mat["z"], (n_multiplex, 1))
        n_contours = posori.shape[0]
        print(f"Processing {n_contours} contour stimuli")

        output = render_gaborfield(posori, params=params, verbose=True)
        # output8 = (torch.clip(output, -1, 1) * 127 + 128).type(torch.uint8)
        # np.savez_compressed("output8_compressed.npz", output8=output8)


# %%
