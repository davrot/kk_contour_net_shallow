import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
mpl.rcParams["text.usetex"] = True
mpl.rcParams["font.family"] = "serif"

def plot_weights(
    plot,
    s,
    grid_color,
    linewidth,
    idx,
    smallDim,
    swap_channels,
    activations,
    layer,
    title,
    colorbar,
    vmin,
    vmax,
):
    plt.imshow(plot.T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)

    ax = plt.gca()
    a = np.arange(0, plot.shape[1] + 1, s[3])
    b = np.arange(0, plot.shape[0] + 1, s[1])
    plt.hlines(a - 0.5, -0.5, plot.shape[0] - 0.5, colors=grid_color, lw=linewidth)
    plt.vlines(b - 0.5, -0.5, plot.shape[1] - 0.5, colors=grid_color, lw=linewidth)
    plt.ylim(-1, plot.shape[1])
    plt.xlim(-1, plot.shape[0])

    ax.set_xticks(s[1] / 2 + np.arange(-0.5, plot.shape[0] - 1, s[1]))
    ax.set_yticks(s[3] / 2 + np.arange(-0.5, plot.shape[1] - 1, s[3]))

    if (
        idx is not None
        and (smallDim is False and swap_channels is False)
        or (activations is True)
    ):
        ax.set_xticklabels(idx, fontsize=15)
        ax.set_yticklabels(np.arange(s[2]), fontsize=15)
    elif idx is not None and layer == "FC1":
        ax.set_xticklabels(np.arange(s[0]), fontsize=15)
        ax.set_yticklabels(idx, fontsize=15)
    elif idx is not None and (smallDim is True or swap_channels is True):
        ax.set_xticklabels(np.arange(s[0]), fontsize=15)
        ax.set_yticklabels(idx, fontsize=15)
    else:
        ax.set_xticklabels(np.arange(s[0]), fontsize=15)
        ax.set_yticklabels(np.arange(s[2]), fontsize=15)
    ax.invert_yaxis()

    ax.xaxis.set_label_position("top")
    ax.tick_params(axis="x", top=True, bottom=False, labeltop=True, labelbottom=False)

    if title is not None:
        is_string = isinstance(title, str)
        if is_string is True:
            plt.title(title)

    if colorbar is True:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="1.5%", pad=0.05)
        cbar = plt.colorbar(ax.get_images()[0], cax=cax)
        tick_font_size = 14
        cbar.ax.tick_params(labelsize=tick_font_size)


def plot_in_grid(
    plot,
    fig_size=(10, 10),
    swap_channels=False,
    title=None,
    idx=None,
    colorbar=False,
    vmin=None,
    vmax=None,
    grid_color="k",
    linewidth=0.75,
    savetitle=None,
    activations=False,
    layer=None,
    format="pdf",
    bias=None,
    plot_bias: bool = False,
):
    smallDim = False
    if plot.ndim < 4:
        smallDim = True
        plot = np.swapaxes(plot, 0, 1)
        plot = plot[:, :, np.newaxis, np.newaxis]
    if vmin is None and vmax is None:
        # plot_abs = np.amax(np.abs(plot))
        vmin = -(np.amax(np.abs(plot)))
        vmax = np.amax(np.abs(plot))

    if swap_channels is True:
        plot = np.swapaxes(plot, 0, 1)

    # print(plot.shape)
    plot = np.ascontiguousarray(np.moveaxis(plot, 1, 2))

    for j in range(plot.shape[2]):
        for i in range(plot.shape[0]):
            plot[(i - 1), :, (j - 1), :] = plot[(i - 1), :, (j - 1), :].T

    s = plot.shape
    plot = plot.reshape((s[0] * s[1], s[2] * s[3]))
    plt.figure(figsize=fig_size)

    if plot_bias and bias is not None:
        if swap_channels:
            # If axes are swapped, arrange the plots side by side
            plt.subplot(1, 2, 1)
            plot_weights(
                plot=plot,
                s=s,
                grid_color=grid_color,
                linewidth=linewidth,
                idx=idx,
                smallDim=smallDim,
                swap_channels=swap_channels,
                activations=activations,
                layer=layer,
                title=title,
                colorbar=colorbar,
                vmin=vmin,
                vmax=vmax,
            )

            plt.subplot(1, 2, 2)
            plt.plot(bias, np.arange(len(bias)))
            plt.ylim(len(bias) - 1, 0)
            plt.title("Bias")
            plt.tight_layout()
            
        else:
            plt.subplot(2, 1, 1)
            plot_weights(
                plot=plot,
                s=s,
                grid_color=grid_color,
                linewidth=linewidth,
                idx=idx,
                smallDim=smallDim,
                swap_channels=swap_channels,
                activations=activations,
                layer=layer,
                title=title,
                colorbar=colorbar,
                vmin=vmin,
                vmax=vmax,
            )

            plt.subplot(2, 1, 2)
            plt.plot(np.arange(len(bias)), bias)
            plt.title("Bias")


    else:
        plot_weights(
            plot=plot,
            s=s,
            grid_color=grid_color,
            linewidth=linewidth,
            idx=idx,
            smallDim=smallDim,
            swap_channels=swap_channels,
            activations=activations,
            layer=layer,
            title=title,
            colorbar=colorbar,
            vmin=vmin,
            vmax=vmax,
        )

    if savetitle is not None:
        plt.savefig(f"plot_as_grid/{savetitle}.{format}")

    plt.tight_layout()
    plt.show(block=True)
