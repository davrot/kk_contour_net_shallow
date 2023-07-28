import numpy as np
import time
import matplotlib.pyplot as plt
import os

file = "/home/kk/Documents/Semester4/code/RenderStimuli/Output/Coignless/base_angle_090_b001_n10000_RENDERED.npz"
t0 = time.perf_counter()
with np.load(file) as data:
    a = data["gaborfield"]
t1 = time.perf_counter()
print(t1 - t0)

for i in range(5):
    plt.imshow(a[i], cmap="gray", vmin=0, vmax=255)
    plt.savefig(
        os.path.join(
            "./poster",
            "classic_90_stimulus.pdf",
        ),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show(block=True)

