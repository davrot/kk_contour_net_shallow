import torch
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["text.usetex"] = True
mpl.rcParams["font.family"] = "serif"


# path to NN
nn = "network_0_seed0_Coignless_83Epoch_2807-1455"
PATH = f"./trained_models/{nn}.pt"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# load and eval model
model = torch.load(PATH).to(device)
model.eval()
print("Full network:")
print(model)
print("")

# enter index to plot:
idx = int(input("Please select layer: "))
print(f"Selected layer {idx}:")
assert idx < len(model)
model = model[: idx + 1]

# random input
input_img = torch.randn(1, 200, 200).to(device)
input_img = input_img.unsqueeze(0)
input_img.requires_grad_(True)  # type: ignore

output_shape = model(input_img).shape
target_image = torch.zeros(
    (*output_shape,), dtype=input_img.dtype, device=input_img.device
)

input_parameter = torch.nn.Parameter(input_img)

# define parameters
num_iterations: int = 10000
learning_rate: float = 0.0005

print(
    (
        f"Available max positions: f:{target_image.shape[1]} "
        f"x:{target_image.shape[2]} y:{target_image.shape[3]}"
    )
)

# select neuron and plot for all feature maps (?)
neuron_f = int(input("Please select neuron_f: "))
neuron_x = target_image.shape[2] // 2
neuron_y = target_image.shape[3] // 2
print(f"Selected neuron {neuron_f}, {neuron_x}, {neuron_y}")


optimizer = torch.optim.Adam([{"params": input_parameter}], lr=learning_rate)
# TODO:
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

target_image[0, neuron_f, neuron_x, neuron_y] = 1e4

for i in range(num_iterations):
    optimizer.zero_grad()

    output = model(input_parameter)

    loss = torch.nn.functional.mse_loss(output, target_image)
    loss.backward()

    if i % 1000 == 0:
        print(f"{i} : loss={float(loss):.3e} lr={optimizer.param_groups[0]['lr']:.3e}")
    optimizer.step()
    # TODO:
    # scheduler.step(float(loss))


# plot image:
plt.imshow(input_img.squeeze().detach().cpu().numpy(), cmap="gray")
plt.show(block=True)
