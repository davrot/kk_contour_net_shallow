import torch
import logging


def train(
    model: torch.nn.modules.container.Sequential,
    loader: torch.utils.data.dataloader.DataLoader,
    optimizer: torch.optim.Adam | torch.optim.SGD,
    epoch: int,
    device: torch.device,
    tb,
    test_acc,
    logger: logging.Logger,
    train_accuracy: list[float],
    train_losses: list[float],
    train_loss: list[float],
    scale_data: float,
) -> float:
    num_train_pattern: int = 0
    running_loss: float = 0.0
    correct: int = 0
    pattern_count: float = 0.0

    model.train()
    for data in loader:
        label = data[0].to(device)
        image = data[1].type(dtype=torch.float32).to(device)
        if scale_data > 0:
            image /= scale_data

        optimizer.zero_grad()
        output = model(image)
        loss = torch.nn.functional.cross_entropy(output, label, reduction="sum")
        loss.backward()

        optimizer.step()

        # for loss and accuracy plotting:
        num_train_pattern += int(label.shape[0])
        pattern_count += float(label.shape[0])
        running_loss += float(loss)
        train_loss.append(float(loss))
        prediction = output.argmax(dim=1)
        correct += prediction.eq(label).sum().item()

        total_number_of_pattern: int = int(len(loader)) * int(label.shape[0])

        # infos:
        logger.info(
            (
                "Train Epoch:"
                f" {epoch}"
                f" [{int(pattern_count)}/{total_number_of_pattern}"
                f" ({100.0 * pattern_count / total_number_of_pattern:.2f}%)],"
                f" Loss: {float(running_loss) / float(num_train_pattern):.4e},"
                f" Acc: {(100.0 * correct / num_train_pattern):.2f}"
                f" Test Acc: {test_acc:.2f}%,"
                f" LR: {optimizer.param_groups[0]['lr']:.2e}"
            )
        )

    acc = 100.0 * correct / num_train_pattern
    train_accuracy.append(acc)

    epoch_loss = running_loss / pattern_count
    train_losses.append(epoch_loss)

    # add to tb:
    tb.add_scalar("Train Loss", loss.item(), epoch)
    tb.add_scalar("Train Performance", torch.tensor(acc), epoch)
    tb.add_scalar("Train Number Correct", torch.tensor(correct), epoch)

    # for parameters:
    for name, param in model.named_parameters():
        if "weight" in name or "bias" in name:
            tb.add_histogram(f"{name}", param.data.clone(), epoch)

    tb.flush()

    return epoch_loss
