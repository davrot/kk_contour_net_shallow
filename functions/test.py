import torch
import logging


@torch.no_grad()
def test(
    model: torch.nn.modules.container.Sequential,
    loader: torch.utils.data.dataloader.DataLoader,
    device: torch.device,
    tb,
    epoch: int,
    logger: logging.Logger,
    test_accuracy: list[float],
    test_losses: list[float],
    scale_data: float,
) -> float:
    test_loss: float = 0.0
    correct: int = 0
    pattern_count: float = 0.0

    model.eval()

    for data in loader:
        label = data[0].to(device)
        image = data[1].type(dtype=torch.float32).to(device)
        if scale_data > 0:
            image /= scale_data

        output = model(image)
        if output.ndim == 4:
            output = output.squeeze(-1).squeeze(-1)
        assert output.ndim == 2

        # loss and optimization
        loss = torch.nn.functional.cross_entropy(output, label, reduction="sum")
        pattern_count += float(label.shape[0])
        test_loss += loss.item()
        prediction = output.argmax(dim=1)
        correct += prediction.eq(label).sum().item()

    logger.info(
        (
            "Test set:"
            f" Average loss: {test_loss / pattern_count:.3e},"
            f" Accuracy: {correct}/{pattern_count},"
            f"({100.0 * correct / pattern_count:.2f}%)"
        )
    )
    logger.info("")

    acc = 100.0 * correct / pattern_count
    test_losses.append(test_loss / pattern_count)
    test_accuracy.append(acc)

    # add to tb:
    tb.add_scalar("Test Loss", (test_loss / pattern_count), epoch)
    tb.add_scalar("Test Performance", 100.0 * correct / pattern_count, epoch)
    tb.add_scalar("Test Number Correct", correct, epoch)
    tb.flush()

    return acc
