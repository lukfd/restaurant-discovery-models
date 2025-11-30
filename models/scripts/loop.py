import torch
from torch import nn
from torch.utils.data import DataLoader


def get_loss(
    m: nn.Module, inputs: torch.Tensor, labels: torch.Tensor, lf: nn.Module, device: str
) -> torch.Tensor:
    inputs, labels = (
        inputs.to(device),
        labels.to(device).to(torch.float32).squeeze(),
    )
    outputs = m(inputs)
    loss = lf(outputs, labels)

    return loss


def train_one_epoch(m: nn.Module, loader: DataLoader, o: torch.optim.Optimizer, lf, device: str):
    running_loss = 0.0

    for i, (inputs, labels) in enumerate(loader):
        o.zero_grad()
        loss = get_loss(m, inputs, labels, lf, device)
        loss.backward()
        o.step()

        running_loss += loss.item()
        del loss

        if (i + 1) % 10 == 0:
            print(f"{i + 1} / {len(loader)} -- running loss: {running_loss / (i + 1)}")

    return m, running_loss / len(loader)


def score_model(m: nn.Module, loader: DataLoader, lf: nn.Module, device: str):
    running_loss = 0.0

    for i, (inputs, labels) in enumerate(loader):
        loss = get_loss(m, inputs, labels, lf, device)
        running_loss += loss.item()

        if (i + 1) % 10 == 0:
            print(f"{i + 1} / {len(loader)} -- running loss: {running_loss / (i + 1)}")

    return running_loss / len(loader)
