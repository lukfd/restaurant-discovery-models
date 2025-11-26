# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: vision-ai-project
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Basic CNN model
#

# %%
# NOTE: I didn't have much time so threw this together in pytorch, we can refactor if necessary
import os
import time
from datetime import datetime

import numpy as np
import pyarrow as pa
import torch
from pyarrow import parquet
from sklearn.metrics import classification_report
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

from dataset import ImageData

photo_directory = "data/clean/reduced_photos"

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)

# %%
photos_scores = parquet.read_table("data/clean/model_data.parquet")
photos_scores = photos_scores.add_column(
    0,
    "star_category",
    pa.array(
        photos_scores.select(["stars"])
        .to_pandas()
        # NOTE: just trying to slightly populate groups
        .stars.apply(lambda r: (0.0 if r <= 3 else 1.0) if r <= 4 else 2.0)
    ),
).take(list(range(10_000)))
_, class_counts = np.unique(
    photos_scores.select(["star_category"]).to_pandas().values, return_counts=True
)
weights = [(class_counts.sum() - x) / (x + 1e-5) for x in iter(class_counts)]


# %%
yelp_photos = ImageData(
    photos_scores.select(["star_category"]).to_pandas().star_category,
    photos_scores.select(["photo_id"]).to_pandas().photo_id,
    photo_directory,
)
# NOTE: change batch size for actual leanring

train, val, test = torch.utils.data.random_split(yelp_photos, [0.5, 0.3, 0.2])
training_loader = DataLoader(
    train, batch_size=2000, shuffle=True, num_workers=4, persistent_workers=True
)
test_loader = DataLoader(test, batch_size=2000, num_workers=4)
validation_loader = DataLoader(val, batch_size=2000, num_workers=4)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # TODO: convert to VAE?
        self.convolutional = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.lin = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.convolutional(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return self.lin(x)


epoch_number = 1
best_vloss = 10000
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
root_dir = f"models/basic_cnn/{timestamp}"
os.makedirs(root_dir)
writer = SummaryWriter("logs/basic_cnn_{}".format(timestamp))
basic_cnn = CNN().to(device)
opt = torch.optim.Adam(basic_cnn.parameters(), lr=0.001)
loss_fn = torch.nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor(weights).to(torch.float32).to(device)
)
print(summary(basic_cnn))
basic_cnn(yelp_photos.__getitem__(1)[0].unsqueeze(0).to(device))


# %%
def get_loss(
    m: nn.Module, inputs: torch.Tensor, labels: torch.Tensor, lf: nn.Module
) -> torch.Tensor:
    inputs, labels = (
        inputs.to(device),
        labels.to(device).to(torch.float32).squeeze(),
    )
    outputs = m(inputs)
    loss = lf(outputs, labels)

    return loss


def train_one_epoch(m: nn.Module, loader, o, lf):
    running_loss = 0.0

    for i, (inputs, labels) in enumerate(loader):
        o.zero_grad()
        loss = get_loss(m, inputs, labels, lf)
        loss.backward()
        o.step()

        running_loss += loss.item()

        if i % 5 == 0:
            print(f"running loss: {running_loss / (i + 1)}")

    return m, running_loss / len(loader)


def score_model(m: nn.Module, loader, lf):
    running_loss = 0.0

    for _, (inputs, labels) in enumerate(loader):
        loss = get_loss(m, inputs, labels, lf)
        running_loss += loss.item()

    return running_loss / len(loader)


for epoch in range(901):
    print(f"EPOCH {epoch_number}")
    st = time.time()
    basic_cnn.train(True)
    basic_cnn, tloss = train_one_epoch(basic_cnn, training_loader, opt, loss_fn)

    basic_cnn.eval()
    with torch.no_grad():
        print(f"EVALUATING MODEL FOR EPOCH {epoch_number}")
        vloss = score_model(basic_cnn, validation_loader, loss_fn)

    print(f"LOSS train {round(tloss, 5)}, validation {round(vloss, 5)}")
    writer.add_scalars(
        "Training vs. Validation Loss",
        {"Training": tloss, "Validation": vloss},
        epoch_number + 1,
    )
    writer.flush()
    if vloss < best_vloss:
        best_vloss = vloss
        model_path = f"{root_dir}/epoch{epoch_number}_{round(vloss, 3)}"
        [os.remove(os.path.join(root_dir, f)) for f in os.listdir(root_dir)]
        torch.save(basic_cnn.state_dict(), model_path)

    epoch_number += 1
    print(f"EPOCH TIME: {time.time() - st}")

# %%
saved_models = os.listdir(root_dir)
best_model = CNN().to(device)
best_model.load_state_dict(
    torch.load(f"{root_dir}/{saved_models[0]}", weights_only=True)
)

all_labels = []
all_predictions = []

basic_cnn.eval()
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = best_model(inputs)
        all_labels.append(labels.argmax(dim=1).cpu().numpy())
        all_predictions.append(outputs.argmax(dim=1).cpu().numpy())

all_labels = np.concatenate(all_labels)
all_predictions = np.concatenate(all_predictions)
print(classification_report(all_labels, all_predictions))

with open(f"{root_dir}/{saved_models[0]}.txt", "w") as f:
    f.write(classification_report(all_labels, all_predictions))
