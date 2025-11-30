import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode


class ImageData(Dataset):
    def __init__(
        self, labels, filenames, photo_directory, width=256, height=256, transform=None
    ) -> None:
        super().__init__()

        self.width = width
        self.height = height
        self.transform = transform
        self.labels = labels
        self.filenames = filenames
        self.photo_directory = photo_directory

    def __getitem__(self, index):
        img = read_image(
            f"{self.photo_directory}/{self.filenames[index]}.jpg",
            mode=ImageReadMode.RGB,
        )

        if self.transform is not None:
            img = self.transform(img)

        label = torch.tensor(self.labels[index], dtype=torch.long)
        label_onehot = nn.functional.one_hot(label, num_classes=3).to(torch.float32)
        return img, label_onehot

    def __len__(self):
        return len(self.labels)
