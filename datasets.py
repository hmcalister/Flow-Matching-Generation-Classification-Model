from abc import abstractmethod
from typing import Iterator

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms


class BaseDataLoader:
    @abstractmethod
    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass


class JointDistributionLoader(BaseDataLoader):
    def __init__(self, base_loader: BaseDataLoader):
        self.base_loader = base_loader

    def __len__(self) -> int:
        return len(self.base_loader)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        for data in self.base_loader:
            x0_samples = data["p0_samples"]
            x1_samples = data["p1_samples"]
            y1_samples = data["class_labels"]

            batch_size, data_dimension = x0_samples.shape
            half_batch_size = batch_size // 2

            # First half of batch is (noise, label) -> (image, label)
            # Second half of batch is (image, noise) -> (image, label)
            x0_samples[half_batch_size:, :] = x1_samples[half_batch_size:]
            y0_samples = torch.randn_like(y1_samples)
            y0_samples[:half_batch_size, :] = y1_samples[:half_batch_size]

            data = {}
            data["x0_samples"] = x0_samples
            data["x1_samples"] = x1_samples
            data["y0_samples"] = y0_samples
            data["y1_samples"] = y1_samples
            yield data


def load_MNIST(
    batch_size: int,
    num_samples: int = 60_000,
    preload: bool = False,
    train: bool = True,
    shuffle: bool = True,
) -> BaseDataLoader:
    """
    Load, transform, and return the MNIST dataset as a DataLoader.
    MNIST has a shape (1, 28, 28).

    :param preload:
        If True, load all samples into a TensorDataset in one pass at startup.
        This avoids repeated per-sample transform and CPU->GPU transfer overhead
        during training. Recommended when the dataset fits in memory.

    :returns:
        A DataLoader yielding dict[str, torch.Tensor]
        Dictionary keys are ["p0_samples", "p1_samples", "class_labels",]
    """

    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize((0.0,), (1.0,)),
            transforms.Lambda(lambda x: x.flatten()),
        ]
    )
    dataset = torchvision.datasets.MNIST(
        "datasets/MNIST", download=True, transform=transform, train=train
    )
    if num_samples < len(dataset):
        dataset = torch.utils.data.Subset(dataset, range(num_samples))

    if preload:
        loader = DataLoader(dataset, batch_size=num_samples, shuffle=False)
        samples, labels = next(iter(loader))
        dataset = torch.utils.data.TensorDataset(samples, labels)

    torch_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    class MNISTDataLoader(BaseDataLoader):
        IMAGE_SHAPE: tuple[int, int, int] = (1, 28, 28)
        DATA_DIMENSION = int(torch.prod(torch.tensor(IMAGE_SHAPE)).item())
        BATCH_SIZE = batch_size

        def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
            for images, image_labels in torch_loader:
                prior_samples = torch.randn_like(images)
                image_labels = torch.nn.functional.one_hot(
                    image_labels, num_classes=10
                ).float()
                yield {
                    "p0_samples": prior_samples,
                    "p1_samples": images,
                    "class_labels": image_labels,
                }

        def __len__(self) -> int:
            return len(torch_loader)

    return MNISTDataLoader()
