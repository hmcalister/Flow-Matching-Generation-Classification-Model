import warnings
from abc import abstractmethod
from typing import Iterator

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms

_RANDOM_STANDARD_DEVIATION = 0.1


class ClassCodeManager:
    CLASS_CODES: torch.Tensor

    def __init__(
        self,
        num_classes: int,
        channel_height: int,
        channel_width: int,
        code_size: int | None,
        torch_device: str,
    ):
        """
        Manages random binary class codes embedded as a centered square patch
        within a (channel_height x channel_width) channel.

        :param num_classes: Number of classes.
        :param channel_height: Height of the class channel (e.g. IMAGE_SHAPE[1]).
        :param channel_width: Width of the class channel (e.g. IMAGE_SHAPE[2]).
        :param code_size: Side length of the centered square patch used for the
            class code. The active region is (code_size x code_size) pixels.
            Distances and classification are computed only over this patch.
            Defaults to min(channel_height, channel_width).
        :param torch_device: Torch device string.
        """
        self.num_classes = num_classes
        self.channel_height = channel_height
        self.channel_width = channel_width
        self.channel_size = channel_height * channel_width
        self.torch_device = torch_device

        # code_size is the side length of the centered square patch
        self.code_size = (
            code_size if code_size is not None else min(channel_height, channel_width)
        )
        assert self.code_size <= channel_height and self.code_size <= channel_width, (
            "code_size must be <= channel_height and channel_width"
        )

        # Build a 2D boolean mask for the centered code_size x code_size patch
        row_start = (channel_height - self.code_size) // 2
        col_start = (channel_width - self.code_size) // 2
        mask_2d = torch.zeros(channel_height, channel_width, dtype=torch.bool)
        mask_2d[
            row_start : row_start + self.code_size,
            col_start : col_start + self.code_size,
        ] = True
        self._patch_mask = mask_2d.flatten().to(torch_device)

        # Compact binary codes of shape (num_classes, code_size*code_size)
        self._CLASS_CODE_CENTERS = (
            torch.randn(
                (num_classes, self.code_size * self.code_size), device=torch_device
            )
            .sign()
            .float()
        )

        # Full-channel embeddings: zeros except in the active center patch
        self.CLASS_CODES = torch.zeros(
            (num_classes, self.channel_size), device=torch_device
        )
        self.CLASS_CODES[:, self._patch_mask] = self._CLASS_CODE_CENTERS

    def class_code_distances(self, prediction: torch.Tensor) -> torch.Tensor:
        """
        Accepts class prediction of shape (batch_size, channel_size) and returns
        (batch_size, num_classes) squared distances computed only over the active
        center patch.
        """
        active = prediction[:, self._patch_mask]
        return (
            (self._CLASS_CODE_CENTERS[None, :, :] - active[:, None, :])
            .square()
            .sum(dim=2)
        )

    def class_code_distribution(self, prediction: torch.Tensor) -> torch.Tensor:
        """
        Accepts class prediction of shape (batch_size, channel_size) and returns
        (batch_size, num_classes) softmin distribution over class codes.
        """

        distances = self.class_code_distances(prediction)
        return torch.nn.functional.softmin(distances, dim=1)


class BaseDataLoader:
    @abstractmethod
    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass


class BaseImageDataLoader(BaseDataLoader):
    IMAGE_SHAPE: tuple[int, int, int]
    DATA_DIMENSION: int
    CLASS_LABELS: list[str]


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
            y0_samples = _RANDOM_STANDARD_DEVIATION * torch.randn_like(y1_samples)
            y0_samples[:half_batch_size, :] = y1_samples[:half_batch_size]

            data = {}
            data["x0_samples"] = x0_samples
            data["x1_samples"] = x1_samples
            data["y0_samples"] = y0_samples
            data["y1_samples"] = y1_samples
            yield data


class OccludedImageLoader:
    def __init__(
        self,
        base_loader: JointDistributionLoader,
        image_shape: tuple[int, int, int],
        p_occlusion: float = 0.5,
        occlusion_scale_range: tuple[float, float] = (0.05, 0.33),
        occlusion_aspect_ratio_range: tuple[float, float] = (0.75, 1.5),
    ):
        self.base_loader = base_loader
        self.image_shape = image_shape
        self.random_erasing = transforms.RandomErasing(
            p=p_occlusion,
            scale=occlusion_scale_range,
            ratio=occlusion_aspect_ratio_range,
            value="random",  # pyright: ignore[reportArgumentType]
        )

    def __len__(self) -> int:
        return len(self.base_loader)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        for data in self.base_loader:
            x0_samples = data["x0_samples"]
            x1_samples = data["x1_samples"]
            y0_samples = data["y0_samples"]
            y1_samples = data["y1_samples"]

            image_to_image_mask = (x0_samples == x1_samples).all(dim=1)
            image_to_image_images = x0_samples[image_to_image_mask]

            occluded_list = [
                self.random_erasing(img.view(*self.image_shape))
                for img in image_to_image_images
            ]
            occluded_images = torch.stack(occluded_list).view(
                image_to_image_images.shape[0], -1
            )
            x0_samples[image_to_image_mask] = occluded_images
            data["x0_samples"] = x0_samples

            # Occluded images get their labels back
            occluded_image_mask = (x0_samples != x1_samples).any(dim=1)
            y0_samples[occluded_image_mask] = y1_samples[occluded_image_mask]

            yield data


def load_MNIST(
    batch_size: int,
    class_code_manager: ClassCodeManager,
    num_samples: int = 60_000,
    preload: bool = False,
    train: bool = True,
    shuffle: bool = True,
) -> BaseImageDataLoader:
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
        _loader = DataLoader(dataset, batch_size=num_samples, shuffle=shuffle)
        samples, labels = next(iter(_loader))
        dataset = torch.utils.data.TensorDataset(samples, labels)

    torch_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    class MNISTDataLoader(BaseImageDataLoader):
        IMAGE_SHAPE: tuple[int, int, int] = (1, 28, 28)
        DATA_DIMENSION = int(torch.prod(torch.tensor(IMAGE_SHAPE)).item())
        BATCH_SIZE = batch_size

        CLASS_LABELS = [
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
        ]

        CLASS_CODE_MANAGER = class_code_manager

        def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
            for images, image_labels in torch_loader:
                prior_samples = _RANDOM_STANDARD_DEVIATION * torch.randn_like(images)
                image_labels = self.CLASS_CODE_MANAGER.CLASS_CODES[image_labels]
                yield {
                    "p0_samples": prior_samples,
                    "p1_samples": images,
                    "class_labels": image_labels,
                }

        def __len__(self) -> int:
            return len(torch_loader)

    return MNISTDataLoader()


def load_CIFAR10(
    batch_size: int,
    class_code_manager: ClassCodeManager,
    num_samples: int = 50_000,
    preload: bool = False,
    train: bool = True,
    shuffle: bool = True,
) -> BaseImageDataLoader:
    """
    Load, transform, and return the CIFAR10 dataset as a DataLoader.
    MNIST has a shape (3, 32, 32).

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

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=np.exceptions.VisibleDeprecationWarning
        )
        dataset = torchvision.datasets.CIFAR10(
            "datasets/CIFAR10", download=True, transform=transform, train=train
        )
    if num_samples < len(dataset):
        dataset = torch.utils.data.Subset(dataset, range(num_samples))

    if preload:
        _loader = DataLoader(dataset, batch_size=num_samples, shuffle=shuffle)
        samples, labels = next(iter(_loader))
        dataset = torch.utils.data.TensorDataset(samples, labels)

    torch_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    class CIFAR10DataLoader(BaseImageDataLoader):
        IMAGE_SHAPE: tuple[int, int, int] = (3, 32, 32)
        DATA_DIMENSION = int(torch.prod(torch.tensor(IMAGE_SHAPE)).item())
        BATCH_SIZE = batch_size

        CLASS_LABELS = [
            "PLANE",
            "CAR",
            "BIRD",
            "CAT",
            "DEER",
            "DOG",
            "FROG",
            "HORSE",
            "SHIP",
            "TRUCK",
        ]

        CLASS_CODE_MANAGER = class_code_manager

        def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
            for images, image_labels in torch_loader:
                prior_samples = _RANDOM_STANDARD_DEVIATION * torch.randn_like(images)
                image_labels = self.CLASS_CODE_MANAGER.CLASS_CODES[image_labels]
                yield {
                    "p0_samples": prior_samples,
                    "p1_samples": images,
                    "class_labels": image_labels,
                }

        def __len__(self) -> int:
            return len(torch_loader)

    return CIFAR10DataLoader()
