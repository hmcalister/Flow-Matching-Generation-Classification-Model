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
            y0_samples = torch.randn_like(y1_samples)
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

            # occluded_images = self.random_erasing(
            #     image_to_image_images.view(-1, *self.image_shape)
            # ).view(image_to_image_images.shape[0], -1)
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

        def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
            for images, image_labels in torch_loader:
                prior_samples = torch.randn_like(images)
                image_labels = torch.nn.functional.one_hot(
                    image_labels, num_classes=len(self.CLASS_LABELS)
                ).float()
                yield {
                    "p0_samples": prior_samples,
                    "p1_samples": images,
                    "class_labels": image_labels,
                }

        def __len__(self) -> int:
            return len(torch_loader)

    return MNISTDataLoader()


