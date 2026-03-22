import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm import tqdm

from datasets import JointDistributionLoader, load_CIFAR10, load_MNIST
from velocity_field_model import MetaUNetModel

TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET = "CIFAR10"
VELOCITY_MASK = False

BATCH_SIZE = 128
FLOW_MATCHING_INITIAL_LEARNING_RATE = 5e-4
FLOW_MATCHING_NUM_EPOCHS = 128

ODE_NUM_TIME_STEPS = 16
EPOCH_SAVE_PERIOD = 4
EPOCH_GENERATE_PERIOD = 1

ENABLE_PROGRESS_BAR = True

# --------------------------------------------------------------------------------

DATASET = DATASET.upper()
EXPERIMENT_DIR = "experiment_00_mask" if VELOCITY_MASK else "experiment_00_nomask"
MODELS_DIR = Path("models").joinpath(DATASET).joinpath(EXPERIMENT_DIR)
FIGURES_DIR = Path("figures").joinpath(DATASET).joinpath(EXPERIMENT_DIR)

if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)
if not os.path.exists(FIGURES_DIR):
    os.makedirs(FIGURES_DIR)

if DATASET == "MNIST":
    base_dataset_loader_method = load_MNIST
elif DATASET == "CIFAR10":
    base_dataset_loader_method = load_CIFAR10
else:
    print("DATASET NOT IDENTIFIED")
    exit()

_base_dataset_loader = base_dataset_loader_method(
    batch_size=BATCH_SIZE, num_samples=BATCH_SIZE, preload=True, train=True
)
IMAGE_SHAPE: tuple[int, int, int] = _base_dataset_loader.IMAGE_SHAPE
IMAGE_DIMENSION = _base_dataset_loader.DATA_DIMENSION
CLASS_LABELS = _base_dataset_loader.CLASS_LABELS
NUM_CLASSES = len(_base_dataset_loader.CLASS_LABELS)
del _base_dataset_loader

joint_distribution_loader = JointDistributionLoader(
    base_dataset_loader_method(
        batch_size=BATCH_SIZE, num_samples=50_000, preload=True, train=True
    )
)

joint_distribution_validation_loader = JointDistributionLoader(
    base_dataset_loader_method(
        batch_size=BATCH_SIZE, num_samples=5_000, preload=True, train=False
    )
)


# --------------------------------------------------------------------------------

mean_square_error_loss = torch.nn.MSELoss()
cross_entropy_loss = torch.nn.CrossEntropyLoss()


def loss_fn(x_pred: torch.Tensor, x_true: torch.Tensor) -> torch.Tensor:
    # Shape (batch_size, IMAGE_DIMENSION + NUM_CLASSES)

    image_component = mean_square_error_loss(
        x_pred[:, :IMAGE_DIMENSION], x_true[:, :IMAGE_DIMENSION]
    )
    label_component = mean_square_error_loss(
        x_pred[:, IMAGE_DIMENSION : IMAGE_DIMENSION + NUM_CLASSES] / x_true.amax(),
        x_true[:, IMAGE_DIMENSION : IMAGE_DIMENSION + NUM_CLASSES] / x_true.amax(),
    )
    loss = image_component + 0.01 * IMAGE_DIMENSION / NUM_CLASSES * label_component
    return loss


# --------------------------------------------------------------------------------


class VelocityFieldModel(torch.nn.Module):
    NUM_CHANNELS = IMAGE_SHAPE[0] + NUM_CLASSES
    EMBEDDING_DIMENSIONS = (256,)
    MODEL_BASE_CHANNELS = 64
    NUM_RES_BLOCKS = 2
    ATTENTION_RESOLUTIONS: tuple[int, ...] = (2, 4)
    CHANNEL_MULT: tuple[int, ...] = (1, 2, 2, 4)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.network = MetaUNetModel(
            num_channels=self.NUM_CHANNELS,
            out_channels=self.NUM_CHANNELS,
            embedding_dimensions=self.EMBEDDING_DIMENSIONS,
            model_channels=self.MODEL_BASE_CHANNELS,
            num_res_blocks=self.NUM_RES_BLOCKS,
            attention_resolutions=self.ATTENTION_RESOLUTIONS,
            dropout=0.1,
            channel_mult=self.CHANNEL_MULT,
        )
        self.network.to(TORCH_DEVICE)

    def forward(self, t: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        # Accepts and returns a tensor of shape (batch_size, IMAGE_DIMENSION + NUM_CLASSES)
        # First part is image data, second part is distribution over class labels

        image_data = X[:, :IMAGE_DIMENSION]
        class_data = X[:, IMAGE_DIMENSION : IMAGE_DIMENSION + NUM_CLASSES]

        image_channel = image_data.view(-1, *IMAGE_SHAPE)
        class_channels = class_data.view(-1, NUM_CLASSES, 1, 1).repeat(
            1, 1, *IMAGE_SHAPE[1:]
        )

        # Shape (batch_size, 1+NUM_CLASSES, IMAGE_WIDTH, IMAGE_HEIGHT)
        X = torch.cat([image_channel, class_channels], dim=1)
        v = self.network((t,), X)

        image_v = v[:, : IMAGE_SHAPE[0], :, :].view(-1, IMAGE_DIMENSION)
        class_v = v[:, IMAGE_SHAPE[0] :, :, :].mean(dim=(2, 3))
        v = torch.cat([image_v, class_v], dim=1)

        return v


velocity_field_model = VelocityFieldModel()
torch.compile(velocity_field_model)

# --------------------------------------------------------------------------------


@torch.no_grad()
def create_and_save_images(filepath: str | Path):
    # Images per row, i.e. num to generate and num to classify
    NUM_IMAGES = 10
    AX_SIZE = 3

    _dataset_loader = base_dataset_loader_method(
        batch_size=2 * NUM_IMAGES,
        num_samples=20 * NUM_IMAGES,
        train=False,
        shuffle=True,
    )
    _joint_distribution_loader = JointDistributionLoader(_dataset_loader)
    _data = next(iter(_joint_distribution_loader))
    x0_samples = _data["x0_samples"].to(TORCH_DEVICE)
    x1_samples = _data["x1_samples"].to(TORCH_DEVICE)
    y0_samples = _data["y0_samples"].to(TORCH_DEVICE)
    y1_samples = _data["y1_samples"].to(TORCH_DEVICE)

    X = torch.cat([x0_samples, y0_samples], dim=1)
    X1 = torch.cat([x1_samples, y1_samples], dim=1)
    velocity_mask = (X1 != X).float()
    if not VELOCITY_MASK:
        velocity_mask = torch.ones_like(velocity_mask)
    t_steps = torch.linspace(0, 1, ODE_NUM_TIME_STEPS).to(TORCH_DEVICE)

    for index in range(ODE_NUM_TIME_STEPS):
        t = t_steps[index] * torch.ones((2 * NUM_IMAGES,), device=TORCH_DEVICE)
        v = velocity_field_model.forward(t, X) * velocity_mask
        X = X + (1 / ODE_NUM_TIME_STEPS) * v

    pushforward_images = X[:, :IMAGE_DIMENSION].view((-1, *IMAGE_SHAPE))
    pushforward_classes = X[:, IMAGE_DIMENSION:].argmax(dim=1)
    pushforward_class_confidence = torch.nn.functional.softmax(
        5 * X[:, IMAGE_DIMENSION:], dim=1
    )
    pushforward_true_classes = y1_samples.argmax(dim=1)

    fig, axes = plt.subplots(
        nrows=2,
        ncols=NUM_IMAGES,
        figsize=(AX_SIZE * NUM_IMAGES, AX_SIZE * 2),
    )

    for image_index, ax in enumerate(axes.ravel()):
        ax.set_axis_off()

        generated_image = (
            pushforward_images[image_index]
            .cpu()
            .clamp(0, 1)
            .view(IMAGE_SHAPE)
            .permute(1, 2, 0)
        )
        predicted_label = pushforward_classes[image_index]
        predicted_label_confidence = pushforward_class_confidence[
            image_index, predicted_label
        ].item()
        true_label = pushforward_true_classes[image_index]
        ax.set_title(
            f"Predicted Label: {CLASS_LABELS[predicted_label]} ({predicted_label_confidence:.4f})\nTrue Label: {CLASS_LABELS[true_label]}"
        )
        ax.imshow(
            generated_image,
            cmap="gray",  # Ignored for color images
        )

    fig.tight_layout()
    fig.savefig(filepath)
    plt.close()


# --------------------------------------------------------------------------------

optimizer = torch.optim.Adam(
    velocity_field_model.parameters(),
    FLOW_MATCHING_INITIAL_LEARNING_RATE,
)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    FLOW_MATCHING_NUM_EPOCHS,
    eta_min=FLOW_MATCHING_INITIAL_LEARNING_RATE / 10,
)

epoch_progress_bar = tqdm(
    range(FLOW_MATCHING_NUM_EPOCHS),
    desc="Flow Matching Training Epochs",
    disable=not ENABLE_PROGRESS_BAR,
)

history = defaultdict(list)

for epoch_index in epoch_progress_bar:
    velocity_field_model.train()
    epoch_loss = torch.zeros(1, device=TORCH_DEVICE)
    batch_progress_bar = tqdm(
        joint_distribution_loader,
        total=len(joint_distribution_loader),
        desc="Batch",
        leave=False,
        disable=not ENABLE_PROGRESS_BAR,
    )
    for batch_index, data in enumerate(batch_progress_bar):
        x0_samples = data["x0_samples"].to(TORCH_DEVICE)
        x1_samples = data["x1_samples"].to(TORCH_DEVICE)
        y0_samples = data["y0_samples"].to(TORCH_DEVICE)
        y1_samples = data["y1_samples"].to(TORCH_DEVICE)

        X0 = torch.cat([x0_samples, y0_samples], dim=1)
        X1 = torch.cat([x1_samples, y1_samples], dim=1)
        velocity_mask = (X1 != X0).float()
        if not VELOCITY_MASK:
            velocity_mask = torch.ones_like(velocity_mask)
        effective_batch_size = X0.shape[0]
        t = torch.rand(effective_batch_size, device=TORCH_DEVICE)

        X_t = (1 - t[:, None]) * X0 + t[:, None] * X1
        expected_velocity = X1 - X0

        predicted_velocity = velocity_field_model(t, X_t)
        batch_loss = loss_fn(
            predicted_velocity * velocity_mask, expected_velocity * velocity_mask
        )
        batch_loss.backward()

        torch.nn.utils.clip_grad_norm_(velocity_field_model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

        with torch.no_grad():
            epoch_loss += batch_loss

        batch_progress_bar.set_postfix(
            {
                "Loss": f"{batch_loss.item():4E}",
            }
        )
    epoch_loss = (epoch_loss / len(joint_distribution_loader)).item()
    epoch_progress_bar.set_postfix(
        {
            "Epoch Loss": f"{epoch_loss:.4E}",
        }
    )
    history["train_loss"].append(epoch_loss)

    velocity_field_model.eval()
    with torch.no_grad():
        epoch_validation_loss = torch.zeros(1, device=TORCH_DEVICE)
        epoch_validation_accuracy = torch.zeros(1, device=TORCH_DEVICE)
        epoch_validation_cross_entropy = torch.zeros(1, device=TORCH_DEVICE)
        batch_progress_bar = tqdm(
            joint_distribution_validation_loader,
            total=len(joint_distribution_validation_loader),
            desc="Batch",
            leave=False,
            disable=not ENABLE_PROGRESS_BAR,
        )
        for batch_index, data in enumerate(batch_progress_bar):
            x0_samples = data["x0_samples"].to(TORCH_DEVICE)
            x1_samples = data["x1_samples"].to(TORCH_DEVICE)
            y0_samples = data["y0_samples"].to(TORCH_DEVICE)
            y1_samples = data["y1_samples"].to(TORCH_DEVICE)

            X0 = torch.cat([x0_samples, y0_samples], dim=1)
            X1 = torch.cat([x1_samples, y1_samples], dim=1)
            velocity_mask = (X1 != X0).float()
            if not VELOCITY_MASK:
                velocity_mask = torch.ones_like(velocity_mask)

            expected_velocity = X1 - X0

            X = X0
            effective_batch_size = X.shape[0]
            t_steps = torch.linspace(0, 1, ODE_NUM_TIME_STEPS).to(TORCH_DEVICE)
            batch_loss = torch.zeros(1, device=TORCH_DEVICE)
            for index in range(ODE_NUM_TIME_STEPS):
                t = t_steps[index] * torch.ones(
                    (effective_batch_size,), device=TORCH_DEVICE
                )
                v = velocity_field_model(t, X) * velocity_mask
                X = X + (1 / ODE_NUM_TIME_STEPS) * v

                batch_loss += loss_fn(v, expected_velocity)
            batch_loss /= ODE_NUM_TIME_STEPS

            pushforward_images = X[:, :IMAGE_DIMENSION].view((-1, *IMAGE_SHAPE))
            pushforward_classes = X[:, IMAGE_DIMENSION:].argmax(dim=1).float()
            pushforward_class_confidence = torch.nn.functional.softmax(
                5 * X[:, IMAGE_DIMENSION:], dim=1
            )
            pushforward_true_classes = y1_samples.argmax(dim=1).float()

            epoch_validation_loss += batch_loss
            epoch_validation_accuracy += (
                (pushforward_classes == pushforward_true_classes)
                .type(torch.float)
                .mean()
            )
            epoch_validation_cross_entropy += cross_entropy_loss(
                pushforward_class_confidence, y1_samples
            )

        num_items = len(joint_distribution_validation_loader)
        history["validation_loss"].append(epoch_validation_loss.item() / num_items)
        history["validation_classification_accuracy"].append(
            epoch_validation_accuracy.item() / num_items
        )
        history["validation_cross_entropy"].append(
            epoch_validation_cross_entropy.item() / num_items
        )

    lr_scheduler.step()
    if epoch_index % EPOCH_SAVE_PERIOD == 0:
        with open(
            MODELS_DIR.joinpath(f"augmentation_model.chk_{epoch_index:05d}"), "wb"
        ) as f:
            torch.save(velocity_field_model.state_dict(), f)
        with open(MODELS_DIR.joinpath("history.csv"), "w") as f:
            df = pd.DataFrame(history)
            df.to_csv(f, index=False)
    if epoch_index % EPOCH_GENERATE_PERIOD == 0:
        create_and_save_images(
            FIGURES_DIR.joinpath(f"augmentation_{epoch_index:05d}.png")
        )


with open(MODELS_DIR.joinpath("augmentation_model.model"), "wb") as f:
    torch.save(velocity_field_model.state_dict(), f)
with open(MODELS_DIR.joinpath("history.csv"), "w") as f:
    df = pd.DataFrame(history)
    df.to_csv(f, index=False)
