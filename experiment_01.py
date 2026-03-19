import os
from collections import defaultdict
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm import tqdm

from datasets import JointDistributionLoader, OccludedImageLoader, load_MNIST
from velocity_field_model import MetaUNetModel

if not os.path.exists("models/experiment_01"):
    os.makedirs("models/experiment_01")
if not os.path.exists("figures/experiment_01"):
    os.makedirs("figures/experiment_01")

TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ENABLE_PROGRESS_BAR = True
IMAGE_SHAPE: tuple[int, int, int] = (1, 28, 28)
IMAGE_DIMENSION = int(torch.prod(torch.tensor(IMAGE_SHAPE)).item())
NUM_CLASSES = 10

BATCH_SIZE = 256
FLOW_MATCHING_INITIAL_LEARNING_RATE = 5e-4
FLOW_MATCHING_NUM_EPOCHS = 128

ODE_NUM_TIME_STEPS = 16
VELOCITY_MASK = True
EPOCH_SAVE_PERIOD = 4
EPOCH_GENERATE_PERIOD = 1

occluded_image_loader = OccludedImageLoader(
    JointDistributionLoader(
        load_MNIST(batch_size=BATCH_SIZE, num_samples=60_000, preload=True, train=True)
    ),
    image_shape=IMAGE_SHAPE,
)

occluded_image_validation_loader = OccludedImageLoader(
    JointDistributionLoader(
        load_MNIST(batch_size=BATCH_SIZE, num_samples=5_000, preload=True, train=False)
    ),
    image_shape=IMAGE_SHAPE,
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
        x_pred[:, IMAGE_DIMENSION : IMAGE_DIMENSION + NUM_CLASSES],
        x_true[:, IMAGE_DIMENSION : IMAGE_DIMENSION + NUM_CLASSES],
    )
    loss = image_component + 0.01 * IMAGE_DIMENSION / NUM_CLASSES * label_component
    return loss

    # image_component = mean_square_error_loss(
    #     x_pred[:, :IMAGE_DIMENSION], x_true[:, :IMAGE_DIMENSION]
    # )
    # label_component = cross_entropy_loss(
    #     x_pred[:, IMAGE_DIMENSION : IMAGE_DIMENSION + NUM_CLASSES].softmax(dim=1),
    #     x_true[:, IMAGE_DIMENSION : IMAGE_DIMENSION + NUM_CLASSES],
    # )
    # loss = image_component + 0.1 * label_component
    # return loss


# --------------------------------------------------------------------------------


class VelocityFieldModel(torch.nn.Module):
    NUM_CHANNELS = IMAGE_SHAPE[0] + NUM_CLASSES
    EMBEDDING_DIMENSIONS = (256,)
    MODEL_BASE_CHANNELS = 64
    NUM_RES_BLOCKS = 2
    ATTENTION_RESOLUTIONS: tuple[int, ...] = (1,)
    CHANNEL_MULT: tuple[int, ...] = (1, 2, 4)

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
def create_and_save_images(filepath: str):
    # Images per row, i.e. num to generate and num to classify
    NUM_IMAGES = 10
    AX_SIZE = 3

    _mnist_loader = load_MNIST(
        batch_size=2 * NUM_IMAGES,
        num_samples=20 * NUM_IMAGES,
        train=False,
        shuffle=True,
    )
    _joint_distribution_loader = OccludedImageLoader(
        JointDistributionLoader(_mnist_loader), IMAGE_SHAPE
    )
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
    pushforward_class_confidence = X[:, IMAGE_DIMENSION:]
    pushforward_true_classes = y1_samples.argmax(dim=1)

    fig, axes = plt.subplots(
        nrows=4,
        ncols=NUM_IMAGES,
        figsize=(AX_SIZE * NUM_IMAGES, AX_SIZE * 4),
    )

    C, H, W = IMAGE_SHAPE
    ax_row_indices = [
        *([0] * NUM_IMAGES),
        *([2] * NUM_IMAGES),
    ]
    ax_col_indices = [*range(NUM_IMAGES), *range(NUM_IMAGES)]
    for image_index, (ax_row_index, ax_col_indices) in enumerate(
        zip(ax_row_indices, ax_col_indices)
    ):
        # print(image_index, (ax_row_index, ax_col_indices))
        # continue
        top_ax = axes[ax_row_index, ax_col_indices]
        bottom_ax = axes[ax_row_index + 1, ax_col_indices]
        top_ax.set_axis_off()
        bottom_ax.set_axis_off()

        x0_image = x0_samples[image_index].cpu().clamp(0, 1).view((H, W, C))
        generated_image = (
            pushforward_images[image_index].cpu().clamp(0, 1).view((H, W, C))
        )
        predicted_label = pushforward_classes[image_index]
        predicted_label_confidence = pushforward_class_confidence[
            image_index, predicted_label
        ].item()
        true_label = pushforward_true_classes[image_index]
        top_ax.set_title(
            f"Predicted Label: {predicted_label} ({predicted_label_confidence:.4f})\nTrue Label: {true_label}"
        )
        top_ax.imshow(
            x0_image,
            cmap="gray",
        )
        bottom_ax.imshow(
            generated_image,
            cmap="gray",
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
        occluded_image_loader,
        total=len(occluded_image_loader),
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
    epoch_loss = (epoch_loss / len(occluded_image_loader)).item()
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
            occluded_image_validation_loader,
            total=len(occluded_image_validation_loader),
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
            pushforward_class_confidence = X[:, IMAGE_DIMENSION:]
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

        num_items = len(occluded_image_validation_loader)
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
            f"models/experiment_01/augmentation_model.chk_{epoch_index:05d}", "wb"
        ) as f:
            torch.save(velocity_field_model.state_dict(), f)
        with open("models/experiment_01/history.csv", "w") as f:
            df = pd.DataFrame(history)
            df.to_csv(f, index=False)
    if epoch_index % EPOCH_GENERATE_PERIOD == 0:
        create_and_save_images(
            f"figures/experiment_01/augmentation_{epoch_index:05d}.png"
        )


with open("models/experiment_01/augmentation_model.model", "wb") as f:
    torch.save(velocity_field_model.state_dict(), f)
with open("models/experiment_01/history.csv", "w") as f:
    df = pd.DataFrame(history)
    df.to_csv(f, index=False)
