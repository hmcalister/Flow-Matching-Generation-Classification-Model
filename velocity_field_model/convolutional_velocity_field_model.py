import math
import torch

from .abstract_velocity_field_model import AbstractVelocityFieldModel
from embedding import SinusoidalEmbedding

class ConvolutionalVelocityFieldModel(AbstractVelocityFieldModel):
    """
    Convolutional velocity field model.

    The flat input vector is reshaped into (C, H, W). Time t and feature
    embeddings are prepended as extra channels broadcast over the spatial
    dimensions. The result passes through 2D conv blocks, is flattened, then
    processed by dense layers to produce the output velocity in R^d.
    """

    def __init__(
        self,
        data_dimension: int,
        input_shape: tuple[int, int, int],
        time_embedding_dimension: int,
        feature_embedding_dimensions: tuple[int, ...],
        conv_channels: list[int],
        dense_units: list[int],
        torch_device: str,
        kernel_size: int = 3,
    ):
        """
        :param dimension:
            An integer describing the dimensionality of the distribution space.
            Must equal math.prod(input_shape).
        :param input_shape:
            A (channels, height, width) tuple defining how the flat dimension
            vector is reshaped before entering the conv blocks.
            math.prod(input_shape) must equal dimension.
        :param time_embedding_dimension:
            Dimensionality of the sinusoidal time embedding. The embedding is
            broadcast as extra channels over the spatial dimensions.
        :param feature_embedding_dimensions:
            Dimensionality of the sinusoidal embedding for each feature parameter.
            Each embedding is broadcast as extra channels over the spatial dimensions.
            Pass an empty tuple if no features are used.
        :param conv_channels:
            A list of integers giving the number of output channels for each
            successive Conv2d block. E.g. [32, 64, 128].
        :param dense_units:
            A list of integers giving the hidden units for each dense layer
            after the flatten. E.g. [256, 128].
        :param torch_device:
            The torch device to operate on.
        :param kernel_size:
            Kernel size used in every Conv2d layer (default 3). Padding is set
            to kernel_size // 2 so spatial dimensions are preserved.
        """

        super().__init__(data_dimension, torch_device)

        in_channels, height, width = input_shape
        assert math.prod(input_shape) == data_dimension, (
            f"math.prod(input_shape) = {math.prod(input_shape)} must equal dimension = {data_dimension}"
        )

        self.input_shape = input_shape
        self.feature_embedding_dimensions = feature_embedding_dimensions

        self.time_embedding_module = SinusoidalEmbedding(time_embedding_dimension, torch_device=torch_device)
        self.feature_embedding_modules = torch.nn.ModuleList(
            [SinusoidalEmbedding(d, torch_device=torch_device) for d in feature_embedding_dimensions]
        )

        padding = kernel_size // 2

        # Time embedding and each feature embedding are each broadcast as one
        # channel per embedding dimension over the spatial dimensions, then
        # concatenated with the data channels before the conv blocks.
        # We map each embedding to a single channel via a 1x1 conv projection.
        self.time_proj = torch.nn.Conv2d(time_embedding_dimension, 1, kernel_size=1)
        self.feature_projs = torch.nn.ModuleList(
            [torch.nn.Conv2d(d, 1, kernel_size=1) for d in feature_embedding_dimensions]
        )

        # in_channels + 1 (time) + len(feature_embedding_dimensions) (features)
        conv_in_channels = in_channels + 1 + len(feature_embedding_dimensions)
        self.conv_blocks = torch.nn.ModuleList()
        for out_channels in conv_channels:
            self.conv_blocks.append(torch.nn.Sequential(
                torch.nn.Conv2d(conv_in_channels, out_channels, kernel_size, padding=padding),
                torch.nn.ELU(),
                torch.nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
                torch.nn.ELU(),
            ))
            conv_in_channels = out_channels

        # Size of flattened feature map after all conv blocks (spatial dims unchanged)
        flat_size = conv_in_channels * height * width

        # Dense layers after flatten
        self.dense_layers = torch.nn.ModuleList()
        dense_in = flat_size
        for units in dense_units:
            self.dense_layers.append(torch.nn.Sequential(
                torch.nn.Linear(dense_in, units),
                torch.nn.ELU(),
            ))
            dense_in = units

        self.output_layer = torch.nn.Linear(dense_in, data_dimension)

        self.to(torch_device)

    def forward(self, t: torch.Tensor, feature_params: tuple[torch.Tensor, ...], x: torch.Tensor) -> torch.Tensor:
        """
        Map the given sample $x(t)$ (with associated time $t$) to the velocity field.

        :param t: Time associated with each sample. Should be in $[0, 1]$.
            Shape: (batch_size,)
        :param feature_params: Feature variables for each sample. Should be in $[0, 1]$.
            Tuple of tensors, each of shape (batch_size,).
        :param x: Tensor of samples from a distribution at time $t$.
            Shape: (batch_size, dimension)
        :returns: Velocity field at each sample.
            Shape: (batch_size, dimension)
        """

        assert len(feature_params) == len(self.feature_embedding_dimensions)

        batch_size = x.shape[0]
        in_channels, height, width = self.input_shape

        # Reshape flat vector to (batch, channels, height, width)
        x_img = x.reshape(batch_size, in_channels, height, width)

        # Embed time and broadcast over spatial dims via 1x1 projection
        t_emb:torch.Tensor = self.time_embedding_module(t)  # (batch, time_embedding_dimension)
        t_emb = t_emb.view(batch_size, self.feature_embedding_dimensions[0], 1, 1).expand(-1, -1, height, width)
        t_channel = self.time_proj(t_emb)  # (batch, 1, H, W)

        # Embed each feature and broadcast over spatial dims via 1x1 projection
        feature_channels = []
        for proj, embedding_module, fp in zip(self.feature_projs, self.feature_embedding_modules, feature_params):
            f_emb = embedding_module(fp)  # (batch, feature_embedding_dimension)
            f_emb = f_emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, height, width)
            feature_channels.append(proj(f_emb))  # (batch, 1, H, W)

        x_cat = torch.cat([x_img, t_channel, *feature_channels], dim=1)

        for conv_block in self.conv_blocks:
            x_cat = conv_block(x_cat)

        x_flat = x_cat.flatten(start_dim=1)

        for dense_layer in self.dense_layers:
            x_flat = dense_layer(x_flat)

        return self.output_layer(x_flat)


if __name__ == "__main__":
    import math

    DIMENSION = 32           # must equal math.prod(INPUT_SHAPE)
    INPUT_SHAPE = (4, 4, 2)  # 4 channels x 4 x 2 = 32
    TIME_EMBEDDING_DIMENSION = 8
    CONV_CHANNELS = [16, 32]
    DENSE_UNITS = [128, 64]
    NUM_SAMPLES = 16

    assert math.prod(INPUT_SHAPE) == DIMENSION

    FEATURE_EMBEDDING_DIMENSIONS = (12, 16)
    feature_params = (
        torch.linspace(0, 1, NUM_SAMPLES).unsqueeze(1),
        torch.linspace(0, 1, NUM_SAMPLES).unsqueeze(1),
    )

    velocity_field_model = ConvolutionalVelocityFieldModel(
        data_dimension=DIMENSION,
        input_shape=INPUT_SHAPE,
        time_embedding_dimension=TIME_EMBEDDING_DIMENSION,
        feature_embedding_dimensions=FEATURE_EMBEDDING_DIMENSIONS,
        conv_channels=CONV_CHANNELS,
        dense_units=DENSE_UNITS,
        torch_device="cpu",
    )

    t = torch.linspace(0, 1, NUM_SAMPLES).reshape(-1, 1)
    x = torch.normal(0, 1, size=(NUM_SAMPLES, DIMENSION))
    print(x.shape, t.shape)
    v = velocity_field_model(t, feature_params, x)
    print(v.shape)