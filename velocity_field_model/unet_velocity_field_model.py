import torch

from .abstract_velocity_field_model import AbstractVelocityFieldModel
from embedding import SinusoidalEmbedding


class UnetVelocityFieldModel(AbstractVelocityFieldModel):
    """
    UNet-style velocity field model with sinusoidal time and feature embeddings.
    Skip connections from encoder layers are concatenated into decoder layers.
    """

    def __init__(
        self,
        data_dimension: int,
        time_embedding_dimension: int,
        feature_embedding_dimensions: tuple[int, ...],
        encoder_layers: list[int],
        decoder_layers: list[int] | None,
        torch_device: str
    ):
        """
        :param data_dimension:
            An integer describing the dimensionality of the distribution space.
        :param time_embedding_dimension:
            An integer describing the dimensionality of the time embedding.
        :param feature_embedding_dimensions:
            A tuple of integers describing the dimensionality of each feature embedding.
            If empty, no features are used.
            Be wary to pass the features in the same order each time!
        :param encoder_layers:
            A list of integers describing the hidden dimensions for each encoder layer.
            Example: [64, 128, 256] creates 3 encoder blocks with those hidden dimensions.
        :param decoder_layers:
            A list of integers describing the hidden dimensions for each decoder layer.
            Example: [256, 128, 64] creates 3 decoder blocks with those hidden dimensions.
            If None, simply mirrors the encoder layers.
        :param torch_device:
            The torch device to operate on.
        """

        super().__init__(data_dimension, torch_device)

        self.time_embedding_module = SinusoidalEmbedding(time_embedding_dimension, torch_device=torch_device)
        self.feature_embedding_dimensions = feature_embedding_dimensions
        self.total_feature_embedding_dimension = sum(feature_embedding_dimensions)
        self.feature_embedding_modules = [SinusoidalEmbedding(d, torch_device=torch_device) for d in feature_embedding_dimensions]

        # Build encoder
        self.encoder_blocks = torch.nn.ModuleList()
        input_dim = data_dimension + time_embedding_dimension + self.total_feature_embedding_dimension
        for hidden_dim in encoder_layers:
            self.encoder_blocks.append(torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.ELU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ELU()
            ))
            input_dim = hidden_dim

        if decoder_layers is None:
            decoder_layers = encoder_layers.copy()
            decoder_layers.reverse()

        # Build decoder with skip connections
        self.decoder_blocks = torch.nn.ModuleList()
        for i, hidden_dim in enumerate(decoder_layers):
            skip_dim = encoder_layers[-(i + 1)] if i < len(encoder_layers) else 0
            self.decoder_blocks.append(torch.nn.Sequential(
                torch.nn.Linear(input_dim + skip_dim, hidden_dim),
                torch.nn.ELU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ELU()
            ))
            input_dim = hidden_dim

        # Final output layer
        self.output_layer = torch.nn.Linear(input_dim, data_dimension)

        self.to(torch_device)

    def forward(self, t: torch.Tensor, feature_params: tuple[torch.Tensor, ...], x: torch.Tensor) -> torch.Tensor:
        """
        Map the given sample $x(t)$ (with associated time $t$) to the velocity field at that position and time.

        :param t: The time associated with the given sample. Should be in $[0, 1]$.
            Shape (batch_size, 1)
        :param feature_params: The feature variable associated with the given sample.
            Note this is NOT time dependent.
            Tuple of tensors, each of shape (batch_size, 1)
        :param x: The tensor of a sample from a distribution at time $t$.
            Shape: (batch_size, dimension)
        """

        assert len(feature_params) == len(self.feature_embedding_dimensions)

        t_embedded = self.time_embedding_module(t)
        embedded_features: tuple[torch.Tensor, ...] = tuple(
            embedding(feature) for embedding, feature in zip(self.feature_embedding_modules, feature_params)
        )
        h = torch.cat((t_embedded, *embedded_features, x), dim=-1)

        # Encoder pass - store skip connections
        skip_connections = []
        for encoder_block in self.encoder_blocks:
            h = encoder_block(h)
            skip_connections.append(h)

        # Decoder pass - use skip connections
        for i, decoder_block in enumerate(self.decoder_blocks):
            skip_idx = -(i + 1)
            h = torch.cat((h, skip_connections[skip_idx]), dim=-1)
            h = decoder_block(h)

        return self.output_layer(h)


if __name__ == "__main__":
    DIMENSION = 32
    TIME_EMBEDDING_DIMENSION = 8
    NUM_SAMPLES = 16
    ENCODER_LAYERS = [64, 128, 256]
    DECODER_LAYERS = [256, 128, 64]

    FEATURE_EMBEDDING_DIMENSIONS = ()
    feature_params = ()
    # FEATURE_EMBEDDING_DIMENSIONS = (12, 16,)
    # feature_params = (torch.linspace(0, 1, NUM_SAMPLES).unsqueeze(1), torch.linspace(0, 1, NUM_SAMPLES).unsqueeze(1))

    velocity_field_model = UnetVelocityFieldModel(
        data_dimension=DIMENSION,
        time_embedding_dimension=TIME_EMBEDDING_DIMENSION,
        feature_embedding_dimensions=FEATURE_EMBEDDING_DIMENSIONS,
        encoder_layers=ENCODER_LAYERS,
        decoder_layers=DECODER_LAYERS,
        torch_device="cpu",
    )

    t = torch.linspace(0, 1, NUM_SAMPLES).reshape(-1, 1)
    x = torch.normal(0, 1, size=(NUM_SAMPLES, DIMENSION))
    print(x.shape, t.shape)
    v = velocity_field_model(t, feature_params, x)
    print(v.shape)