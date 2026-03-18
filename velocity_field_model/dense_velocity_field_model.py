import torch

from .abstract_velocity_field_model import AbstractVelocityFieldModel
from embedding import SinusoidalEmbedding

class DenseVelocityFieldModel(AbstractVelocityFieldModel):

    def __init__(self, data_dimension: int, time_embedding_dimension: int, feature_embedding_dimensions: tuple[int, ...], dense_units: list[int], torch_device: str):
        """
        :param dimension:
            An integer describing the dimensionality of the distribution space.
        :param time_embedding_dimension:
            An integer describing the dimensionality of the time embedding.
        :param time_embedding_dimension:
            An integer describing the dimensionality of each feature embedding.
            If empty, no features are used.
            Be wary to pass the features in the same order each time!
        :param dense_units:
            A list of integers giving the hidden units for each dense layer
            after the flatten. E.g. [256, 128].
        :param torch_device:
            The torch device to operate on.
        """

        super().__init__(data_dimension, torch_device)

        self.time_embedding_module = SinusoidalEmbedding(time_embedding_dimension, torch_device=torch_device)
        self.feature_embedding_dimensions = feature_embedding_dimensions
        self.total_feature_embedding_dimension = sum(feature_embedding_dimensions)
        self.feature_embedding_modules = [SinusoidalEmbedding(d, torch_device=torch_device) for d in feature_embedding_dimensions]
        dense_layers = []
        dense_input_size = time_embedding_dimension+self.total_feature_embedding_dimension+data_dimension
        for units in dense_units:
            dense_layers.append(torch.nn.Linear(dense_input_size, units))
            dense_layers.append(torch.nn.ELU())
            dense_input_size = units
        dense_layers.append(torch.nn.Linear(dense_input_size, data_dimension))

        # Network for modelling the actual flow.
        # Inputs:  $[0, 1] x [0, 1]^|features| x R^d$, in that order, e.g. `torch.cat([t, feature_params, x]. -1)` to create sample
        # Outputs  $R^d$
        self.network = torch.nn.Sequential(*dense_layers)
        self.to(torch_device)

    def forward(self, t: torch.Tensor, feature_params: tuple[torch.Tensor, ...], x: torch.Tensor) -> torch.Tensor:
        """
        Map the given sample $x(t)$ (with associated time $t$) to a the velocity field at that position and time.

        :param t: The time associated with the given sample. Should be in $[0, 1]$.
            Shape (batch_size,)
        :param feature_params: The feature variable associated with the given sample. Should be in $[0, 1]$.
            Note this is NOT time dependent. We are merely referring to a linearized 
            version of the feature space indexed by feature_params.
            Ensure features are passed in the same order with each call.
            Tuple of tensors, each of shape (batch_size,)
        :param x_t: The tensor of a sample from a distribution at time $t$.
            Shape: (batch_size, dimension)
        """

        # Must have the same number of features as expected features
        assert len(feature_params) == len(self.feature_embedding_dimensions)

        t = self.time_embedding_module(t)
        embedded_features: tuple[torch.Tensor, ...] = tuple(embedding(feature) for embedding, feature in zip(self.feature_embedding_modules, feature_params))
        return self.network(torch.cat((t, *embedded_features, x), dim = -1))

if __name__ == "__main__":
    DIMENSION = 32
    TIME_EMBEDDING_DIMENSION = 8
    NUM_SAMPLES = 16

    FEATURE_EMBEDDING_DIMENSIONS = ()
    feature_params = ()
    # FEATURE_EMBEDDING_DIMENSIONS = (12, 16,)
    # feature_params = (torch.linspace(0, 1, NUM_SAMPLES), torch.linspace(0, 1, NUM_SAMPLES))

    velocity_field_model = DenseVelocityFieldModel(
        data_dimension = DIMENSION,
        time_embedding_dimension = TIME_EMBEDDING_DIMENSION,
        feature_embedding_dimensions = FEATURE_EMBEDDING_DIMENSIONS,
        dense_units = [256, 128, 64],
        torch_device = "cpu",
    )

    t = torch.linspace(0, 1, NUM_SAMPLES)
    x = torch.normal(0, 1, size=(NUM_SAMPLES, DIMENSION))
    print(x.shape, t.shape)
    v = velocity_field_model(t, feature_params, x)
    print(v.shape)
