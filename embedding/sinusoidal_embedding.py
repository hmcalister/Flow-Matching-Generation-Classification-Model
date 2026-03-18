import torch

from .abstract_embedding import AbstractEmbedding

class SinusoidalEmbedding(AbstractEmbedding):
    def __init__(self,
                 latent_dimension: int,
                 torch_device: str,
                 embedding_scale_base: float = 128,
                ):
        """
        Data dimension must always be one.

        :param latent_dimension:
            The dimension of the embedded data.
        :param torch_device:
            The torch device to operate on. It is assumed all incoming data will be on this device.
        :param embedding_scale_base:
            The dimension of the embedded data.
        """
        
        super().__init__(
            data_dimension=1,
            latent_dimension=latent_dimension,
            torch_device=torch_device
        )

        self.half_latent_dimension = latent_dimension // 2
        self.embedding_scale = torch.log(torch.tensor(embedding_scale_base)) / (self.half_latent_dimension - 1)
        self.time_embedding_frequencies = torch.exp(torch.arange(self.half_latent_dimension, device=torch_device) * - self.embedding_scale)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Create and return a sinusoidal time embedding of shape (|t|, time_embedding_dimension)
        """

        # print(t[:, None].shape, self.time_embedding_frequencies[None, :].shape)
        embedding = t[:, None] * self.time_embedding_frequencies[None, :]
        embedding = torch.cat([torch.sin(embedding), torch.cos(embedding)], dim=1)
        return embedding
