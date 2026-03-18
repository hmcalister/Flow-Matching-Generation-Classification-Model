import torch

from .abstract_embedding import AbstractEmbedding

class NullEmbedding(AbstractEmbedding):
    def __init__(self,
                 torch_device: str,
                ):
        """
        Data dimension and latent dimension are both 1.
        Embedding is simply returning the original data.

        :param torch_device:
            The torch device to operate on. It is assumed all incoming data will be on this device.
        """
        
        super().__init__(
            data_dimension=1,
            latent_dimension=1,
            torch_device=torch_device
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:

        return t
