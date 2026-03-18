from abc import  abstractmethod
import torch

class AbstractEmbedding(torch.nn.Module):

    def __init__(
            self,
            data_dimension: int,
            latent_dimension: int,
            torch_device: str,
        ):
        """
        :param data_dimension:
            The dimension of the original data.
        :param latent_dimension:
            The dimension of the embedded data.
        :param torch_device:
            The torch device to operate on. It is assumed all incoming data will be on this device.
        """

        super().__init__()

        self.data_dimension = data_dimension
        self.latent_dimension = latent_dimension
        self.torch_device = torch_device
    
    @abstractmethod
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        :param t:
            Data to be embedded.
        :returns:
            Embedded data.
            Shape (|t|, latent_dimension).
        """

        pass