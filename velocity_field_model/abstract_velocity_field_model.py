from abc import abstractmethod
import torch

class AbstractVelocityFieldModel(torch.nn.Module):

    @abstractmethod
    def __init__(self, dimension: int, torch_device: str):
        """
        :param dimension: An integer describing the dimensionality of the distribution space.
            In the above, $d$.
        :param torch_device:
            The torch device to operate on
        """

        super().__init__()

        self.dimension = dimension
        self.torch_device = torch_device

        pass

    @abstractmethod
    def forward(self, t: torch.Tensor, feature_params: tuple[torch.Tensor, ...], x: torch.Tensor) -> torch.Tensor:
        """
        Map the given sample $x(t)$ (with associated time $t$) to a the velocity field at that position and time.

        :param t: The time associated with the given sample. Should be in $[0, 1]$.
            Shape (batch_size,)
        :param feature_params: The feature variable associated with the given sample. Should be in $[0, 1]$.
            Note this is NOT time dependent. We are merely referring to a linearized 
            version of the feature space indexed by feature_params.
            Tuple of tensors, each of shape (batch_size,)
        :param x: The tensor of a sample from a distribution at time $t$.
            Shape: (batch_size, dimension)
        """

        pass
