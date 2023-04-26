from abc import ABC, abstractmethod
from typing import List

class Model(ABC):
    @abstractmethod
    def get_max_token_size(self) -> int:
        """
        Get the max number of tokens for the text. The input text longer than
        the max token may be truncated.
        """
        raise NotImplementedError

    @abstractmethod
    def get_dim(self) -> int:
        """
        Return the embedding dimension
        """
        raise NotImplementedError

    @abstractmethod
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Takes in a list of texts and returns a list of embeddings for each text.
        """
        raise NotImplementedError

    @abstractmethod
    def set_model(self, model_name: str, max_token_size: int, dim: int):
        """
        Set to use the specified model.
        """
        raise NotImplementedError
