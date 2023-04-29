from abc import ABC, abstractmethod
from enum import Enum
from typing import List

class Space(str, Enum):
    l2 = "l2" # L2/Euclidean
    ip = "ip" # inner/dot product
    # The embedding model usually generates the normalized vectors. The cosine
    # similarity similarity is a dot product on normalized vectors. Usually
    # would not need to use cosine.
    cosine = "cosine"


class VectorStore(ABC):
    @abstractmethod
    def save(self, index_path: str):
        """
        Save the vectors to the file specified by index_path.
        """
        raise NotImplementedError

    @abstractmethod
    def load(self, index_path: str):
        """
        Load the vectors from the file specified by index_path.
        """
        raise NotImplementedError

    @abstractmethod
    def add(self, embeddings: List[List[float]], labels: List[int]):
        """
        Add the embeddings and the corresponding labels.
        """
        raise NotImplementedError

    @abstractmethod
    def query(
        self,
        embeddings: List[List[float]],
        top_k: int = 1,
    ) -> (List[List[int]], List[List[float]]):
        """
        Take one or more embeddings and return the top_k embedding ids and
        distances for each embedding.
        The distances are the original distances defined by the space, such as
        L2, inner/dot product, etc. The vector store provider should return the
        original distances.
        """
        raise NotImplementedError
