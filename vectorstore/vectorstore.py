from abc import ABC, abstractmethod
from typing import List

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
        This function follows the HNSWLIB query results that a larger distance
        value indicates lower similarity, while a smaller distance value
        indicates higher similarity. So the distances returned by this function
        are arranged in ascending order, with the nearest neighbors appearing
        first.
        """
        raise NotImplementedError
