import os
from typing import List
import hnswlib

from vectorstore.vectorstore import VectorStore

class HnswlibStore(VectorStore):
    index: hnswlib.Index

    # hnswlib params
    dim: int
    space: str # l2, consine, or ip
    max_elements: int
    # M: int # max number of connections on upper layers
    # ef_construction: int # number of the nearest neighbors at index time
    # ef_search: int # number of the nearest neighbors to search

    def __init__(self, dim: int, space: str, max_elements: int):
        self.index = hnswlib.Index(space, dim)
        self.index.init_index(max_elements)
        self.dim = dim
        self.max_elements = max_elements
        self.space = space

    def save(self, index_path: str):
        self.index.save_index(index_path)

    def load(self, index_path: str):
        self.index.load_index(index_path, self.max_elements)

    def add(self, embeddings: List[List[float]], labels: List[int]):
        self.index.add_items(embeddings, labels)

    def query(
        self,
        embeddings: List[List[float]],
        top_k: int = 1,
    ) -> (List[List[int]], List[List[float]]):
        """
        Take one or more embeddings and return the top_k embedding labels and
        distances for each embedding.
        """
        return self.index.knn_query(embeddings, top_k)
