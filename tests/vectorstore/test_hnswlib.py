import os
import pytest
import random
from typing import List
import numpy as np

from vectorstore.factory import get_vector_store

class TestHnswlib():
    def test_save_empty_index(self):
        dim = 384
        max_elements = 1
        space = "cosine"
        store = get_vector_store("hnswlib", dim, space, max_elements)

        index_path = f"ut_empty_index.bin"
        store.save(index_path)

        assert os.path.exists(index_path)
        os.remove(index_path)

    def test_index(self):
        dim = 16
        max_elements = 5
        space = "l2"
        store = get_vector_store("hnswlib", dim, space, max_elements)

        embeddings = np.float32(np.random.random((max_elements, dim)))
        labels = np.arange(max_elements)

        store.add(embeddings, labels)

        qlabels, distances = store.query(embeddings=embeddings[0], top_k=1)
        assert 1 == len(qlabels)
        assert 1 == len(distances)
        assert 1 == len(qlabels[0])
        assert 1 == len(distances[0])
        assert labels[0] == qlabels[0][0]
        assert 0.0 == distances[0][0]

        qlabels, distances = store.query(
            embeddings=embeddings[0], top_k=max_elements)
        assert 1 == len(qlabels)
        assert 1 == len(distances)
        assert max_elements == len(qlabels[0])
        assert max_elements == len(distances[0])
        assert labels[0] == qlabels[0][0]
        assert 0.0 == distances[0][0]
        qlabels[0].sort()
        assert all([a == b for a, b in zip(qlabels[0], labels)])

        index_path = "ut_index.bin"
        store.save(index_path)

        store1 = get_vector_store("hnswlib", dim, space, max_elements)
        store1.load(index_path)

        qlabels, distances = store1.query(embeddings=embeddings[0], top_k=1)
        assert 1 == len(qlabels)
        assert 1 == len(distances)
        assert 1 == len(qlabels[0])
        assert 1 == len(distances[0])
        assert labels[0] == qlabels[0][0]
        assert 0.0 == distances[0][0]

        qlabels, distances = store1.query(
            embeddings=embeddings[0], top_k=max_elements)
        assert 1 == len(qlabels)
        assert 1 == len(distances)
        assert max_elements == len(qlabels[0])
        assert max_elements == len(distances[0])
        assert labels[0] == qlabels[0][0]
        assert 0.0 == distances[0][0]
        qlabels[0].sort()
        assert all([a == b for a, b in zip(qlabels[0], labels)])

        os.remove(index_path)

    def test_negative_cases(self):
        dim = 384
        max_elements = 5
        space = "cosine"
        store = get_vector_store("hnswlib", dim, space, max_elements)

        # negative test: num_elements > max_elements
        num_elements = max_elements + 1
        embeddings = np.float32(np.random.random((num_elements, dim)))
        labels = np.arange(num_elements)

        with pytest.raises(RuntimeError):
            store.add(embeddings, labels)

