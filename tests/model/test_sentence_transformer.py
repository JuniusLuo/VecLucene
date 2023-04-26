import os
import pytest
import time
from typing import List
import numpy as np

from model.factory import get_model
#from model.providers.sentence_transformer_model import SentenceTransformerModel

class TestSentTransformerModel():
    def test_embeddings(self):
        """
        simple measure the latency of different models on a MacBook M1Pro.
        python3 -m pytest -s tests/model/test_sentence_transformer.py
        default model load time: 1.4939462076872587
        get embeddings time: 0.05871379096060991
        all-mpnet-base-v2 model load time: 1.011457541026175
        get embeddings time: 0.17692300025373697
        """
        start = time.monotonic()
        stmodel = get_model("sentence_transformer")
        assert 256 == stmodel.get_max_token_size()
        assert 384 == stmodel.get_dim()
        dur = time.monotonic() - start
        print(f"\ndefault model load time: {dur}")

        sentences = ['A person is eating food.',
                     'A person is eating a piece of bread.',
                     'A person is riding a horse.',
                     'A person is riding a white horse on an enclosed ground.']

        start = time.monotonic()
        embeddings = stmodel.get_embeddings(sentences)
        assert len(sentences) == len(embeddings)
        assert stmodel.get_dim() == len(embeddings[0])
        dur = time.monotonic() - start
        print(f"get embeddings time: {dur}")

        # https://huggingface.co/sentence-transformers/all-mpnet-base-v2
        start = time.monotonic()
        stmodel.set_model("all-mpnet-base-v2", 384, 768)
        assert 384 == stmodel.get_max_token_size()
        assert 768 == stmodel.get_dim()
        dur = time.monotonic() - start
        print(f"all-mpnet-base-v2 model load time: {dur}")

        start = time.monotonic()
        embeddings = stmodel.get_embeddings(sentences)
        assert len(sentences) == len(embeddings)
        assert stmodel.get_dim() == len(embeddings[0])
        dur = time.monotonic() - start
        print(f"get embeddings time: {dur}")
