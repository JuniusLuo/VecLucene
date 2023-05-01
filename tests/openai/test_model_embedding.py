import os
import pytest
import time

from model.factory import get_model

class TestOpenAIEmbeddingModel():
    def test_embeddings(self):
        m = get_model("openai_embedding")
        assert 256 == m.get_max_token_size()
        assert 1536 == m.get_dim()

        sentences = ['A person is eating food.',
                     'A person is eating a piece of bread.',
                     'A person is riding a horse.',
                     'A person is riding a white horse on an enclosed ground.']

        # example run time on a MacBook.
        # run the test first time, get embeddings time: 0.48015683237463236
        # run the second time, get embeddings time: 0.25255241710692644
        start = time.monotonic()
        embeddings = m.get_embeddings(sentences)
        assert len(sentences) == len(embeddings)
        assert m.get_dim() == len(embeddings[0])
        dur = time.monotonic() - start
        print(f"get embeddings time: {dur}")

        with pytest.raises(NotImplementedError):
            m.set_model("model", 1, 1)
