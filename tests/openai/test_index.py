import os
import lucene
import pytest


from tests.index.test_index import IndexAndSearchTest

class TestIndexWithOpenAIAdaModel:
    def test_index(self):
        t = IndexAndSearchTest()
        t.index_docs_and_search("./tests/openai/", "openai_embedding", "hnswlib")
