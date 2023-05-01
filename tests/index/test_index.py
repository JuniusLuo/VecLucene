import os
import logging
import lucene
import pytest
import time
from typing import List
import shutil

from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Field, StringField

from index.index import Index

class TestSentenceTransformerWithIndex:
    def test_index(self):
        t = IndexAndSearchTest()
        t.index_docs_and_search(
            "./tests/index/", "sentence_transformer", "hnswlib")

class IndexAndSearchTest:
    def index_docs_and_search(
        self, base_dir: str, model_name: str, vector_store: str,
    ):
        ut_dir = os.path.join(base_dir, "utdir-index")
        if os.path.exists(ut_dir):
            # remove the possible garbage by previous failed test
            shutil.rmtree(ut_dir)
        os.mkdir(ut_dir)

        lucene.initVM(vmargs=['-Djava.awt.headless=true'])

        analyzer = StandardAnalyzer()
        index = Index(ut_dir, analyzer, model_name, vector_store)

        try:
            # step1: add the first file
            doc_path1 = "./tests/testfiles/single_sentence.txt"
            fields: List[Field] = []
            pathField = StringField("path", doc_path1, Field.Store.YES)
            fields.append(pathField)

            doc_id1 = index.add(doc_path1, fields)

            # search lucene
            query_string = "A person is eating food."
            top_k = 3
            start = time.monotonic()
            lucene_score_docs = index.lucene_search(query_string, top_k)
            dur = time.monotonic() - start
            logging.info(f"1 doc, lucene search time: {dur}s")
            assert 1 == len(lucene_score_docs)
            assert doc_id1 == lucene_score_docs[0].doc_id

            # search vector index
            start = time.monotonic()
            vector_score_docs = index.vector_search(query_string, top_k)
            dur = time.monotonic() - start
            logging.info(f"1 doc, vector search time: {dur}s")
            assert 1 == len(vector_score_docs)
            assert doc_id1 == vector_score_docs[0].doc_id
            assert vector_score_docs[0].score > 0.9

            # commit and verify the vector index version
            index.commit()
            vector_index_version = index._get_vector_index_version()
            assert 1 == vector_index_version

            # step2: add the second file
            doc_path2 = "./tests/testfiles/chatgpt.txt"
            fields.clear()
            pathField = StringField("path", doc_path2, Field.Store.YES)
            fields.append(pathField)

            doc_id2 = index.add(doc_path2, fields)

            # search lucene only
            query_string = "A person is eating food."
            top_k = 3
            start = time.monotonic()
            lucene_score_docs = index.lucene_search(query_string, top_k)
            dur = time.monotonic() - start
            logging.info(f"2 docs, lucene search time: {dur}s")
            assert 2 == len(lucene_score_docs)

            # search vector index
            start = time.monotonic()
            vector_score_docs = index.vector_search(query_string, top_k)
            dur = time.monotonic() - start
            logging.info(f"2 docs, vector search time: {dur}s")
            # sentence_transformer returns:
            # [DocChunkScore(doc_id1, offset=0, length=25, score=1.0),
            #  DocChunkScore(doc_id2, offset=15234, length=1172, score=0.34),
            #  DocChunkScore(doc_id2, offset=2219, length=1182, score=0.34)]
            # openai returns, open file, seek and read, the text looks not
            # related to the query_string, not sure why openai scores 0.63
            # [DocChunkScore(doc_id1, offset=0, length=25, score=1.0),
            #  DocChunkScore(doc_id2, offset=15234, length=1172, score=0.63),
            #  DocChunkScore(doc_id2, offset=16406, length=1272, score=0.63)]
            #logging.info(f"=== {vector_score_docs}")
            assert 3 == len(vector_score_docs)
            assert doc_id1 == vector_score_docs[0].doc_id
            assert doc_id2 == vector_score_docs[1].doc_id
            assert doc_id2 == vector_score_docs[2].doc_id
            assert vector_score_docs[0].score > 0.9
            if model_name == "sentence_transformer":
                assert vector_score_docs[1].score < 0.5 # doc2 has low score
                assert vector_score_docs[2].score < 0.5 # doc2 has low score
            if vector_score_docs[1].score > 0.5:
                score = vector_score_docs[1].score
                logging.info(f"{model_name} scores high {score}")

            # commit and verify the vector index version
            index.commit()
            vector_index_version = index._get_vector_index_version()
            assert 2 == vector_index_version

            index.close()

            # step3: reload index
            index = Index(ut_dir, analyzer, model_name, vector_store)
            assert 2 == index.vector_index_version

            # search lucene only
            query_string = "A person is eating food."
            top_k = 3
            start = time.monotonic()
            lucene_score_docs = index.lucene_search(query_string, top_k)
            dur = time.monotonic() - start
            logging.info(f"2 docs, reload, lucene search time: {dur}s")
            assert 2 == len(lucene_score_docs)

            # search vector index
            start = time.monotonic()
            vector_score_docs = index.vector_search(query_string, top_k)
            dur = time.monotonic() - start
            logging.info(f"2 docs, reload, vector search time: {dur}s")
            assert 3 == len(vector_score_docs)
            assert doc_id1 == vector_score_docs[0].doc_id
            assert doc_id2 == vector_score_docs[1].doc_id
            assert doc_id2 == vector_score_docs[2].doc_id
            assert vector_score_docs[0].score > 0.9
            if model_name == "sentence_transformer":
                assert vector_score_docs[1].score < 0.5 # doc2 has low score
                assert vector_score_docs[2].score < 0.5 # doc2 has low score

        finally:
            index.close()

        # cleanup
        shutil.rmtree(ut_dir)

