import os
import lucene
import pytest
from typing import List
import shutil

from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Field, StringField

from index.index import Index

class TestSentenceTransformerWithIndex:
    def test_index(self):
        t = IndexAndSearch()
        t.index_docs_and_search(
            "./tests/index/", "sentence_transformer", "hnswlib")

class IndexAndSearch:
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

            # search lucene only
            query_string = "A person is eating food."
            top_k = 2
            lucene_score_docs = index._search_lucene(query_string, top_k)
            assert 1 == len(lucene_score_docs)
            assert 0 == lucene_score_docs[0].doc

            # search both lucene and vector index
            score_docs = index.search(query_string, top_k)
            assert 1 == len(score_docs)
            assert doc_id1 == score_docs[0].doc_id
            # both vector store and lucene will return the doc
            assert score_docs[0].vector_ratio < 1.0
            assert score_docs[0].vector_ratio > 0.5
            assert score_docs[0].score > 1.0

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
            top_k = 2
            lucene_score_docs = index._search_lucene(query_string, top_k)
            assert 2 == len(lucene_score_docs)

            # search both lucene and vector index
            score_docs = index.search(query_string, top_k)
            assert 2 == len(score_docs)
            assert doc_id1 == score_docs[0].doc_id
            assert doc_id2 == score_docs[1].doc_id
            # both vector store and lucene will return the doc
            assert score_docs[0].vector_ratio < 1.0
            assert score_docs[1].vector_ratio < 1.0
            assert score_docs[0].score > 2.0

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
            top_k = 2
            lucene_score_docs = index._search_lucene(query_string, top_k)
            assert 2 == len(lucene_score_docs)

            # search both lucene and vector index
            score_docs = index.search(query_string, top_k)
            assert 2 == len(score_docs)
            assert doc_id1 == score_docs[0].doc_id
            assert doc_id2 == score_docs[1].doc_id
            # both vector store and lucene will return the doc
            assert score_docs[0].vector_ratio < 1.0
            assert score_docs[1].vector_ratio < 1.0
            assert score_docs[0].score > 2.0

        finally:
            index.close()

        # cleanup
        shutil.rmtree(ut_dir)

