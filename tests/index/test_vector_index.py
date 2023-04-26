import os
import pytest
from typing import List
import shutil

from index.vector_index import VectorIndex

class TestVectorIndex:
    def test_single_sentence(self):
        index = VectorIndex("./", "sentence_transformer", "hnswlib")

        text = "A person is eating food."
        doc_path = "./tests/testfiles/single_sentence.txt"
        chunk_embeddings, chunk_metas = index._get_embeddings(doc_path)
        assert 1 == len(chunk_embeddings)
        assert 1 == len(chunk_metas)
        assert 0 == chunk_metas[0].offset
        assert len(text)+1 == chunk_metas[0].length
        assert 0 == chunk_metas[0].label

        texts = []
        texts.append(text)
        embeddings = index.model.get_embeddings(texts)
        assert 1 == len(embeddings)
        assert index.model.get_dim() == len(embeddings[0])
        assert all([a == b for a, b in zip(chunk_embeddings[0], embeddings[0])])


    def test_small_file(self):
        index = VectorIndex("./", "sentence_transformer", "hnswlib")

        doc_path = "./tests/testfiles/chatgpt.txt"
        chunk_embeddings, chunk_metas = index._get_embeddings(doc_path)
        assert 28 == len(chunk_embeddings)
        assert 28 == len(chunk_metas)
        # the first meta has offset == 0
        assert 0 == chunk_metas[0].offset
        # _get_embeddings does not assign label
        assert 0 == chunk_metas[0].label

        # test small embedding batch
        chunk_embeddings, chunk_metas = index._get_embeddings(doc_path, 5)
        assert 28 == len(chunk_embeddings)
        assert 28 == len(chunk_metas)
        # the first meta has offset == 0
        assert 0 == chunk_metas[0].offset
        # _get_embeddings does not assign label
        assert 0 == chunk_metas[0].label


    def test_special_files(self):
        index = VectorIndex("./", "sentence_transformer", "hnswlib")

        # test empty file
        doc_path = "./tests/testfiles/empty.txt"
        chunk_embeddings, chunk_metas = index._get_embeddings(doc_path)
        assert 0 == len(chunk_embeddings)
        assert 0 == len(chunk_metas)

        # test file with only whitespaces
        doc_path = "./tests/testfiles/whitespaces.txt"
        chunk_embeddings, chunk_metas = index._get_embeddings(doc_path)
        assert 0 == len(chunk_embeddings)
        assert 0 == len(chunk_metas)

        # test file with only 3 chars
        doc_path = "./tests/testfiles/3chars.txt"
        chunk_embeddings, chunk_metas = index._get_embeddings(doc_path)
        assert 0 == len(chunk_embeddings)
        assert 0 == len(chunk_metas)


    def test_index(self):
        # test index with 2 files, cover the mapping of doc ids and labels
        ut_dir = "./tests/index/utdir-vectorindex"
        if os.path.exists(ut_dir):
            # remove the possible garbage by previous failed test
            shutil.rmtree(ut_dir)
        os.mkdir(ut_dir)

        index = VectorIndex(ut_dir, "sentence_transformer", "hnswlib")

        # add the first file
        text = "A person is eating food."
        doc_path1 = "./tests/testfiles/single_sentence.txt"
        doc_id1 = "doc_id1"
        doc1_chunks = 1
        label1 = 1
        index.add(doc_path1, doc_id1)

        assert doc1_chunks == index.metadata.elements
        assert label1 == index.metadata.last_label

        assert 1 == len(index.doc_id_to_metas)
        assert doc1_chunks == len(index.doc_id_to_metas[doc_id1])
        assert 0 == index.doc_id_to_metas[doc_id1][0].offset
        assert len(text)+1 == index.doc_id_to_metas[doc_id1][0].length
        assert label1 == index.doc_id_to_metas[doc_id1][0].label

        assert 1 == len(index.label_to_chunk_id)
        assert doc_id1 == index.label_to_chunk_id[label1].doc_id
        assert 0 == index.label_to_chunk_id[label1].offset
        assert len(text)+1 == index.label_to_chunk_id[label1].length

        # add the second file
        doc_path2 = "./tests/testfiles/chatgpt.txt"
        doc_id2 = "doc_id2"
        doc2_chunks = 28
        index.add(doc_path2, doc_id2)

        assert doc1_chunks+doc2_chunks == index.metadata.elements
        assert label1+doc2_chunks == index.metadata.last_label

        assert 2 == len(index.doc_id_to_metas)
        # verify doc1 metas
        assert 1 == len(index.doc_id_to_metas[doc_id1])
        assert 0 == index.doc_id_to_metas[doc_id1][0].offset
        assert len(text)+1 == index.doc_id_to_metas[doc_id1][0].length
        assert label1 == index.doc_id_to_metas[doc_id1][0].label
        # verify doc2 metas
        assert doc2_chunks == len(index.doc_id_to_metas[doc_id2])
        assert 0 == index.doc_id_to_metas[doc_id2][0].offset
        for i, chunk_meta in enumerate(index.doc_id_to_metas[doc_id2]):
            assert label1+i+1 == chunk_meta.label

        assert doc1_chunks+doc2_chunks == len(index.label_to_chunk_id)
        # verify doc1 chunk ids
        assert doc_id1 == index.label_to_chunk_id[label1].doc_id
        assert 0 == index.label_to_chunk_id[label1].offset
        assert len(text)+1 == index.label_to_chunk_id[label1].length
        # verify doc2 chunk ids
        for label in range(label1+1, len(index.label_to_chunk_id)):
            assert doc_id2 == index.label_to_chunk_id[label].doc_id

        # save the vectors to file
        version = 1
        index.save(version)

        # load from file
        index1 = VectorIndex(ut_dir, "sentence_transformer", "hnswlib")
        assert 0 == index1.metadata.elements
        assert 0 == index1.metadata.last_label

        index1.load(version)

        assert doc1_chunks+doc2_chunks == index1.metadata.elements
        assert label1+doc2_chunks == index1.metadata.last_label

        assert 2 == len(index1.doc_id_to_metas)
        # verify doc1 metas
        assert 1 == len(index1.doc_id_to_metas[doc_id1])
        assert 0 == index1.doc_id_to_metas[doc_id1][0].offset
        assert len(text)+1 == index1.doc_id_to_metas[doc_id1][0].length
        assert label1 == index1.doc_id_to_metas[doc_id1][0].label
        # verify doc2 metas
        assert doc2_chunks == len(index1.doc_id_to_metas[doc_id2])
        assert 0 == index1.doc_id_to_metas[doc_id2][0].offset
        for i, chunk_meta in enumerate(index1.doc_id_to_metas[doc_id2]):
            assert label1+i+1 == chunk_meta.label

        assert doc1_chunks+doc2_chunks == len(index1.label_to_chunk_id)
        # verify doc1 chunk ids
        assert doc_id1 == index1.label_to_chunk_id[label1].doc_id
        assert 0 == index1.label_to_chunk_id[label1].offset
        assert len(text)+1 == index1.label_to_chunk_id[label1].length
        # verify doc2 chunk ids
        for label in range(label1+1, len(index1.label_to_chunk_id)):
            assert doc_id2 == index1.label_to_chunk_id[label].doc_id

        # cleanup
        shutil.rmtree(ut_dir)
