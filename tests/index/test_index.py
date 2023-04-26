import os
import lucene
import pytest
from typing import List
import shutil

from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Field, StringField

from index.index import Index

class TestIndex:
    def test_index(self):
        # test index with 2 files, cover the mapping of doc ids and labels
        ut_dir = "./tests/index/utdir-index"
        if os.path.exists(ut_dir):
            # remove the possible garbage by previous failed test
            shutil.rmtree(ut_dir)
        os.mkdir(ut_dir)

        lucene.initVM(vmargs=['-Djava.awt.headless=true'])

        analyzer = StandardAnalyzer()
        index = Index(ut_dir, analyzer, "sentence_transformer", "hnswlib")

        try:
            doc_path1 = "./tests/testfiles/single_sentence.txt"
            fields: List[Field] = []
            pathField = StringField("path", doc_path1, Field.Store.YES)
            fields.append(pathField)

            doc_id1 = index.add(doc_path1, fields)

            index.commit()
        finally:
            index.close()

        # cleanup
        #shutil.rmtree(ut_dir)

