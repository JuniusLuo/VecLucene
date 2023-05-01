import os
import logging
import lucene
from typing import List
import uuid

from java.nio.file import Files, Path
from org.apache.lucene.analysis import Analyzer
from org.apache.lucene.document import \
    Document, Field, StringField, TextField, StoredField
from org.apache.lucene.index import \
    DirectoryReader, IndexWriter, IndexWriterConfig, Term
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.search import IndexSearcher, ScoreDoc, TermQuery
from org.apache.lucene.store import FSDirectory

from index.docs import DocChunkScore
from index.vector_index import VectorIndex

# the reserved field names for the doc
FIELD_DOC_ID = "doc_id"
FIELD_DOC_TEXT = "doc_text"
FIELD_VECTOR_INDEX_VERSION = "vector_index_version"

# the reserved doc ids for the internal usage
# the reserved doc id for the vector index metadata
SYS_DOC_ID_VECTOR_INDEX = "$sys_doc_id_vector_index"


# the subdir for Lucene
SUBDIR_LUCENE = "lucene"
SUBDIR_VECTOR = "vector"


"""
The Index class combines Lucene index with the vector index. It accepts a
document, splits the document content to chunks, generates embeddings for each
chunk using the specified model, persists the embeddings in the vector index
and persists Lucene fields in the Lucene index. Search could search both Lucene
and vector index, and merge the results.
The Index class guarantees the consistency between Lucene index and vector
index, and manages the lifecycle of the documents.
TODO this class is not thread safe for concurrent write and read. The underline
vector store, such as Hnswlib, does not support concurrent write and read.
"""
class Index:
    index_dir: str
    writer: IndexWriter
    searcher: IndexSearcher

    vector_index: VectorIndex
    vector_index_version: int

    def __init__(
        self,
        index_dir: str,
        analyzer: Analyzer,
        model_provider: str,
        vector_store: str,
    ):
        if not os.path.exists(index_dir):
            os.mkdir(index_dir)

        lucene_dir = os.path.join(index_dir, SUBDIR_LUCENE)
        if not os.path.exists(lucene_dir):
            os.mkdir(lucene_dir)

        vector_dir = os.path.join(index_dir, SUBDIR_VECTOR)
        if not os.path.exists(vector_dir):
            os.mkdir(vector_dir)

        # initialize the IndexWriter for Lucene
        fs_dir = FSDirectory.open(Path.of(lucene_dir))
        config = IndexWriterConfig(analyzer)
        config.setOpenMode(IndexWriterConfig.OpenMode.CREATE_OR_APPEND)
        self.writer = IndexWriter(fs_dir, config)
        self.index_dir = index_dir

        # initialize the IndexSearcher from the writer
        reader = DirectoryReader.open(self.writer)
        self.searcher = IndexSearcher(reader)

        # initialize the vector index
        self.vector_index = VectorIndex(
            vector_dir, model_provider, vector_store)

        # get the latest vector index version from Lucene
        self.vector_index_version = self._get_vector_index_version()
        if self.vector_index_version > 0:
            # load the existing vectors
            self.vector_index.load(self.vector_index_version)

        logging.info(f"Initialize the index {index_dir}, "
                     f"vector_index_version={self.vector_index_version}")


    def _get_vector_index_version(self) -> int:
        reader = DirectoryReader.openIfChanged(self.searcher.getIndexReader())
        if reader:
            self.searcher.getIndexReader().close()
            self.searcher = IndexSearcher(reader)

        # doc may not exist if no doc is added to the index
        vector_index_version = 0
        term = Term(FIELD_DOC_ID, SYS_DOC_ID_VECTOR_INDEX)
        q = TermQuery(term)
        docs = self.searcher.search(q, 1).scoreDocs
        if len(docs) > 0:
            # get the latest vector index version
            doc = self.searcher.doc(docs[0].doc)
            field = doc.getField(FIELD_VECTOR_INDEX_VERSION)
            vector_index_version = field.numericValue().longValue()
        return vector_index_version


    def close(self):
        """
        Close the index. The user must call commit before close, to make sure
        the possible in-memory changes are committed.
        """
        self.writer.close()
        self.searcher.getIndexReader().close()
        logging.info("Close the index")


    def add(self, doc_path: str, fields: List[Field]) -> str:
        """
        Add a doc to the index. The doc file must be a plain text file.
        This function automatically generates the embeddings for the doc text.
        
        Return the document id.
        """
        # TODO support only a limited number of docs, e.g. less than
        # vector_index.DEFAULT_VECTOR_FILE_MAX_ELEMENTS. One vector index
        # element is one doc chunk.
        # TODO support embeddings for other fields, such as title, etc.
        # TODO support other type files, such as pdf, etc, e.g. extract text
        # from file, write to a temporary text file, and then pass the
        # temporary text file to this function.
        # TODO support small files, such as 10KB. no need to persist the file
        # to a temporary file, when running as http server.

        # get doc_id from fields, assign a unique id to doc if doc_id is None
        doc_id = ""
        for field in fields:
            if field.name() == FIELD_DOC_ID:
                doc_id = field.stringValue()
                break

        if doc_id == "":
            doc_id = str(uuid.uuid4())
            fields.append(StringField(FIELD_DOC_ID, doc_id, Field.Store.YES))

        # add the doc to vector writer
        self.vector_index.add(doc_path, doc_id)

        # add the doc to Lucene
        self._add_to_lucene(doc_path, fields)

        logging.debug(f"add doc id={doc_id} to index")
        return doc_id


    def _add_to_lucene(self, doc_path: str, fields: List[Field]):
        file_path = Path.of(doc_path)
        br = Files.newBufferedReader(file_path)
        try:
            doc = Document()

            for field in fields:
                doc.add(field)

            text_field = TextField(FIELD_DOC_TEXT, br)
            doc.add(text_field)

            self.writer.addDocument(doc)
        finally:
            br.close()


    def commit(self):
        # flush the vector index. TODO delete the older vector index files.
        self.vector_index.save(self.vector_index_version + 1)

        # update the latest vector index version as the special doc0 in Lucene
        doc = Document()
        doc_id_field = StringField(
            FIELD_DOC_ID, SYS_DOC_ID_VECTOR_INDEX, Field.Store.YES)
        doc.add(doc_id_field)
        vector_version_field = StoredField(
            FIELD_VECTOR_INDEX_VERSION, self.vector_index_version + 1)
        doc.add(vector_version_field)
        if self.vector_index_version == 0:
            # create the vector doc
            self.writer.addDocument(doc)
        else:
            # update the vector doc
            term = Term(FIELD_DOC_ID, SYS_DOC_ID_VECTOR_INDEX)
            self.writer.updateDocument(term, doc)

        # commit Lucene
        self.writer.commit()

        # successfully commit both vector and lucene indexes
        self.vector_index_version += 1
        logging.info(f"Commit the index {self.index_dir}, "
                     f"vector_index_version={self.vector_index_version}")


    def vector_search(
        self, query_string: str, top_k: int,
    ) -> List[DocChunkScore]:
        """
        Take the query string, search over the doc content (text) and return
        the top docs. The search will include both the traditional inverted
        search and vector search.
        """
        # TODO
        # - support index and search other fields, such as title.
        # - support more Lucene query abilities vs natural language search
        #   like gmail. For example, user inputs "a query string. field:value",
        #   automatically search the query string over all invert/vector
        #   indexed fields, and search the specified field.
        # - support retrieving the specified fields.
        # - etc.

        doc_chunk_scores = self.vector_index.search(query_string, top_k)

        logging.debug(
            f"vector search query=\'{query_string}\' docs={doc_chunk_scores}")
        return doc_chunk_scores


    def lucene_search(
        self, query_string: str, top_k: int,
    ) -> List[DocChunkScore]:
        # TODO support concurrent reads
        reader = DirectoryReader.openIfChanged(self.searcher.getIndexReader())
        if reader:
            self.searcher.getIndexReader().close()
            self.searcher = IndexSearcher(reader)

        analyzer = self.writer.getConfig().getAnalyzer()
        parser = QueryParser(FIELD_DOC_TEXT, analyzer)
        query = parser.parse(query_string)

        lucene_score_docs = self.searcher.search(query, top_k).scoreDocs

        doc_chunk_scores: List[DocChunkScore] = []
        for score_doc in lucene_score_docs:
            # get doc id
            doc = self.searcher.doc(score_doc.doc)
            doc_id = doc.get(FIELD_DOC_ID)

            # TODO get the offset and length via TermVector or Highlighter
            doc_chunk_score = DocChunkScore(
                doc_id=doc_id, offset=0, length=0, score=score_doc.score)
            doc_chunk_scores.append(doc_chunk_score)

        logging.debug(
            f"lucene search query=\'{query_string}\' docs={doc_chunk_scores}")
        return doc_chunk_scores
