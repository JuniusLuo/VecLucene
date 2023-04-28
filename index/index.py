import os
import io
from typing import List
import uuid
import lucene

from java.nio.file import Files, Path
from org.apache.lucene.analysis import Analyzer
from org.apache.lucene.document import \
    Document, Field, NumericDocValuesField, StringField, TextField
from org.apache.lucene.index import \
    DirectoryReader, IndexWriter, IndexWriterConfig, Term
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.search import IndexSearcher, TermQuery
from org.apache.lucene.store import FSDirectory

from index.docs import VLScoreDoc
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


class Index:
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
        config.setOpenMode(IndexWriterConfig.OpenMode.CREATE)
        self.writer = IndexWriter(fs_dir, config)

        # initialize the IndexSearcher from the writer
        reader = DirectoryReader.open(self.writer)
        self.searcher = IndexSearcher(reader)

        # initialize the vector index
        self.vector_index = VectorIndex(
            vector_dir, model_provider, vector_store)

        # get the latest vector index version from Lucene
        self.vector_index_version = 0
        term = Term(FIELD_DOC_ID, SYS_DOC_ID_VECTOR_INDEX)
        q = TermQuery(term)
        # doc may not exist if no doc is added to the index
        docs = self.searcher.search(q, 1).scoreDocs
        if len(docs) == 1:
            # get the latest vector index version
            doc = self.searcher.doc(docs[0].doc)
            self.vector_index_version = doc.get(FIELD_VECTOR_INDEX_VERSION)

            # load the existing vectors
            self.vector_index.load(self.vector_index_version)


    def close(self):
        """
        Close the index. The user must call commit before close, to make sure
        the possible in-memory changes are committed.
        """
        self.writer.close()
        self.searcher.getIndexReader().close()


    def add(self, doc_path: str, fields: List[Field]) -> str:
        """
        Add a doc to the index. The doc file must be a plain text file.
        This function automatically generates the embeddings for the doc text.
        
        Return the document id.
        """
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
        # flush the vector index. TODO support multiple index files.
        self.vector_index.save(self.vector_index_version + 1)

        # update the latest vector index version as the special doc0 in Lucene
        doc = Document()
        vector_version_field = NumericDocValuesField(
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


    def search(
        self, query_string: str, top_k: int, vector_weight: float = 1.5,
    ) -> List[VLScoreDoc]:
        """
        Take the query string, search over the doc content (text) and return
        the top docs. The search will include both the traditional inverted
        search and vector search.

        Args:
            vector_weight: the score weight for the vector score. Adjust
            it to tune the score between vector search and traditional search.
        """
        # TODO
        # - support searching other fields, such as title.
        # - support more Lucene query abilities vs natural language search
        #   like gmail. For example, user inputs "a query string. field:value",
        #   automatically search the query string over all invert/vector
        #   indexed fields, and search the specified field.
        # - support retrieving the specified fields.
        # - support pagination search.
        # - etc.

        # search the FIELD_DOC_TEXT only

        # do vector search. skip normalization for vector scores, as
        # vector_index normalizes the distances and then merges the scores
        # for the chunks that belongs to the same doc.
        vector_score_docs = self.vector_index.search(query_string, top_k)

        # normalize the scores, simply use Min-Max Scaling.
        # TODO may support other normalization algorithms.

        # do traditional inverted search
        analyzer = self.writer.getConfig().getAnalyzer()
        parser = QueryParser(FIELD_DOC_TEXT, analyzer)
        query = parser.parse(query_string)
        invert_score_docs = self.searcher.search(query, top_k).scoreDocs
        if len(invert_score_docs) == 0:
            return vector_score_docs

        # find docs in the inverted index, merge with the vector score docs
        merged_score_docs: Dict[str, VLScoreDoc] = {}
        # initialize with vector_score_docs
        for score_doc in vector_score_docs:
            merged_score_docs[score_doc.doc_id] = score_doc

        # fetch doc ids and normalize scores, and merge
        min_score = invert_score_docs[len(invert_score_docs)-1].score
        max_min = invert_score_docs[0] - min_score
        for score_doc in invert_score_docs:
            # normalize score
            if max_min == 0:
                score = 1.0
            else:
                score = (score_doc.score - min_score) / max_min

            # get doc id
            doc = searcher.doc(score_doc.doc)
            doc_id = doc.get(FIELD_DOC_ID)

            if doc_id in merged_score_docs:
                # vector search gets the same doc, merge the scores
                vl_score_doc = merged_score_docs[doc_id]
                score += vl_score_doc.score
                vl_score_doc.vector_ratio = vl_score_doc.score / score
                vl_score_doc.score = score
                continue

            # vector search does not get the doc, add it
            vl_score_doc = VLScoreDoc(
                doc_id=doc_id, score=score, vector_ratio=0.0)
            merged_score_docs[doc_id] = vl_score_doc

        # sort by scores
        score_docs: List[VLScoreDoc] = []
        for score_doc in merged_score_docs.values():
            score_docs.append(score_doc)

        score_docs.sort(key=lambda s:s.score)

        if len(score_docs) > top_k:
            return score_docs[:top_k]

        return score_docs
