import os
import logging
import pickle
from pydantic import BaseModel
from typing import List, Dict

import tiktoken

from index.docs import DocChunkScore
from model.model import Model
from model.factory import get_model
from vectorstore.vectorstore import Space, VectorStore
from vectorstore.factory import get_vector_store

DEFAULT_SPACE = Space.l2 # The default space
DEFAULT_VECTOR_FILE_MAX_ELEMENTS = 5000 # The max elements in one vector file
MIN_CHUNK_SIZE_CHARS = 350  # The minimum size of each text chunk in characters
MIN_CHUNK_LENGTH_TO_EMBED = 5  # Discard chunks shorter than this
EMBEDDINGS_BATCH_SIZE = 128 # The number of embeddings to request at a time

# Global tokenizer
default_tokenizer = tiktoken.get_encoding("cl100k_base")


# The metadata for a document chunk
class ChunkMetadata(BaseModel):
    offset: int # the chunk's start offset in the doc
    length: int # the length of the chunk text
    label: int # the label of the chunk embedding in the vector store


# The id to uniquely define a document chunk in the vector index
class ChunkId(BaseModel):
    doc_id: str
    offset: int
    length: int


class VectorIndexMetadata(BaseModel):
    elements: int
    last_label: int


class VectorIndex:
    store_dir: str # the dir where the index files will be stored

    # use openai cl100k_base as tokenizer.
    tokenizer: tiktoken.core.Encoding

    model: Model # the model to get the embeddings for text

    space: Space
    store: VectorStore # the vector store to store the embeddings

    # the underline vector store usually supports int as label. maintain the
    # mapping between the doc ids and labels. These metadatas are directly
    # persisted using pickle.
    # TODO support the metadata definition change, maybe use other
    # serialization format, such as json, protobuf, etc.
    # key: doc id, value: a list of chunk metadata
    doc_id_to_metas: Dict[str, List[ChunkMetadata]]
    # key: label, value: chunk id
    label_to_chunk_id: Dict[int, ChunkId]
    # the vector index metadata
    metadata: VectorIndexMetadata


    def __init__(
        self,
        store_dir: str,
        model_provider:str,
        vector_store: str,
    ):
        self.store_dir = store_dir
        self.tokenizer = default_tokenizer
        self.model = get_model(model_provider)
        dim = self.model.get_dim()
        # default max elements. TODO support more elements
        max_elements = DEFAULT_VECTOR_FILE_MAX_ELEMENTS
        self.space = DEFAULT_SPACE
        self.store = get_vector_store(
            vector_store, dim, self.space, max_elements)
        self.doc_id_to_metas = {}
        self.label_to_chunk_id ={}
        self.metadata = VectorIndexMetadata(elements=0, last_label=0)


    def load(self, version: int):
        """
        Load the vectors from file.
        """
        file_path = self._get_index_file(version)
        self.store.load(file_path)

        # load the mapping between doc ids and labels
        id_file = self._get_id_to_label_file(version)
        with open(id_file, "rb") as f:
            self.doc_id_to_metas = pickle.load(f)

        label_file = self._get_label_to_id_file(version)
        with open(label_file, "rb") as f:
            self.label_to_chunk_id = pickle.load(f)

        metadata_file = self._get_metadata_file(version)
        with open(metadata_file, "rb") as f:
            self.metadata = pickle.load(f)


    def save(self, version: int):
        """
        Save the vectors to the file.
        """
        file_path = self._get_index_file(version)
        self.store.save(file_path)

        # save the mapping between doc ids and labels
        id_file = self._get_id_to_label_file(version)
        with open(id_file, "wb") as f:
            pickle.dump(self.doc_id_to_metas, f, pickle.HIGHEST_PROTOCOL)

        label_file = self._get_label_to_id_file(version)
        with open(label_file, "wb") as f:
            pickle.dump(self.label_to_chunk_id, f, pickle.HIGHEST_PROTOCOL)

        metadata_file = self._get_metadata_file(version)
        with open(metadata_file, "wb") as f:
            pickle.dump(self.metadata, f, pickle.HIGHEST_PROTOCOL)


    def _get_index_file(self, version: int) -> str:
        return os.path.join(self.store_dir, f"{version}.index")

    def _get_id_to_label_file(self, version: int) -> str:
        return os.path.join(self.store_dir, f"id_to_label_{version}.pkl")

    def _get_label_to_id_file(self, version: int) -> str:
        return os.path.join(self.store_dir, f"label_to_id_{version}.pkl")

    def _get_metadata_file(self, version: int) -> str:
        return os.path.join(self.store_dir, f"metadata_{version}.pkl")

    def _get_chunk_id(self, doc_id: str, offset: int, length: int) -> str:
        return f"{doc_id}_{offset}_{length}"


    def add(self, doc_path: str, doc_id: str):
        """
        Add a doc to the vector index. This function reads the doc text, splits
        the doc to chunk if the doc is large, generates the embeddings for
        chunks and adds the embeddings to the vector store.
        TODO support multi-threads.
        """
        # get embeddings for the doc text
        chunk_embeddings, chunk_metas = self._get_embeddings(doc_path)

        logging.info(
            f"get {len(chunk_embeddings)} embeddings for doc path={doc_path} "
            f"id={doc_id}, last_label={self.metadata.last_label}")

        if len(chunk_embeddings) == 0:
            # doc has no content, return
            return

        # assign the labels to the doc chunks
        label = self.metadata.last_label
        # update index metadata
        self.metadata.last_label += len(chunk_metas)
        self.metadata.elements += len(chunk_metas)

        labels: List[int] = []
        for i, chunk_meta in enumerate(chunk_metas):
            label += 1
            chunk_meta.label = label
            labels.append(label)
            # update the label_to_chunk_id Dict
            self.label_to_chunk_id[label] = ChunkId(
                doc_id=doc_id,
                offset=chunk_meta.offset,
                length=chunk_meta.length,
            )

        # update the doc_id_to_metas
        self.doc_id_to_metas[doc_id] = chunk_metas

        # add embeddings to the store
        self.store.add(chunk_embeddings, labels)


    def _get_embeddings(
        self, doc_path: str, batch_size: int = EMBEDDINGS_BATCH_SIZE,
    ) -> (List[List[float]], List[ChunkMetadata]):
        """
        Split the doc's text into chunks, generate one embedding and metadata
        for each chunk.

        Returns:
            A list of embeddings and metadatas for all chunks in the doc.
        """
        # the embeddings for all chunks in the doc
        chunk_embeddings: List[List[float]] = []
        # the metadata for all chunks in the doc
        chunk_metas: List[ChunkMetadata] = []

        # read the whole file. TODO support pagination for large files.
        with open(doc_path, mode="r", encoding="utf-8") as f:
            text = f.read()

        # return an empty list if the text is empty or whitespace
        if not text or text.isspace():
            return chunk_embeddings, chunk_metas

        # split the doc text to chunks
        chunk_token_size = self.model.get_max_token_size()
        chunk_texts, chunk_metas = self._get_text_chunks(
            doc_path, text, chunk_token_size, MIN_CHUNK_SIZE_CHARS)

        # get embeddings for all chunks
        for i in range(0, len(chunk_texts), EMBEDDINGS_BATCH_SIZE):
            batch_texts = chunk_texts[i:i+EMBEDDINGS_BATCH_SIZE]

            embeddings = self.model.get_embeddings(batch_texts)

            chunk_embeddings.extend(embeddings)

        return chunk_embeddings, chunk_metas


    def _get_text_chunks(
        self,
        doc_path: str, # the doc path, for logging
        text: str, # the doc text
        chunk_token_size: int, # the number of tokens in one chunk
        min_chunk_chars: int, # the minimum size of each text chunk in chars
    ) -> (List[str], List[ChunkMetadata]):
        """
        Split the text into chunks.
        Return a list of texts and metadadatas for all chunks in the text.
        """
        chunk_texts: List[str] = []
        chunk_metas: List[ChunkMetadata] = []

        # tokenize the text
        # according to tiktoken/core.py, "encode_ordinary is equivalent to
        # `encode(text, disallowed_special=())` (but slightly faster)."
        tokens = self.tokenizer.encode_ordinary(text)

        # loop until all tokens are consumed or the max elements are reached
        offset = 0
        while tokens:
            # take the next chunk
            chunk = tokens[:chunk_token_size]

            # decode to text to check whitespace and sentence boundary
            chunk_text = self.tokenizer.decode(chunk)

            # skip the chunk if it is empty or whitespace
            if not chunk_text or chunk_text.isspace():
                # remove from the remaining tokens
                tokens = tokens[len(chunk):]
                # increase the offset
                offset += len(chunk_text)
                continue

            # truncate chunk_text to the last complete sentence (punctation).
            # TODO support other languages, maybe consider such as NLTK.
            last_punc = max(
                chunk_text.rfind("."),
                chunk_text.rfind("?"),
                chunk_text.rfind("!"),
                chunk_text.rfind("\n"),
            )
            if last_punc != -1 and last_punc > min_chunk_chars:
                chunk_text = chunk_text[:last_punc+1]

            chunk_text_len = len(chunk_text)

            # adjust the chunk_text_len if needed.
            # check if some text in the last token is skipped. For example,
            # cl100k_base takes '."[' as one token. If two sentences have this
            # string, 'This sentence."[1] Next sentence.', and "This sentence."
            # is the last sentence, the next offset will not align with tokens,
            # e.g. the next offset will point to the first char in '"[1',
            # while, the decoded text of the next token is '1'.
            chunk_tokens = self.tokenizer.encode_ordinary(chunk_text)
            last_chunk_token = len(chunk_tokens) - 1
            if chunk_tokens[last_chunk_token] != tokens[last_chunk_token]:
                # align chunk_text_len with the last token
                last_token_text = self.tokenizer.decode(
                    chunk_tokens[last_chunk_token:])

                token_text = self.tokenizer.decode(
                    tokens[last_chunk_token:last_chunk_token+1])

                chunk_text_len += len(token_text) - len(last_token_text)

                logging.debug(f"align last_token_text={last_token_text} "
                              f"token_text={token_text}")

            logging.debug(f"offset={offset} chunk_text_len={chunk_text_len}")

            # sanity check
            if text[offset:offset+10] != chunk_text[:10]:
                logging.warning(f"doc_path={doc_path} offset={offset},"
                                f"text chars={text[offset:offset+10]}"
                                f"chunk chars={chunk_text[:20]}")
                raise Exception(
                    f"text and chunk not aligned, {doc_path} offset={offset}")

            # remove any newline characters and strip any leading or trailing
            # whitespaces. Not needed if use NLTK.
            chunk_text_to_append = chunk_text.replace("\n", " ").strip()
            if len(chunk_text_to_append) > MIN_CHUNK_LENGTH_TO_EMBED:
                # add the chunk text
                chunk_texts.append(chunk_text_to_append)

                # add the chunk meta
                chunk_metas.append(ChunkMetadata(
                    offset=offset,
                    length=chunk_text_len,
                    label=0, # initial 0 label, will be assigned later
                ))

            # increase the offset
            offset += chunk_text_len

            # remove the chunk text tokens from the remaining tokens.
            tokens = tokens[last_chunk_token+1:]

        return chunk_texts, chunk_metas


    def delete(self, doc_id: str):
        """
        Delete a doc from the vector index.
        """
        raise NotImplementedError


    def search(self, query_string: str, top_k: int) -> List[DocChunkScore]:
        """
        Take a query string, get embedding for the query string, find the
        similar doc chunks in the store, calculate the scores and return the
        top_k doc chunks.
        The score for a doc chunk is calculated based on the distance to the
        query string embedding.

        Return the top-k doc chunks, sorted in descending order based on score.
        """
        texts: List[str] = []
        texts.append(query_string)
        embeddings = self.model.get_embeddings(texts)

        # check k with the current number of elements. Some store, such as
        # hnswlib, throws RuntimeError if k > elements.
        if top_k > self.metadata.elements:
            top_k = self.metadata.elements

        # query the vector store
        labels, distances = self.store.query(embeddings, top_k)

        # convert distances to scores
        return self._distance_to_scores(labels[0], distances[0])


    def _distance_to_scores(
        self, labels: List[int], distances: List[float],
    ) -> List[DocChunkScore]:
        # Convert the distances to the scores in range (0, 1),
        # higher score means closer.
        chunk_scores: List[DocChunkScore] = []
        for i, label in enumerate(labels):
            if self.space == Space.l2:
                # l2 distance, lower distance means closer
                score = 1 / (1 + distances[i])
            else:
                # ip or cosine distance, higher distance means closer
                score = (1 + distances[i]) / 2

            # get the doc id for the chunk
            chunk_id = self.label_to_chunk_id[label]

            chunk_score = DocChunkScore(
                doc_id=chunk_id.doc_id, offset=chunk_id.offset,
                length=chunk_id.length, score=score)
            chunk_scores.append(chunk_score)

        return chunk_scores
