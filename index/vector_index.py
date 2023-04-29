import os
import numpy as np
import pickle
from pydantic import BaseModel
from typing import List, Dict

import tiktoken

from index.docs import VLScoreDoc
from model.model import Model
from model.factory import get_model
from vectorstore.vectorstore import Space, VectorStore
from vectorstore.factory import get_vector_store

DEFAULT_SPACE = Space.l2 # The default space
DEFAULT_VECTOR_FILE_MAX_ELEMENTS = 5000 # The max elements in one vector file
MIN_CHUNK_SIZE_CHARS = 350  # The minimum size of each text chunk in characters
MIN_CHUNK_LENGTH_TO_EMBED = 5  # Discard chunks shorter than this
EMBEDDINGS_BATCH_SIZE = 128 # The number of embeddings to request at a time
MIN_TOP_K = 5 # The minimum top_k to query from the vector store

# Global tokenizer
default_tokenizer = tiktoken.get_encoding("cl100k_base")


# The metadata for a document chunk
class ChunkMetadata(BaseModel):
    offset: int # the chunk's start offset in the doc
    length: int # the length of the chunk text
    label: int # the label of the chunk embedding in the vector store


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
    # TODO support other tokenizers, such as NLTK. Note: NLTK could also split
    # text to sentences.
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
        with open(doc_path, "rb") as f:
            text = f.read().decode("utf-8")

        # return an empty list if the text is empty or whitespace
        if not text or text.isspace():
            return chunk_embeddings, chunk_metas

        # tokenize the text        
        tokens = self.tokenizer.encode(text, disallowed_special=())

        # loop until all tokens are consumed or the max elements are reached
        chunk_size = self.model.get_max_token_size()
        num_chunks = 0
        offset = 0
        batch_chunk_texts: List[str] = []
        while tokens and num_chunks < DEFAULT_VECTOR_FILE_MAX_ELEMENTS:
            # take the next chunk
            chunk = tokens[:chunk_size]

            # decode to text to check whitespace and sentence boundary
            chunk_text = self.tokenizer.decode(chunk)

            # skip the chunk if it is empty or whitespace
            if not chunk_text or chunk_text.isspace():
                # remove from the remaining tokens
                tokens = tokens[len(chunk) :]
                # increase the offset
                offset += len(chunk_text)
                continue

            # truncate the chunk text to the last complete sentence
            last_sentence = max(
                chunk_text.rfind("."),
                chunk_text.rfind("?"),
                chunk_text.rfind("!"),
                chunk_text.rfind("\n"),
            )
            if last_sentence != -1 and last_sentence > MIN_CHUNK_SIZE_CHARS:
                chunk_text = chunk_text[: last_sentence + 1]

            # get the original chunk text length
            chunk_org_len = len(chunk_text)
            # remove any newline characters and strip any leading or trailing
            # whitespaces
            chunk_text_to_append = chunk_text.replace("\n", " ").strip()
            if len(chunk_text_to_append) > MIN_CHUNK_LENGTH_TO_EMBED:
                # append the chunk text to the list of chunks
                batch_chunk_texts.append(chunk_text_to_append)
                # add to chunk_metas
                chunk_metas.append(ChunkMetadata(
                    offset=offset,
                    length=chunk_org_len,
                    label=0,
                ))
                    

            # increase the offset
            offset += chunk_org_len

            # remove the chunk text tokens from the remaining tokens
            start = len(self.tokenizer.encode(chunk_text, disallowed_special=()))
            tokens = tokens[start :]

            # increment the number of chunks
            num_chunks += 1

            # get embeddings for one batch
            if len(batch_chunk_texts) >= batch_size:
                embeddings = self.model.get_embeddings(batch_chunk_texts)
                # add to chunk_embeddings
                chunk_embeddings.extend(embeddings)
                # clear the batch
                batch_chunk_texts.clear()

        # handle the last batch
        if len(batch_chunk_texts) > 0:
            embeddings = self.model.get_embeddings(batch_chunk_texts)
            # add to chunk_embeddings
            chunk_embeddings.extend(embeddings)

        return chunk_embeddings, chunk_metas


    def delete(self, doc_id: str):
        """
        Delete a doc from the vector index.
        """
        raise NotImplementedError


    def search(self, query_string: str, top_k: int) -> List[VLScoreDoc]:
        """
        Take a query string, get embedding for the query string, find the
        similar doc chunks in the store, calculate the scores and return the
        top_k docs.
        The score for a doc is calculated based on the distance to the query
        string embedding. If multiple chunks within a doc are close to the
        query string embedding, the doc's score is the sum of the scores of
        those chunks.

        Return the top-k docs, sorted in descending order based on the score.
        Note: the returned docs may be less than top_k, as multiple chunks may
        belong to one doc.
        """
        texts: List[str] = []
        texts.append(query_string)
        embeddings = self.model.get_embeddings(texts)

        # multiple chunks may belong to one doc. adjust top_k to get enough
        # neighbors from the vector store. TODO is there a better way?
        k = top_k
        if k < MIN_TOP_K:
            k = MIN_TOP_K

        # check k with the current number of elements. Some store, such as
        # hnswlib, throws RuntimeError if k > elements.
        if k > self.metadata.elements:
            k = self.metadata.elements

        # query the vector store
        labels, distances = self.store.query(embeddings, k)

        # convert distances to scores
        if self.space == Space.l2:
            doc_ids, scores = self._l2_distance_to_scores(
                labels[0], distances[0])
        else:
            doc_ids, scores = self._ip_distance_to_scores(
                labels[0], distances[0])

        # sort in the descending order based on the score
        score_docs: List[VLScoreDoc] = []
        for i, doc_id in enumerate(doc_ids):
            score_doc = VLScoreDoc(
                doc_id=doc_id, score=scores[i], vector_ratio=1.0)
            score_docs.append(score_doc)

        score_docs.sort(key=lambda s:s.score, reverse=True)

        return score_docs


    def _l2_distance_to_scores(
        self, labels: List[int], distances: List[float],
    ) -> (List[str], List[float]):
        """
        Convert the l2 distances to the scores.
        Return the list of doc ids and scores.
        """
        # For the L2 distance, lower distance means closer.
        # Get the lowest distance for each doc.
        # key: doc_id, value: the lowest distance for the chunk(s) of one doc
        doc_distances: Dict[str, float] = {}
        for i, label in enumerate(labels):
            distance = distances[i]

            chunk_id = self.label_to_chunk_id[label]
            # update distance if doc_id is not included or has lower distance
            if (chunk_id.doc_id not in doc_distances) or \
                (distance < doc_distances[chunk_id.doc_id]):
                doc_distances[chunk_id.doc_id] = distance

        # convert distances to scores
        scores: List[float] = []
        for distance in doc_distances.values():
            scores.append(1 / (1 + distance))

        return doc_distances.keys(), scores


    def _ip_distance_to_scores(
        self, labels: List[int], distances: List[float],
    ) -> (List[str], List[float]):
        """
        Convert the ip or cosine distances to the scores.
        Return the list of doc ids and scores.
        """
        # For the ip or cosine distances, higher distance means closer.
        # Get the highest distance for each doc.
        # key: doc_id, value: the highest distance for the chunk(s) of one doc
        doc_distances: Dict[str, float] = {}
        for i, label in enumerate(labels):
            distance = distances[i]
            chunk_id = self.label_to_chunk_id[label]
            if (chunk_id.doc_id not in doc_distances) or \
                (distance > doc_distances[chunk_id.doc_id]):
                doc_distances[chunk_id.doc_id] = distance

        # convert distances to scores
        scores: List[float] = []
        for distance in doc_distances.values():
            scores.append((1 + distance) / 2)

        return doc_distances.keys(), scores
