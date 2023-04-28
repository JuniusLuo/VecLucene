from typing import List

from sentence_transformers import SentenceTransformer

from model.model import Model

class SentenceTransformerModel(Model):
    model: SentenceTransformer
    max_token_size: int
    dim: int

    def __init__(self):
        """
        https://huggingface.co/blog/mteb, all-mpnet-base-v2 or all-MiniLM-L6-v2
        provide a good balance between speed and performance.

        https://www.sbert.net/docs/pretrained_models.html, test on a V100 GPU.
        all-mpnet-base-v2, model size 420MB, encoding speed 2800 sentence/s.
        all-MiniLM-L6-v2,  model size 80MB,  encoding speed 14200 sentence/s.
        """
        # initialize with the default model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        # https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
        # By default, input text longer than 256 word pieces is truncated.
        self.max_token_size = 256
        self.dim = 384

    def get_max_token_size(self) -> int:
        """
        Return the max token for the text.
        """
        # TODO depending on the tokenizer, 256 word pieces may not equal to
        # 256 tokens.
        return self.max_token_size

    def get_dim(self) -> int:
        """
        Return the embedding dimension
        """
        return self.dim

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Takes in a list of texts and returns a list of embeddings for each text.
        """
        embeddings: List[List[float]] = []
        for text in texts:
            embedding = self.model.encode(text)
            embeddings.append(embedding)

        return embeddings

    def set_model(self, model_name: str, max_token_size: int, dim: int):
        """
        Set to use the specified model.
        """
        self.model = SentenceTransformer(model_name)
        self.max_token_size = max_token_size
        self.dim = dim

