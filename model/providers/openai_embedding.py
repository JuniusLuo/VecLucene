from typing import List
from openai import Embedding
from tenacity import retry, wait_random_exponential, stop_after_attempt

from model.model import Model

class OpenAIEmbeddingModel(Model):
    # https://platform.openai.com/docs/guides/embeddings/what-are-embeddings
    model_name: str
    max_token_size: int
    dim: int

    def __init__(self):
        self.model_name = "text-embedding-ada-002"
        # what is the best token size? chatgpt-retrieval-plugin uses 200
        self.max_token_size = 256
        self.dim = 1536

    def get_max_token_size(self) -> int:
        """
        Return the max token for the text.
        """
        return self.max_token_size

    def get_dim(self) -> int:
        """
        Return the embedding dimension
        """
        return self.dim

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3))
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Takes in a list of texts and returns a list of embeddings for each text.
        """
        # Call the OpenAI API to get the embeddings
        response = Embedding.create(input=texts, model=self.model_name)

        # Extract the embedding data from the response
        data = response["data"]  # type: ignore
    
        # Return the embeddings as a list of lists of floats
        return [result["embedding"] for result in data]

    def set_model(self, model_name: str, max_token_size: int, dim: int):
        """
        Set to use the specified model.
        """
        raise NotImplementedError

