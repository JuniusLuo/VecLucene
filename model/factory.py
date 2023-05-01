from model.model import Model

def get_model(provider: str) -> Model:
    match provider:
        case "openai_embedding":
            from model.providers.openai_embedding import OpenAIEmbeddingModel
            return OpenAIEmbeddingModel()
        case "sentence_transformer":
            from model.providers.sentence_transformer_model import \
                SentenceTransformerModel
            return SentenceTransformerModel()
        case _:
            raise ValueError(f"Unsupported model provider: {provider}")

