from vectorstore.vectorstore import VectorStore

def get_vector_store(
    store_name: str, dim: int, space: str, max_elements: int,
) -> VectorStore:
    match store_name:
        case "hnswlib":
            from vectorstore.providers.hnswlib_store import HnswlibStore
            return HnswlibStore(dim, space, max_elements)
        case _:
            return ValueError(f"Unsupported vector store: {store_name}")

