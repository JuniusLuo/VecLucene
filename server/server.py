from contextlib import asynccontextmanager
from fastapi import FastAPI, File, Form, HTTPException, Depends, Body, UploadFile
import logging
import lucene
import mimetypes
import os
from typing import Optional
import sys
import uvicorn

from index.index import Index
from index.docs import DocField, DocFields, DocChunkScore
from server.api import QueryType, QueryRequest, QueryResponse

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# The embedding model provider: openai_embedding, sentence_transformer.
# If model is set to openai_embedding, please remember to set OPENAI_API_KEY.
ENV_EMBEDDING_MODEL_PROVIDER = os.environ.get("ENV_EMBEDDING_MODEL_PROVIDER")
# The directory to store the lucene and vector index
ENV_INDEX_DIR = os.environ.get("ENV_INDEX_DIR")

DEFAULT_EMBEDDING_MODEL_PROVIDER = "sentence_transformer"
DEFAULT_INDEX_DIR = "./server_index_dir"

embedding_model = DEFAULT_EMBEDDING_MODEL_PROVIDER
index_dir = DEFAULT_INDEX_DIR
if ENV_EMBEDDING_MODEL_PROVIDER is not None \
    and ENV_EMBEDDING_MODEL_PROVIDER != "":
    embedding_model = ENV_EMBEDDING_MODEL_PROVIDER
if ENV_INDEX_DIR is not None and ENV_INDEX_DIR != "":
    index_dir = ENV_INDEX_DIR

# the sub directory under index_dir to store the doc content
index_doc_dir = os.path.join(index_dir, "docs")


def start(host: str, port: int):
    uvicorn.run("server.server:app", host=host, port=port, reload=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # init Index
    global index
    lucene.initVM(vmargs=['-Djava.awt.headless=true'])
    index = Index(index_dir=index_dir,
                  model_provider=embedding_model,
                  vector_store="hnswlib")
    logging.info("start the index")
    yield
    # close the index
    # TODO when use Ctrl+C to stop, close is not called. using the old shutdown
    # does not work as well.
    logging.info("close the index")
    index.close()

app = FastAPI(lifespan=lifespan)


# TODO support creating the index
@app.post("/add_doc")
async def add_doc(
    file: UploadFile = File(...),
    fields: Optional[str] = Form(None),
):
    filename = file.filename
    try:
        # parse the fields
        doc_fields: DocFields = None
        if fields is not None:
            doc_fields = DocFields.parse_raw(fields)

        # save the file text
        doc_path = await save_file_text(file)

        # add file to index
        doc_id = index.add(doc_path, doc_fields)
        return doc_id
    except Exception as e:
        logging.error(f"add doc {filename} error: {e}")
        raise HTTPException(status_code=500, detail=f"str({e})")


@app.post("/commit")
async def commit():
    index.commit()


@app.get(
    "/query",
    response_model=QueryResponse,
)
async def query(
    request: QueryRequest = Body(...),
):
    try:
        docs: List[DocChunkScore] = None
        if request.query_type == QueryType.vector:
            docs = index.vector_search(request.query, request.top_k)
        else:
            docs = index.lucene_search(request.query, request.top_k)

        return QueryResponse(doc_scores=docs)
    except Exception as e:
        logging.error(f"query {request.query} error: {e}")
        raise HTTPException(status_code=500, detail=f"str({e})")


async def save_file_text(file: UploadFile) -> str:
    """
    Extract text from file and save under index_doc_dir.
    Return the absolute file path saved under index_doc_dir.
    """
    # check file type. only support text file now.
    mimetype = file.content_type
    if mimetype is None:
        mimetype, _ = mimetypes.guess_type(file.filename)

    if mimetype != "text/plain" and mimetype != "text/markdown":
        raise ValueError(f"Unsupported file type: {mimetype}")

    # store the file text under index_doc_dir.
    # TODO support other type file, extract the text from the file.
    # TODO for small files, directly store in Lucene.
    doc_path = os.path.join(index_doc_dir, file.filename)
    os.makedirs(os.path.dirname(doc_path), exist_ok=True)

    file_stream = await file.read()

    # TODO if file exists, update doc
    with open(doc_path, "wb") as f:
        f.write(file_stream)

    return doc_path

