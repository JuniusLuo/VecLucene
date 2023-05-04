from enum import Enum
from pydantic import BaseModel
from typing import List, Optional

from index.docs import DocChunkScore

class QueryType(str, Enum):
    vector = "vector"
    lucene = "lucene"


class QueryRequest(BaseModel):
    # the query string. 
    # for lucene search, input the string supported by Lucene QueryParser.
    # for vector search, simply input a string. TODO support QueryParser.
    query: str
    query_type: Optional[QueryType] = QueryType.vector
    top_k: Optional[int] = 3


# for now, simply return DocChunkScore.
# TODO include the matched text, for such as highlight, etc.
# TODO add auto QA ability. For a question, the server automatically sends the
# top_k chunk texts as context to the QA model, such as ChatGPT, and includes
# the answer in the response.
class QueryResponse(BaseModel):
    doc_scores: List[DocChunkScore]
