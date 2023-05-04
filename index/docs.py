from pydantic import BaseModel
from typing import List, Optional

# TODO support more attributes, such as tokenize, DocValuesType, etc.
class DocField(BaseModel):
    name: str # field name
    string_value: Optional[str] = None
    numeric_value: Optional[int] = None
    float_value: Optional[float] = None


# All doc fields except the doc text. There are two reserved fields:
# 1. The reserved field name for the doc text, "doc_text", defined by
# index.FIELD_DOC_TEXT. This field name should not be used by application.
# 2. The reserved field name for the doc id, "doc_id", defined by
# index.FIELD_DOC_ID. If doc_id is not specified in the request, server will
# automatically generate a unique id for the doc.
#
# For now, all fields are stored and not indexed. Only the doc contents are
# indexed and also stored. TODO allow indexing more fields, such as title.
class DocFields(BaseModel):
    fields: List[DocField]


class DocChunkScore(BaseModel):
    doc_id: str
    offset: int
    length: int
    score: float

