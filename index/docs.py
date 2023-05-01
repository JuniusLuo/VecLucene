from pydantic import BaseModel

class DocChunkScore(BaseModel):
    doc_id: str
    offset: int
    length: int
    score: float

