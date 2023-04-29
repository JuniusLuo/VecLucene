from pydantic import BaseModel

# The VectorLucene ScoreDoc
class VLScoreDoc(BaseModel):
    doc_id: str
    score: float
    # The ratio of the vector score in the score, 1 means only vector score,
    # 0 means only inverted score, 0.6 means 60% vector score.
    vector_ratio: float
    # TODO add the offset and length for all doc chunks. The user could fetch
    # the text and send to a model, such as ChatGPT, to generate the answer.
