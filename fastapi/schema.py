from pandas import Timestamp
from pydantic import BaseModel
from typing import List, Dict
import json

# Define the structure of the input data
class CommentItem(BaseModel):
    text: str
    timestamp: str  # or datetime if you're using actual datetime objects
    authorId: str

    class Config:
        extra = "ignore"  # Ignore extra fields like authorId

class commentData(BaseModel):
    comments: List[str]


class commentwithTSData(BaseModel):
    comments: List[CommentItem]

class sentimentCount(BaseModel):
    sentiment_counts: Dict[str, int] 

class sentimentItem(BaseModel):
    sentiment: int
    Timestamp: str

class comments(BaseModel):
    comments: List[str]

class SentimentArray(BaseModel):
    sentiment_data: List[sentimentItem]