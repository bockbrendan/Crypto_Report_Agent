from datetime import datetime
from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        max_length=500,
        examples=["analyze solana", "ethereum risk report", "is chainlink safe?"],
    )


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str
