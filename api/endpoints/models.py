from pydantic import BaseModel
from typing import List

class OutputData(BaseModel):
    waveform: List[List[float]]
    message: str