from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class chunk_schema(BaseModel):
    chunk_id: str
    source_doc_id: str

    content: str
    content_type: str

    poem_structure: Optional[dict]
    table_info: Optional[dict]

    metadata: dict

    position: dict

    embedding: Optional[List[float]] = None

    search_text: str

    token_count: int
    char_count: int

    has_overlap: bool

    model_version: str

    is_active: bool = True

    ingested_at: datetime