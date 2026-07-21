from pydantic import BaseModel, Field
from typing import List, Optional

# ── Constants ────────────────────────────────────────────────────
EXPECTED_SNAPSHOT_SCHEMA = "literature_work_snapshot.v1"
EXPECTED_AUTHOR_SCHEMA = "literature_author_snapshot.v1"
SOURCE_BE_SYNC = "be_sync"


class WorkData(BaseModel):
    id: str
    title: str
    slug: str
    original_title: Optional[str] = None
    genre: str
    sub_genre: Optional[str] = None
    period: str
    grade: int
    semester: int
    publish_year: Optional[int] = None
    summary: Optional[str] = None
    cover_url: Optional[str] = None
    is_published: bool
    historical_context: Optional[str] = None
    realistic_value: Optional[str] = None
    humanistic_value: Optional[str] = None
    artistic_value: Optional[str] = None
    famous_quote: Optional[str] = None
    quote_attribution: Optional[str] = None


class AuthorData(BaseModel):
    id: str
    name: str
    pen_name: Optional[str] = None
    slug: str
    birth_year: Optional[int] = None
    death_year: Optional[int] = None
    period: str
    bio: Optional[str] = None
    portrait_url: Optional[str] = None


class AuthorRefData(BaseModel):
    id: str
    slug: str
    name: str


class SectionData(BaseModel):
    id: str
    number: int
    title: str
    content: str
    content_type: str
    word_count: Optional[int] = None


class CommentaryData(BaseModel):
    id: str
    title: str
    content: str
    commentator_name: Optional[str] = None
    commentator_type: Optional[str] = None
    source_title: Optional[str] = None
    source_url: Optional[str] = None
    published_year: Optional[int] = None
    display_order: int
    is_featured: bool
    is_published: bool


class TagData(BaseModel):
    id: str
    name: str
    slug: str
    description: Optional[str] = None


class WorkSnapshot(BaseModel):
    schema_version: str
    synced_at: str
    work: WorkData
    author_ref: AuthorRefData
    sections: List[SectionData] = []
    commentaries: List[CommentaryData] = []
    tags: List[TagData] = []


class AuthorSnapshot(BaseModel):
    schema_version: str
    synced_at: str
    author: AuthorData


class SyncResponse(BaseModel):
    success: bool
    work_slug: Optional[str] = None
    author_slug: Optional[str] = None
    chunks_upserted: int
    chunks_deactivated: int
    detail: Optional[str] = None
