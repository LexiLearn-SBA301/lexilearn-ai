"""
SyncService — Xử lý đồng bộ dữ liệu tác phẩm từ BE (Java/Postgres) sang AI (MongoDB).

Luồng:
    1. Nhận WorkSnapshot (full snapshot) từ BE
    2. Deactivate tất cả chunk cũ (cả docx_ingest lẫn be_sync) theo work_slug
    3. Build chunks mới từ snapshot (meta_overview, sections, commentaries, bio, values...)
    4. Batch embed → Upsert Mongo

Module này HOÀN TOÀN ĐỘC LẬP với IngestService (DOCX pipeline).
"""

import re
import logging
from typing import List, Dict, Any, Optional

from core.embedder import Embedder
from core.mongo_writer import MongoWriter
from models.chunk_schema import ChunkSchema, ChunkPosition, ChunkMetadata
from schemas.sync_schema import (
    WorkSnapshot,
    AuthorSnapshot,
    EXPECTED_SNAPSHOT_SCHEMA,
    EXPECTED_AUTHOR_SCHEMA,
    SOURCE_BE_SYNC,
)
from services.rag_service import remove_vietnamese_accents

logger = logging.getLogger("rag-service.services.sync-service")

# ── Constants ────────────────────────────────────────────────────
SOURCE_DOC_PREFIX = "be_sync"
DEFAULT_CONTENT_TYPE = "MIXED"
MAX_SPLIT_TOKENS = 800
CHARS_PER_TOKEN = 3  # Approximation for Vietnamese text


class SyncService:
    """
    SyncService handles syncing documents from BE to AI Service.
    It takes a WorkSnapshot, generates appropriate chunks, computes embeddings,
    and upserts them into MongoDB while maintaining backward compatibility
    with DOCX ingested chunks.
    """

    def __init__(self, writer: MongoWriter, embedder: Embedder) -> None:
        """
        Inject dependencies thay vì tự tạo connection.
        Được khởi tạo 1 lần ở lifespan() trong main.py.
        """
        self.writer = writer
        self.embedder = embedder
        
        # Load taxonomy for beautiful display names in meta overview
        import os, json
        config_path = os.path.join(os.path.dirname(__file__), "..", "config", "genre_taxonomy.json")
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                taxonomy = json.load(f)
                self.genre_labels = {k: v["label"] for k, v in taxonomy.get("genres", {}).items()}
                self.sub_genre_labels = taxonomy.get("sub_genre_labels", {})
        except Exception as e:
            logger.warning(f"Could not load genre_taxonomy.json: {e}")
            self.genre_labels = {}
            self.sub_genre_labels = {}
            
        self.period_labels = {
            "dan_gian": "dân gian",
            "trung_dai": "trung đại",
            "can_dai": "cận đại",
            "hien_dai": "hiện đại"
        }
        
        logger.info("SyncService initialized successfully.")

    # ── Public API ───────────────────────────────────────────────

    def sync_work(self, payload: WorkSnapshot) -> Dict[str, Any]:
        """
        Process the sync payload:
        1. Deactivate all existing chunks for this work (from both BE and DOCX ingest)
        2. Build new chunks from the snapshot
        3. Batch embed
        4. Upsert into Mongo
        """
        # Convert slugs to snake_case
        payload.work.slug = payload.work.slug.replace("-", "_")
        payload.author_ref.slug = payload.author_ref.slug.replace("-", "_")
        work_slug = payload.work.slug

        # 1. Deactivate old chunks (cả docx_ingest lẫn be_sync)
        chunks_deactivated = self.writer.deactivate_by_work_slug(work_slug)
        logger.info(f"Deactivated {chunks_deactivated} old chunks for work '{work_slug}'")

        # 2. Build chunks
        chunks_to_upsert, texts_to_embed = self._build_all_chunks(payload)

        # 3. Batch Embed
        if texts_to_embed:
            logger.info(f"Embedding {len(texts_to_embed)} chunks for work '{work_slug}'")
            embeddings = self.embedder.embed_documents(texts_to_embed)
            for chunk_doc, emb in zip(chunks_to_upsert, embeddings):
                chunk_doc.embedding = emb

        # 4. Upsert into Mongo
        chunks_upserted = 0
        for chunk_doc in chunks_to_upsert:
            self.writer.upsert_chunk(chunk_doc)
            chunks_upserted += 1

        logger.info(f"Successfully upserted {chunks_upserted} chunks for work '{work_slug}'")

        return {
            "success": True,
            "work_slug": work_slug,
            "chunks_upserted": chunks_upserted,
            "chunks_deactivated": chunks_deactivated,
        }

    def delete_work(self, work_slug: str) -> Dict[str, Any]:
        """Soft delete a work's chunks (mark is_active = false)."""
        work_slug = work_slug.replace("-", "_")
        chunks_deactivated = self.writer.deactivate_by_work_slug(work_slug)
        logger.info(f"Deactivated {chunks_deactivated} chunks for work '{work_slug}'")

        return {
            "success": True,
            "work_slug": work_slug,
            "chunks_upserted": 0,
            "chunks_deactivated": chunks_deactivated,
        }

    def sync_author(self, payload: AuthorSnapshot) -> Dict[str, Any]:
        """
        Process the sync payload for an author.
        """
        # Convert slug to snake_case
        payload.author.slug = payload.author.slug.replace("-", "_")
        author_slug = payload.author.slug

        chunks_deactivated = self.writer.deactivate_by_author_slug(author_slug)
        logger.info(f"Deactivated {chunks_deactivated} old chunks for author '{author_slug}'")

        if not payload.author.bio:
            return {
                "success": True,
                "author_slug": author_slug,
                "chunks_upserted": 0,
                "chunks_deactivated": chunks_deactivated,
            }

        # Build chunk
        bio_parts = []
        pen_name_str = f" (Bút danh: {payload.author.pen_name})" if payload.author.pen_name else ""
        life_span = ""
        if payload.author.birth_year and payload.author.death_year:
            life_span = f", sinh năm {payload.author.birth_year}, mất năm {payload.author.death_year}"
        elif payload.author.birth_year:
            life_span = f", sinh năm {payload.author.birth_year}"

        period_label = self.period_labels.get(payload.author.period, payload.author.period)
        bio_parts.append(f"{payload.author.name}{pen_name_str}{life_span} thuộc giai đoạn văn học {period_label}.")
        bio_parts.append(payload.author.bio)

        bio_text = " ".join(bio_parts)
        search_text = remove_vietnamese_accents(bio_text)
        chunk_id = f"{author_slug}__author_bio__001"

        metadata = ChunkMetadata(
            schema_version="literature_seed.v1",
            source=SOURCE_BE_SYNC,
            author_id=payload.author.id,
            author_name=payload.author.name,
            author_slug=author_slug,
            author_period=payload.author.period,
            chunk_category="author_bio",
            content_type=DEFAULT_CONTENT_TYPE,
        )

        position = ChunkPosition(page=1, chunk_index=0, total_chunks=1)

        char_count = len(bio_text)
        token_count = char_count // CHARS_PER_TOKEN

        chunk_doc = ChunkSchema(
            chunk_id=chunk_id,
            source_doc_id=f"{SOURCE_DOC_PREFIX}:{author_slug}_bio",
            content=bio_text,
            content_type=DEFAULT_CONTENT_TYPE,
            position=position,
            metadata=metadata,
            token_count=token_count,
            char_count=char_count,
            has_overlap=False,
            search_text=search_text,
            model_version=self.embedder.model_name,
            is_active=True,
            embedding=[],
        )

        embeddings = self.embedder.embed_documents([bio_text])
        chunk_doc.embedding = embeddings[0]

        self.writer.upsert_chunk(chunk_doc)
        logger.info(f"Successfully upserted 1 bio chunk for author '{author_slug}'")

        return {
            "success": True,
            "author_slug": author_slug,
            "chunks_upserted": 1,
            "chunks_deactivated": chunks_deactivated,
        }

    def delete_author(self, author_slug: str) -> Dict[str, Any]:
        author_slug = author_slug.replace("-", "_")
        chunks_deactivated = self.writer.deactivate_by_author_slug(author_slug)
        logger.info(f"Deactivated {chunks_deactivated} chunks for author '{author_slug}'")

        return {
            "success": True,
            "author_slug": author_slug,
            "chunks_upserted": 0,
            "chunks_deactivated": chunks_deactivated,
        }

    # ── Chunk Building ───────────────────────────────────────────

    def _build_all_chunks(self, payload: WorkSnapshot):
        """Duyệt toàn bộ snapshot, build danh sách chunks + texts song song."""
        work_slug = payload.work.slug
        tag_text = " ".join(t.name for t in payload.tags) if payload.tags else ""

        chunks: List[ChunkSchema] = []
        texts: List[str] = []

        def _add(content: str, category: str, *,
                 section_id: Optional[str] = None,
                 section_slug: Optional[str] = None,
                 section_title: Optional[str] = None,
                 section_order: Optional[int] = None,
                 content_type: str = DEFAULT_CONTENT_TYPE,
                 chunk_index_prefix: str = ""):
            """Inner helper: split if needed, build ChunkSchema, append to lists."""
            content = (content or "").strip()
            if not content:
                return

            split_contents = self._split_long_text(content)

            for idx, part_content in enumerate(split_contents):
                part_idx_str = f"{idx + 1:03d}"

                # Deterministic chunk_id
                if section_slug:
                    chunk_id = f"{work_slug}__{category}__{section_slug}__{part_idx_str}"
                elif chunk_index_prefix:
                    chunk_id = f"{work_slug}__{category}__{chunk_index_prefix}__{part_idx_str}"
                else:
                    chunk_id = f"{work_slug}__{category}__{part_idx_str}"

                search_text = remove_vietnamese_accents(part_content + " " + tag_text)

                metadata = ChunkMetadata(
                    schema_version="literature_seed.v1",  # Internal chunk schema, not snapshot schema
                    source=SOURCE_BE_SYNC,
                    work_id=payload.work.id,
                    work_title=payload.work.title.upper(),
                    work_slug=work_slug,
                    author_id=payload.author_ref.id,
                    author_name=payload.author_ref.name,
                    author_slug=payload.author_ref.slug,
                    work_period=payload.work.period,
                    genre=payload.work.genre,
                    sub_genre=payload.work.sub_genre,
                    grade=payload.work.grade,
                    semester=payload.work.semester,
                    publish_year=payload.work.publish_year,
                    chunk_category=category,
                    section_id=section_id,
                    section_slug=section_slug,  # Already slugified by caller
                    section_title=section_title,
                    section_order=section_order,
                    content_type=content_type.upper(),
                )

                char_count = len(part_content)
                token_count = char_count // CHARS_PER_TOKEN

                position = ChunkPosition(
                    page=1,
                    chunk_index=idx,
                    total_chunks=len(split_contents),
                )

                chunk_doc = ChunkSchema(
                    chunk_id=chunk_id,
                    source_doc_id=f"{SOURCE_DOC_PREFIX}:{work_slug}",
                    content=part_content,
                    content_type=content_type.upper(),
                    position=position,
                    metadata=metadata,
                    token_count=token_count,
                    char_count=char_count,
                    has_overlap=False,
                    search_text=search_text,
                    model_version=self.embedder.model_name,
                    is_active=True,
                    embedding=[],  # Filled later by batch embed
                )

                chunks.append(chunk_doc)
                texts.append(part_content)

        # ── 2.1 Meta Overview (luôn tạo) ────────────────────────
        _add(self._build_meta_overview_text(payload), category="meta_overview")

        # ── 2.2 Work Values ─────────────────────────────────────
        if payload.work.summary:
            _add(payload.work.summary, category="summary")
        if payload.work.historical_context:
            _add(payload.work.historical_context, category="historical_context")
        if payload.work.realistic_value:
            _add(payload.work.realistic_value, category="reality_value")
        if payload.work.humanistic_value:
            _add(payload.work.humanistic_value, category="human_value")
        if payload.work.artistic_value:
            _add(payload.work.artistic_value, category="art_value")
        if payload.work.famous_quote:
            quote_text = payload.work.famous_quote
            if payload.work.quote_attribution:
                quote_text += f"\n— {payload.work.quote_attribution}"
            _add(quote_text, category="famous_quote")

        # ── 2.3 Sections ────────────────────────────────────────
        for sec in payload.sections:
            sec_slug = self._make_slug(sec.title) or f"section_{sec.number}"
            _add(
                sec.content,
                category="text_section",
                section_id=sec.id,
                section_slug=sec_slug,
                section_title=sec.title,
                section_order=sec.number,
                content_type=sec.content_type,
            )

        # ── 2.4 Commentaries (chỉ published) ────────────────────
        for comm in payload.commentaries:
            if not comm.is_published:
                continue
            comm_content = f"{comm.title}\n\n{comm.content}"
            _add(
                comm_content,
                category="commentary",
                chunk_index_prefix=str(comm.display_order),
            )

        return chunks, texts

    # ── Helpers ──────────────────────────────────────────────────

    @staticmethod
    def _make_slug(s: str) -> str:
        """Convert Vietnamese text to snake_case slug."""
        if not s:
            return ""
        s = remove_vietnamese_accents(s).lower()
        s = re.sub(r'[^a-z0-9]+', '_', s)
        return s.strip('_')

    @staticmethod
    def _split_long_text(text: str, max_tokens: int = MAX_SPLIT_TOKENS) -> List[str]:
        """
        Simple paragraph-based split for texts exceeding max token count.
        Approximation: 1 token ≈ 3 chars for Vietnamese.
        """
        if not text:
            return []

        max_chars = max_tokens * CHARS_PER_TOKEN
        paragraphs = text.split('\n')
        chunks: List[str] = []
        current_chunk: List[str] = []
        current_length = 0

        for p in paragraphs:
            p_len = len(p)
            if p_len == 0:
                continue

            if current_length + p_len > max_chars and current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = [p]
                current_length = p_len
            else:
                current_chunk.append(p)
                current_length += p_len + 1  # +1 for newline

        if current_chunk:
            chunks.append('\n'.join(current_chunk))

        return chunks

    def _build_meta_overview_text(self, payload: WorkSnapshot) -> str:
        """
        Tự sinh text tổng quan từ work + author + tags.
        Giúp tác phẩm có thể search được ngay cả khi chưa có nội dung section.
        """
        title = payload.work.title
        original = f" (tên khác: {payload.work.original_title})" if payload.work.original_title else ""
        author_name = payload.author_ref.name
        
        # Format beautiful genre
        genre_display = self.genre_labels.get(payload.work.genre, payload.work.genre)
        if payload.work.sub_genre:
            sub = self.sub_genre_labels.get(payload.work.sub_genre, payload.work.sub_genre)
            genre_display += f" ({sub})"

        period = self.period_labels.get(payload.work.period, payload.work.period)
        grade = payload.work.grade
        semester = payload.work.semester

        tag_names = ", ".join(t.name for t in payload.tags) if payload.tags else "Không có"

        parts = [
            f"Tác phẩm {title}{original} do tác giả {author_name} sáng tác.",
            f"Tác phẩm thuộc thể loại {genre_display} giai đoạn văn học {period}.",
            f"Tác phẩm được giảng dạy trong chương trình Ngữ Văn lớp {grade}, học kỳ {semester}.",
            f"Các chủ đề liên quan: {tag_names}.",
        ]

        return " ".join(parts)
