"""
Sync Router — API nội bộ cho BE (Java) đồng bộ dữ liệu tác phẩm sang AI Service.

Endpoints:
    PUT    /internal/works/{work_slug}/sync     — Upsert work
    DELETE /internal/works/{work_slug}/sync     — Soft delete work
    PUT    /internal/authors/{author_slug}/sync — Upsert author
    DELETE /internal/authors/{author_slug}/sync — Soft delete author
"""

import logging

from fastapi import APIRouter, HTTPException, Path, Request

from schemas.sync_schema import (
    WorkSnapshot,
    AuthorSnapshot,
    SyncResponse,
    EXPECTED_SNAPSHOT_SCHEMA,
    EXPECTED_AUTHOR_SCHEMA,
)

logger = logging.getLogger("rag-service.api.sync-router")
router = APIRouter(prefix="/internal", tags=["Internal Sync"])


@router.put("/works/{work_slug}/sync", response_model=SyncResponse)
async def sync_work(
    request: Request,
    payload: WorkSnapshot,
    work_slug: str = Path(..., description="Slug của tác phẩm cần đồng bộ"),
):
    # Validate schema version
    if payload.schema_version != EXPECTED_SNAPSHOT_SCHEMA:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid schema_version '{payload.schema_version}', expected '{EXPECTED_SNAPSHOT_SCHEMA}'",
        )

    # Validate slug consistency
    if work_slug != payload.work.slug:
        raise HTTPException(
            status_code=400,
            detail=f"Path workSlug '{work_slug}' does not match payload.work.slug '{payload.work.slug}'",
        )

    try:
        svc = request.app.state.sync_svc
        result = svc.sync_work(payload)
        return SyncResponse(**result)
    except Exception as e:
        logger.error(f"Error syncing work '{work_slug}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/works/{work_slug}/sync", response_model=SyncResponse)
async def delete_work(
    request: Request,
    work_slug: str = Path(..., description="Slug của tác phẩm cần xóa"),
):
    try:
        svc = request.app.state.sync_svc
        result = svc.delete_work(work_slug)
        return SyncResponse(**result)
    except Exception as e:
        logger.error(f"Error deleting work '{work_slug}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/authors/{author_slug}/sync", response_model=SyncResponse)
async def sync_author(
    request: Request,
    payload: AuthorSnapshot,
    author_slug: str = Path(..., description="Slug của tác giả cần đồng bộ"),
):
    if payload.schema_version != EXPECTED_AUTHOR_SCHEMA:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid schema_version '{payload.schema_version}', expected '{EXPECTED_AUTHOR_SCHEMA}'",
        )

    if author_slug != payload.author.slug:
        raise HTTPException(
            status_code=400,
            detail=f"Path authorSlug '{author_slug}' does not match payload.author.slug '{payload.author.slug}'",
        )

    try:
        svc = request.app.state.sync_svc
        result = svc.sync_author(payload)
        return SyncResponse(**result)
    except Exception as e:
        logger.error(f"Error syncing author '{author_slug}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/authors/{author_slug}/sync", response_model=SyncResponse)
async def delete_author(
    request: Request,
    author_slug: str = Path(..., description="Slug của tác giả cần xóa"),
):
    try:
        svc = request.app.state.sync_svc
        result = svc.delete_author(author_slug)
        return SyncResponse(**result)
    except Exception as e:
        logger.error(f"Error deleting author '{author_slug}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
