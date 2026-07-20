"""
Sync Router — API nội bộ cho BE (Java) đồng bộ dữ liệu tác phẩm sang AI Service.

Endpoints:
    PUT    /internal/works/{work_slug}/sync   — Upsert (idempotent create/update)
    DELETE /internal/works/{work_slug}/sync   — Soft delete (is_active = false)
"""

import logging

from fastapi import APIRouter, HTTPException, Path, Request

from schemas.sync_schema import WorkSnapshot, SyncResponse, EXPECTED_SNAPSHOT_SCHEMA

logger = logging.getLogger("rag-service.api.sync-router")
router = APIRouter(prefix="/internal/works", tags=["Internal Sync"])


@router.put("/{work_slug}/sync", response_model=SyncResponse)
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


@router.delete("/{work_slug}/sync", response_model=SyncResponse)
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
