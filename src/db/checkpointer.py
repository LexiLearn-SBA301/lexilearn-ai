"""
Redis checkpointer cho LangGraph — persist/resume state theo thread_id.

Đặt cùng tầng với mongo_client.py (db/ = nơi giữ kết nối datastore). Redis cũng
là 1 datastore. graph/workflow.py chỉ nhận checkpointer trừu tượng, KHÔNG biết Redis.

YÊU CẦU: Redis có module RediSearch + RedisJSON (Redis Stack / Redis 8+),
vì RedisSaver.setup() tạo search index.
"""
import logging
import os

logger = logging.getLogger("rag-service.db.checkpointer")


async def get_checkpointer():
    """Tạo AsyncRedisSaver từ REDIS_URL (mặc định redis://localhost:6379), đã setup index.

    Import lazy: chỉ cần langgraph-checkpoint-redis khi thật sự bật persist.
    Dùng bản ASYNC vì graph chạy qua app.ainvoke() -> LangGraph gọi checkpointer.aget_tuple(),
    RedisSaver (sync) không override method async này nên rơi vào NotImplementedError.
    """
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    from langgraph.checkpoint.redis import AsyncRedisSaver

    try:
        checkpointer = AsyncRedisSaver(redis_url)
        await checkpointer.asetup()  # tạo index lần đầu (idempotent các lần sau)
        logger.info("AsyncRedisSaver checkpointer sẵn sàng (%s).", redis_url)
        return checkpointer
    except Exception as e:
        logger.warning(f"Lỗi kết nối Redis ({e}). Tự động fallback sang MemorySaver (chỉ dùng cho DEV)!")
        from langgraph.checkpoint.memory import MemorySaver
        return MemorySaver()


async def close_checkpointer(checkpointer) -> None:
    """Đóng Redis client mà checkpointer TỰ mở, gọi lúc shutdown để tránh rò connection.

    get_checkpointer() tạo AsyncRedisSaver tự sở hữu client (_owns_its_client=True); lifespan
    KHÔNG tự đóng. Teardown mirror đúng from_conn_string của lib: chỉ đóng client do chính
    checkpointer mở (không đụng client inject từ ngoài), close rồi disconnect pool.
    """
    if checkpointer is None:
        return
    if not getattr(checkpointer, "_owns_its_client", False):
        return
    try:
        redis = checkpointer._redis
        await redis.aclose()
        pool = getattr(redis, "connection_pool", None)
        if pool is not None:
            await pool.disconnect()
        logger.info("AsyncRedisSaver checkpointer đã đóng kết nối.")
    except Exception as e:  # shutdown không được vỡ chỉ vì lỗi đóng kết nối
        logger.warning("Lỗi khi đóng checkpointer: %s", e)