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


def get_checkpointer():
    """Tạo RedisSaver từ REDIS_URL (mặc định redis://localhost:6379), đã setup index.

    Import lazy: chỉ cần langgraph-checkpoint-redis khi thật sự bật persist.
    """
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    from langgraph.checkpoint.redis import RedisSaver

    checkpointer = RedisSaver(redis_url)
    checkpointer.setup()  # tạo index lần đầu (idempotent các lần sau)
    logger.info("RedisSaver checkpointer sẵn sàng (%s).", redis_url)
    return checkpointer


def close_checkpointer(checkpointer) -> None:
    """Đóng Redis client mà checkpointer TỰ mở, gọi lúc shutdown để tránh rò connection.

    get_checkpointer() tạo RedisSaver tự sở hữu client (_owns_its_client=True); lifespan
    KHÔNG tự đóng. Teardown mirror đúng from_conn_string của lib: chỉ đóng client do chính
    checkpointer mở (không đụng client inject từ ngoài), close rồi disconnect pool.
    """
    if checkpointer is None:
        return
    if not getattr(checkpointer, "_owns_its_client", False):
        return
    try:
        redis = checkpointer._redis
        redis.close()
        pool = getattr(redis, "connection_pool", None)
        if pool is not None:
            pool.disconnect()
        logger.info("RedisSaver checkpointer đã đóng kết nối.")
    except Exception as e:  # shutdown không được vỡ chỉ vì lỗi đóng kết nối
        logger.warning("Lỗi khi đóng checkpointer: %s", e)