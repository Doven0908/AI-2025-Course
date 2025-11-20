from __future__ import annotations
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .logger import get_logger

logger = get_logger(__name__)


class MetricsCollector:
    """记录查询性能指标到 SQLite 数据库（线程安全版本）。"""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()  # ✅ 使用线程局部存储
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        """初始化数据库表结构（只在主线程执行一次）。"""
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                latency_ms REAL NOT NULL,
                retrieval_ms REAL NOT NULL,
                generation_ms REAL NOT NULL,
                retrieved_k INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                status TEXT DEFAULT 'success'
            )
        """)
        conn.commit()
        conn.close()

    @contextmanager
    def _get_connection(self):
        """获取当前线程的数据库连接（线程安全）。"""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False  # ✅ 允许跨线程
            )
        try:
            yield self._local.conn
        finally:
            pass  # 保持连接打开，由线程退出时清理

    def record(
            self,
            *,
            query: str,
            latency_ms: float,
            retrieval_ms: float,
            generation_ms: float,
            retrieved_k: int,
            status: str = "success",
    ) -> None:
        """记录一次查询的性能指标（线程安全）。"""
        timestamp = datetime.utcnow().isoformat()

        with self._lock:  # ✅ 使用锁保护数据库写入
            try:
                with self._get_connection() as conn:
                    conn.execute(
                        """
                        INSERT INTO metrics 
                        (query, latency_ms, retrieval_ms, generation_ms, retrieved_k, timestamp, status)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (query, latency_ms, retrieval_ms, generation_ms, retrieved_k, timestamp, status),
                    )
                    conn.commit()
            except Exception as exc:
                logger.error("Failed to record metrics: %s", exc)

    def recent(self, limit: int = 50) -> List[Dict]:
        """返回最近的查询记录。"""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT query, latency_ms, retrieval_ms, generation_ms, 
                       retrieved_k, timestamp, status
                FROM metrics
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cursor.fetchall()

        return [
            {
                "query": row[0],
                "latency_ms": row[1],
                "retrieval_ms": row[2],
                "generation_ms": row[3],
                "retrieved_k": row[4],
                "timestamp": row[5],
                "status": row[6],
            }
            for row in rows
        ]

    def aggregates(self) -> Dict:
        """返回聚合统计信息。"""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_queries,
                    AVG(latency_ms) as avg_latency_ms,
                    AVG(retrieval_ms) as avg_retrieval_ms,
                    AVG(generation_ms) as avg_generation_ms
                FROM metrics
            """)
            row = cursor.fetchone()

        return {
            "total_queries": row[0] or 0,
            "avg_latency_ms": row[1] or 0.0,
            "avg_retrieval_ms": row[2] or 0.0,
            "avg_generation_ms": row[3] or 0.0,
        }