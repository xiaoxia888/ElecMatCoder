"""批量编码任务的 SQLite 持久化存储。

替代原先把 results / traces 全量缓存在进程内存的做法：
- 任务元数据与每条编码结果落盘到 data/batch/batch_jobs.db
- 内存只保留运行调度所需的轻量状态（状态、计数、订阅者队列等）
- 自带保留策略（按时间 + 数量上限）控制磁盘体积

SQLite 为 Python 标准库内置，无需安装数据库服务，数据库即单个文件。
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_ACTIVE_STATUSES = {"queued", "running", "cancelling"}


class BatchJobStore:
    def __init__(self, db_path: Path, *, keep_days: float = 7.0, max_jobs: int = 200) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._keep_days = float(keep_days)
        self._max_jobs = int(max_jobs)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT,
                    total INTEGER,
                    processed INTEGER,
                    success_count INTEGER,
                    review_count INTEGER,
                    threshold REAL,
                    max_concurrent INTEGER,
                    error TEXT,
                    items_meta TEXT,
                    created_at REAL,
                    started_at REAL,
                    finished_at REAL,
                    duration_seconds REAL,
                    updated_at REAL
                );
                CREATE TABLE IF NOT EXISTS results (
                    job_id TEXT,
                    order_index INTEGER,
                    client_index INTEGER,
                    result TEXT,
                    PRIMARY KEY (job_id, order_index)
                );
                CREATE INDEX IF NOT EXISTS idx_results_job ON results(job_id);
                CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at);
                """
            )
            self._ensure_column("jobs", "duration_seconds", "REAL")
            self._conn.commit()

    def _ensure_column(self, table: str, column: str, definition: str) -> None:
        columns = {row["name"] for row in self._conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if column not in columns:
            self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    # ---------- 写 ----------
    def create_job(self, job: Dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO jobs
                (job_id,status,total,processed,success_count,review_count,threshold,
                 max_concurrent,error,items_meta,created_at,started_at,finished_at,duration_seconds,updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    job["job_id"],
                    str(job.get("status", "") or ""),
                    int(job.get("total", 0) or 0),
                    int(job.get("processed", 0) or 0),
                    int(job.get("success_count", 0) or 0),
                    int(job.get("review_count", 0) or 0),
                    job.get("threshold"),
                    int(job.get("max_concurrent", 1) or 1),
                    str(job.get("error", "") or ""),
                    json.dumps(job.get("items_meta", []), ensure_ascii=False),
                    job.get("created_at"),
                    job.get("started_at"),
                    job.get("finished_at"),
                    job.get("duration_seconds"),
                    job.get("updated_at"),
                ),
            )
            self._conn.commit()

    def update_job(self, job_id: str, **fields: Any) -> None:
        if not fields:
            return
        columns = ", ".join(f"{key}=?" for key in fields)
        with self._lock:
            self._conn.execute(
                f"UPDATE jobs SET {columns} WHERE job_id=?",
                (*fields.values(), job_id),
            )
            self._conn.commit()

    def save_result(self, job_id: str, order_index: int, client_index: int, result: Dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO results (job_id,order_index,client_index,result) VALUES (?,?,?,?)",
                (job_id, int(order_index), int(client_index), json.dumps(result, ensure_ascii=False)),
            )
            self._conn.commit()

    def mark_interrupted_jobs(self) -> None:
        """服务启动时把上次未跑完（非终态）的任务标记为失败，避免出现僵尸"运行中"。"""
        now = time.time()
        with self._lock:
            rows = self._conn.execute(
                "SELECT job_id, started_at FROM jobs WHERE status IN ('queued','running','cancelling')"
            ).fetchall()
            for row in rows:
                started_at = row["started_at"]
                duration_seconds = max(0.0, now - float(started_at)) if started_at else None
                self._conn.execute(
                    """UPDATE jobs
                       SET status='failed', error='服务重启中断', finished_at=?, duration_seconds=?, updated_at=?
                       WHERE job_id=?""",
                    (now, duration_seconds, now, row["job_id"]),
                )
            self._conn.commit()

    # ---------- 读 ----------
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return self._row_to_job(row) if row else None

    def get_result(self, job_id: str, order_index: int) -> Optional[Dict[str, Any]]:
        """按 (job_id, order_index) 主键查单条结果，供前端点击描述时按需查询/调试。"""
        with self._lock:
            row = self._conn.execute(
                "SELECT result FROM results WHERE job_id=? AND order_index=?",
                (job_id, int(order_index)),
            ).fetchone()
        if not row:
            return None
        try:
            return json.loads(row["result"])
        except (TypeError, ValueError):
            return None

    def get_results(self, job_id: str) -> Dict[str, Any]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT order_index, result FROM results WHERE job_id=? ORDER BY order_index",
                (job_id,),
            ).fetchall()
        out: Dict[str, Any] = {}
        for row in rows:
            try:
                out[str(row["order_index"])] = json.loads(row["result"])
            except (TypeError, ValueError):
                continue
        return out

    def list_recent_jobs(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [self._row_to_job(row) for row in rows]

    def _row_to_job(self, row: sqlite3.Row) -> Dict[str, Any]:
        try:
            items_meta = json.loads(row["items_meta"] or "[]")
        except (TypeError, ValueError):
            items_meta = []
        return {
            "job_id": row["job_id"],
            "status": row["status"],
            "total": row["total"],
            "processed": row["processed"],
            "success_count": row["success_count"],
            "review_count": row["review_count"],
            "threshold": row["threshold"],
            "max_concurrent": row["max_concurrent"],
            "error": row["error"],
            "items_meta": items_meta,
            "created_at": row["created_at"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "duration_seconds": row["duration_seconds"],
            "updated_at": row["updated_at"],
        }

    # ---------- 清理 ----------
    def cleanup(self) -> None:
        """按保留策略删除过期 / 超量任务（活跃任务永不删除）。"""
        cutoff = time.time() - self._keep_days * 86400
        with self._lock:
            stale = self._conn.execute(
                "SELECT job_id FROM jobs WHERE created_at < ? AND status NOT IN ('queued','running','cancelling')",
                (cutoff,),
            ).fetchall()
            overflow = self._conn.execute(
                """SELECT job_id FROM jobs
                   WHERE status NOT IN ('queued','running','cancelling')
                   ORDER BY created_at DESC LIMIT -1 OFFSET ?""",
                (self._max_jobs,),
            ).fetchall()
            doomed = {row["job_id"] for row in stale} | {row["job_id"] for row in overflow}
            for job_id in doomed:
                self._conn.execute("DELETE FROM results WHERE job_id=?", (job_id,))
                self._conn.execute("DELETE FROM jobs WHERE job_id=?", (job_id,))
            if doomed:
                self._conn.commit()
