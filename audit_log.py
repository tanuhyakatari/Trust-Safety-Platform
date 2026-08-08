import sqlite3
from datetime import datetime

DB_PATH = "audit_trail.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            agent TEXT,
            input_summary TEXT,
            decision TEXT,
            reason TEXT,
            latency_ms REAL
        )
    """)
    conn.commit()
    conn.close()

def log_decision(agent: str, input_summary: str, decision: str, reason: str, latency_ms: float):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO audit_log (timestamp, agent, input_summary, decision, reason, latency_ms) VALUES (?, ?, ?, ?, ?, ?)",
        (datetime.now().isoformat(), agent, input_summary, decision, reason, latency_ms)
    )
    conn.commit()
    conn.close()

def get_all_logs():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute("SELECT * FROM audit_log ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows

def get_fairness_stats():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute("""
        SELECT
            CASE WHEN input_summary LIKE '%seller_type:small%' THEN 'small' ELSE 'established' END as seller_type,
            decision,
            COUNT(*) as count
        FROM audit_log
        WHERE agent != 'risk_scoring'
        GROUP BY seller_type, decision
    """)
    rows = cursor.fetchall()
    conn.close()
    return rows

init_db()
