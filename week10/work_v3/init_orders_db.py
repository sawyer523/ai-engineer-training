"""初始化订单 SQLite 数据库脚本

创建 `orders` 表并插入一条示例记录，同时确保存在 `start_time` 列。
"""
import os
import sqlite3

base = os.path.dirname(__file__)
db_dir = os.path.join(base, "db")
os.makedirs(db_dir, exist_ok=True)
db_path = os.path.join(db_dir, "orders.sqlite")

conn = sqlite3.connect(db_path)
c = conn.cursor()
c.execute("CREATE TABLE IF NOT EXISTS orders (order_id TEXT PRIMARY KEY, status TEXT, amount REAL, updated_at TEXT)")

# 若不存在 `start_time` 列，则添加（兼容旧库）
cols = {row[1] for row in c.execute("PRAGMA table_info(orders)").fetchall()}
if "start_time" not in cols:
    c.execute("ALTER TABLE orders ADD COLUMN start_time TEXT")

# 插入/更新一条示例订单，便于联调
c.execute(
    "INSERT OR REPLACE INTO orders(order_id,status,amount,updated_at,start_time) VALUES(?,?,?,?,?)",
    ("20251114001", "processing", 199.0, "2025-11-15 12:00:00", "2025-11-16 09:00:00"),
)
conn.commit()
c.close()
conn.close()

print("created:", db_path)
