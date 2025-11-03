"""
SQLite数据库管理器 - 作为PostgreSQL的备选方案
"""
import sqlite3
import logging
from contextlib import contextmanager
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import uuid

# 配置日志
logger = logging.getLogger(__name__)

class ConversationHistory(BaseModel):
    """对话历史记录模型"""
    id: Optional[int] = None
    session_id: str
    user_input: str
    ai_response: str
    timestamp: datetime
    error_message: Optional[str] = None

class SQLiteDatabaseManager:
    """SQLite数据库管理器"""
    
    def __init__(self, db_path: str = "conversation_history.db"):
        self.db_path = db_path
        logger.info(f"初始化SQLite数据库管理器，数据库路径: {db_path}")
    
    @contextmanager
    def get_connection(self):
        """获取数据库连接的上下文管理器。"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # 使结果可以像字典一样访问
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"数据库连接错误: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def init_database(self):
        """初始化数据库表结构。"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # 创建对话历史表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS conversation_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        user_input TEXT NOT NULL,
                        ai_response TEXT NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        error_message TEXT
                    )
                """)
                
                # 创建索引以提高查询性能
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_session_id 
                    ON conversation_history(session_id)
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_timestamp 
                    ON conversation_history(timestamp)
                """)
                
                conn.commit()
                logger.info("数据库表结构初始化成功")
                
        except Exception as e:
            logger.error(f"数据库初始化失败: {e}")
            raise
    
    def save_conversation(self, session_id: str, user_input: str, 
                         ai_response: str, error_message: Optional[str] = None) -> bool:
        """保存对话记录到数据库。"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO conversation_history 
                    (session_id, user_input, ai_response, error_message)
                    VALUES (?, ?, ?, ?)
                """, (session_id, user_input, ai_response, error_message))
                
                conn.commit()
                logger.info(f"对话记录保存成功，会话ID: {session_id}")
                return True
                
        except Exception as e:
            logger.error(f"保存对话记录失败: {e}")
            return False
    
    def get_conversation_history(self, limit: int = 50, 
                               session_id: Optional[str] = None) -> List[ConversationHistory]:
        """获取对话历史记录。"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                if session_id:
                    cursor.execute("""
                        SELECT * FROM conversation_history 
                        WHERE session_id = ?
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    """, (session_id, limit))
                else:
                    cursor.execute("""
                        SELECT * FROM conversation_history 
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    """, (limit,))
                
                rows = cursor.fetchall()
                
                conversations = []
                for row in rows:
                    conversations.append(ConversationHistory(
                        id=row['id'],
                        session_id=row['session_id'],
                        user_input=row['user_input'],
                        ai_response=row['ai_response'],
                        timestamp=datetime.fromisoformat(row['timestamp']),
                        error_message=row['error_message']
                    ))
                
                logger.info(f"获取到 {len(conversations)} 条对话记录")
                return conversations
                
        except Exception as e:
            logger.error(f"获取对话历史失败: {e}")
            return []
    
    def delete_conversation_history(self, session_id: Optional[str] = None) -> bool:
        """删除对话历史记录。"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                if session_id:
                    cursor.execute("""
                        DELETE FROM conversation_history 
                        WHERE session_id = ?
                    """, (session_id,))
                    logger.info(f"删除会话 {session_id} 的对话记录")
                else:
                    cursor.execute("DELETE FROM conversation_history")
                    logger.info("删除所有对话记录")
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"删除对话历史失败: {e}")
            return False

# 创建全局数据库管理器实例
sqlite_db_manager = SQLiteDatabaseManager()