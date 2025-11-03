"""
数据库模型和连接管理模块。
"""
import logging
from datetime import datetime
from typing import List, Optional
from contextlib import contextmanager

import psycopg2
from psycopg2.extras import RealDictCursor
from pydantic import BaseModel

from config import config

logger = logging.getLogger(__name__)


class ConversationHistory(BaseModel):
    """对话历史数据模型。"""
    id: Optional[int] = None
    user_input: str
    ai_response: str
    timestamp: Optional[datetime] = None
    session_id: Optional[str] = None


class DatabaseManager:
    """数据库管理器类。"""
    
    def __init__(self):
        self.connection_params = {
            'host': config.DB_HOST,
            'port': config.DB_PORT,
            'database': config.DB_NAME,
            'user': config.DB_USER,
            'password': config.DB_PASSWORD
        }
    
    @contextmanager
    def get_connection(self):
        """获取数据库连接的上下文管理器。"""
        conn = None
        try:
            # 使用与testdb.py相同的连接方式
            conn = psycopg2.connect(
                host=self.connection_params['host'],
                port=self.connection_params['port'],
                database=self.connection_params['database'],
                user=self.connection_params['user'],
                password=self.connection_params['password']
            )
            yield conn
        except psycopg2.Error as e:
            if conn:
                conn.rollback()
            logger.error(f"数据库连接错误: {e}")
            raise ConnectionError(f"无法连接到数据库: {e}")
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
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS conversation_history (
            id SERIAL PRIMARY KEY,
            user_input TEXT NOT NULL,
            ai_response TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            session_id VARCHAR(255)
        );
        
        CREATE INDEX IF NOT EXISTS idx_conversation_timestamp 
        ON conversation_history(timestamp DESC);
        
        CREATE INDEX IF NOT EXISTS idx_conversation_session 
        ON conversation_history(session_id);
        """
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(create_table_sql)
                    conn.commit()
                    logger.info("数据库表初始化成功")
        except Exception as e:
            logger.error(f"数据库初始化失败: {e}")
            raise
    
    def save_conversation(self, user_input: str, ai_response: str, session_id: Optional[str] = None) -> int:
        """
        保存对话到数据库。
        
        Args:
            user_input: 用户输入
            ai_response: AI响应
            session_id: 会话ID（可选）
            
        Returns:
            插入记录的ID
        """
        insert_sql = """
        INSERT INTO conversation_history (user_input, ai_response, session_id)
        VALUES (%s, %s, %s)
        RETURNING id;
        """
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(insert_sql, (user_input, ai_response, session_id))
                    record_id = cursor.fetchone()[0]
                    conn.commit()
                    logger.info(f"对话保存成功，ID: {record_id}")
                    return record_id
        except Exception as e:
            logger.error(f"保存对话失败: {e}")
            raise
    
    def get_conversation_history(self, limit: int = 50, session_id: Optional[str] = None) -> List[ConversationHistory]:
        """
        获取对话历史。
        
        Args:
            limit: 返回记录数限制
            session_id: 会话ID过滤（可选）
            
        Returns:
            对话历史列表
        """
        if session_id:
            select_sql = """
            SELECT id, user_input, ai_response, timestamp, session_id
            FROM conversation_history
            WHERE session_id = %s
            ORDER BY timestamp DESC
            LIMIT %s;
            """
            params = (session_id, limit)
        else:
            select_sql = """
            SELECT id, user_input, ai_response, timestamp, session_id
            FROM conversation_history
            ORDER BY timestamp DESC
            LIMIT %s;
            """
            params = (limit,)
        
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(select_sql, params)
                    records = cursor.fetchall()
                    
                    conversations = []
                    for record in records:
                        conversations.append(ConversationHistory(
                            id=record['id'],
                            user_input=record['user_input'],
                            ai_response=record['ai_response'],
                            timestamp=record['timestamp'],
                            session_id=record['session_id']
                        ))
                    
                    logger.info(f"获取到 {len(conversations)} 条对话历史")
                    return conversations
        except Exception as e:
            logger.error(f"获取对话历史失败: {e}")
            raise
    
    def delete_conversation_history(self, days_old: int = 30) -> int:
        """
        删除指定天数之前的对话历史。
        
        Args:
            days_old: 删除多少天前的记录
            
        Returns:
            删除的记录数
        """
        delete_sql = """
        DELETE FROM conversation_history
        WHERE timestamp < CURRENT_TIMESTAMP - INTERVAL '%s days';
        """
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(delete_sql, (days_old,))
                    deleted_count = cursor.rowcount
                    conn.commit()
                    logger.info(f"删除了 {deleted_count} 条旧对话记录")
                    return deleted_count
        except Exception as e:
            logger.error(f"删除对话历史失败: {e}")
            raise


# 创建全局数据库管理器实例
db_manager = DatabaseManager()