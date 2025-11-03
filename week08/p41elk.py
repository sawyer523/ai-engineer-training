#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ELK 日志系统整合模块
实现与 Elasticsearch, Logstash, Kibana 的日志传输和处理
"""

import json
import logging
import logging.handlers
import socket
import time
import threading
from datetime import datetime
from typing import Optional


class LogstashTCPHandler(logging.Handler):
    """
    自定义 Logstash TCP 处理器
    支持错误处理和自动重连机制
    """
    
    def __init__(self, host: str = 'localhost', port: int = 5044, 
                 timeout: int = 5, max_retries: int = 3):
        """
        初始化 Logstash TCP 处理器
        
        Args:
            host: Logstash 服务器地址
            port: Logstash 监听端口
            timeout: 连接超时时间（秒）
            max_retries: 最大重试次数
        """
        super().__init__()
        self.host = host
        self.port = port
        self.timeout = timeout
        self.max_retries = max_retries
        self.socket: Optional[socket.socket] = None
        self._lock = threading.Lock()
        
    def _connect(self) -> bool:
        """
        建立到 Logstash 的 TCP 连接
        
        Returns:
            bool: 连接是否成功
        """
        try:
            if self.socket:
                self.socket.close()
            
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.timeout)
            self.socket.connect((self.host, self.port))
            return True
        except Exception as e:
            print(f"连接 Logstash 失败: {e}")
            if self.socket:
                self.socket.close()
                self.socket = None
            return False
    
    def _send_with_retry(self, data: str) -> bool:
        """
        带重试机制的数据发送
        
        Args:
            data: 要发送的日志数据
            
        Returns:
            bool: 发送是否成功
        """
        for attempt in range(self.max_retries):
            try:
                if not self.socket or self.socket.fileno() == -1:
                    if not self._connect():
                        continue
                
                # 发送 JSON 数据，每行以换行符结尾
                message = data + '\n'
                self.socket.sendall(message.encode('utf-8'))
                return True
                
            except Exception as e:
                print(f"发送日志失败 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                if self.socket:
                    self.socket.close()
                    self.socket = None
                
                if attempt < self.max_retries - 1:
                    time.sleep(1)  # 重试前等待1秒
        
        return False
    
    def emit(self, record: logging.LogRecord) -> None:
        """
        发送日志记录到 Logstash
        
        Args:
            record: 日志记录对象
        """
        try:
            with self._lock:
                # 格式化日志记录为 JSON
                log_entry = {
                    'timestamp': datetime.fromtimestamp(record.created).isoformat(),
                    'level': record.levelname,
                    'logger': record.name,
                    'message': record.getMessage(),
                    'module': record.module,
                    'function': record.funcName,
                    'line': record.lineno,
                    'thread': record.thread,
                    'process': record.process
                }
                
                # 添加异常信息（如果存在）
                if record.exc_info:
                    log_entry['exception'] = self.format(record)
                
                json_data = json.dumps(log_entry, ensure_ascii=False)
                
                # 发送到 Logstash
                if not self._send_with_retry(json_data):
                    print(f"无法发送日志到 Logstash: {log_entry['message']}")
                    
        except Exception as e:
            print(f"处理日志记录时出错: {e}")
    
    def close(self) -> None:
        """关闭连接"""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            finally:
                self.socket = None
        super().close()


def setup_logger() -> logging.Logger:
    """
    配置日志记录器
    同时输出到控制台、文件和 Logstash
    
    Returns:
        logging.Logger: 配置好的日志记录器
    """
    # 创建日志记录器
    logger = logging.getLogger('elk_integration')
    logger.setLevel(logging.DEBUG)
    
    # 清除现有处理器
    logger.handlers.clear()
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 1. 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 2. 文件处理器
    file_handler = logging.FileHandler('elk_integration.log', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 3. Logstash TCP 处理器
    try:
        logstash_handler = LogstashTCPHandler(
            host='localhost',
            port=5044,
            timeout=5,
            max_retries=3
        )
        logstash_handler.setLevel(logging.INFO)
        logger.addHandler(logstash_handler)
        logger.info("Logstash 处理器配置成功")
    except Exception as e:
        logger.error(f"配置 Logstash 处理器失败: {e}")
    
    return logger


def generate_test_logs(logger: logging.Logger, duration_minutes: int = 10) -> None:
    """
    生成测试日志数据
    
    Args:
        logger: 日志记录器
        duration_minutes: 运行时长（分钟）
    """
    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)
    log_count = 0
    
    logger.info(f"开始生成测试日志，持续时间: {duration_minutes} 分钟")
    
    # 测试日志消息模板
    test_messages = [
        "用户登录成功",
        "数据库查询执行",
        "API 请求处理完成",
        "缓存更新操作",
        "文件上传成功",
        "邮件发送完成",
        "定时任务执行",
        "系统健康检查",
        "配置更新通知",
        "性能监控数据"
    ]
    
    log_levels = [
        (logging.INFO, "INFO"),
        (logging.WARNING, "WARNING"),
        (logging.ERROR, "ERROR"),
        (logging.DEBUG, "DEBUG")
    ]
    
    try:
        while time.time() < end_time:
            # 选择随机的日志级别和消息
            level, level_name = log_levels[log_count % len(log_levels)]
            message = test_messages[log_count % len(test_messages)]
            
            # 添加计数器和时间戳信息
            full_message = f"{message} - 序号: {log_count + 1}, 时间: {datetime.now().strftime('%H:%M:%S')}"
            
            # 记录日志
            logger.log(level, full_message)
            
            log_count += 1
            
            # 每秒生成一条日志
            time.sleep(1)
            
            # 每100条日志输出一次统计信息
            if log_count % 100 == 0:
                elapsed_time = time.time() - start_time
                logger.info(f"已生成 {log_count} 条日志，运行时间: {elapsed_time:.1f} 秒")
    
    except KeyboardInterrupt:
        logger.info("日志生成被用户中断")
    except Exception as e:
        logger.error(f"日志生成过程中出错: {e}")
    finally:
        total_time = time.time() - start_time
        logger.info(f"日志生成完成，总计: {log_count} 条，耗时: {total_time:.1f} 秒")


def test_elk_connection(logger: logging.Logger) -> None:
    """
    测试 ELK 系统连接
    
    Args:
        logger: 日志记录器
    """
    logger.info("=== ELK 系统连接测试开始 ===")
    
    # 测试不同级别的日志
    logger.debug("这是一条调试日志 - DEBUG level")
    logger.info("这是一条信息日志 - INFO level")
    logger.warning("这是一条警告日志 - WARNING level")
    logger.error("这是一条错误日志 - ERROR level")
    logger.critical("这是一条严重错误日志 - CRITICAL level")
    
    # 测试包含特殊字符的日志
    logger.info("测试中文字符和特殊符号: 你好世界! @#$%^&*()")
    
    # 测试异常日志
    try:
        raise ValueError("这是一个测试异常")
    except Exception as e:
        logger.exception("捕获到异常信息")
    
    logger.info("=== ELK 系统连接测试完成 ===")


def main():
    """主函数"""
    print("ELK 日志系统整合程序启动")
    print("=" * 50)
    
    # 设置日志记录器
    logger = setup_logger()
    
    # 测试连接
    test_elk_connection(logger)
    
    # 等待一下让测试日志发送完成
    time.sleep(2)
    
    # 询问用户是否开始生成测试日志
    try:
        duration = input("请输入日志生成持续时间（分钟，默认10分钟）: ").strip()
        if not duration:
            duration = 10
        else:
            duration = int(duration)
        
        print(f"开始生成测试日志，持续 {duration} 分钟...")
        print("按 Ctrl+C 可以提前停止")
        
        # 生成测试日志
        generate_test_logs(logger, duration)
        
    except KeyboardInterrupt:
        logger.info("程序被用户中断")
    except ValueError:
        logger.error("输入的时间格式不正确，使用默认值10分钟")
        generate_test_logs(logger, 10)
    except Exception as e:
        logger.error(f"程序执行出错: {e}")
    finally:
        # 清理资源
        for handler in logger.handlers:
            if hasattr(handler, 'close'):
                handler.close()
        
        print("\n程序执行完成")
        print("请在 Kibana (http://localhost:5601) 中查看日志数据")


if __name__ == "__main__":
    main()