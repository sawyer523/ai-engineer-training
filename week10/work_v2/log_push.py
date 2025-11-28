#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import json
import logging
import threading
import time
import socket
import queue
import argparse
import re
import base64
from datetime import datetime
from urllib import request as urlrequest, error as urlerror

def abs_path(p):
    if not p:
        return p
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(os.getcwd(), p))

def default_config():
    return {
        "input_file": "work/logs/requests.log",
        "state_file": "work/state/log_push_state.json",
        "logstash_http_url": "http://localhost:8080",
        "auth": {"type": "none", "username": "", "password": "", "bearer_token": ""},
        "batch_size": 200,
        "batch_flush_ms": 1500,
        "max_retries": 5,
        "retry_backoff_ms": 500,
        "threads": 2,
        "content_type": "ndjson",
        "field_map": {
            "timestamp": "@timestamp",
            "level": "level",
            "logger": "logger",
            "route": "route",
            "cost_ms": "latency_ms",
            "request_query": "request.query"
        },
        "metadata": {"source": "requests.log"},
        "interval_sec": 2,
        "from_start": False
    }

def load_config(path):
    cfg = default_config()
    p = abs_path(path) if path else None
    if p and os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            loaded = json.load(f)
            def merge(a, b):
                for k, v in b.items():
                    if isinstance(v, dict) and isinstance(a.get(k), dict):
                        merge(a[k], v)
                    else:
                        a[k] = v
            merge(cfg, loaded)
    return cfg

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

class StateManager:
    def __init__(self, path):
        self.path = abs_path(path) if path else None
        self.lock = threading.Lock()
        if self.path:
            ensure_dir(os.path.dirname(self.path))
    def load(self):
        if not self.path or not os.path.exists(self.path):
            return {}
        with self.lock:
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except:
                return {}
    def save(self, state):
        if not self.path:
            return
        with self.lock:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False)

class LogReader:
    def __init__(self, file_path, state_mgr, logger):
        self.file_path = abs_path(file_path)
        self.state_mgr = state_mgr
        self.logger = logger
        self.fd = None
        self.pos = 0
        self.file_ctime = None
        self.state = self.state_mgr.load()
        self.key = os.path.abspath(self.file_path)
        self._load_state()
    def _load_state(self):
        s = self.state.get(self.key)
        try:
            st = os.stat(self.file_path)
            self.file_ctime = st.st_ctime
            if s and s.get("ctime") == st.st_ctime and s.get("pos", 0) <= st.st_size:
                self.pos = int(s.get("pos", 0))
            else:
                self.pos = 0
        except FileNotFoundError:
            self.pos = 0
    def _persist(self, pos):
        self.state[self.key] = {"pos": pos, "ctime": self.file_ctime}
        self.state_mgr.save(self.state)
    def open(self):
        self.fd = open(self.file_path, "r", encoding="utf-8", errors="ignore")
        if self.pos:
            self.fd.seek(self.pos)
    def close(self):
        try:
            if self.fd:
                self.fd.close()
        except:
            pass
        self.fd = None
    def _rotated(self):
        try:
            st = os.stat(self.file_path)
            if st.st_ctime != self.file_ctime or self.pos > st.st_size:
                self.file_ctime = st.st_ctime
                return True
            return False
        except FileNotFoundError:
            return False
    def read_lines(self, stop_event, out_queue):
        self.open()
        pending_pos = self.pos
        try:
            while not stop_event.is_set():
                line = self.fd.readline()
                if not line:
                    if self._rotated():
                        self.close()
                        self.pos = 0
                        pending_pos = 0
                        self.open()
                    else:
                        time.sleep(0.2)
                    continue
                pending_pos += len(line.encode("utf-8"))
                out_queue.put((pending_pos, line.rstrip("\n")))
        finally:
            self.close()
    def snapshot(self, count=10000):
        self.open()
        res = []
        pending_pos = self.pos
        try:
            for _ in range(count):
                line = self.fd.readline()
                if not line:
                    break
                pending_pos += len(line.encode("utf-8"))
                res.append((pending_pos, line.rstrip("\n")))
        finally:
            self.close()
        return res
    def commit(self, pos):
        self.pos = pos
        self._persist(pos)

class EventParser:
    def __init__(self, field_map, metadata):
        self.field_map = field_map or {}
        self.metadata = metadata or {}
        self.hostname = socket.gethostname()
        self.ts_re = re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})\s*\|\s*(?P<level>\w+)\s*\|\s*(?P<logger>[\w\-]+)\s*\|\s*(?P<msg>.*)$")
    def _map(self, k, v, out):
        mk = self.field_map.get(k, k)
        out[mk] = v
    def parse(self, line):
        m = self.ts_re.match(line)
        base = {}
        if m:
            ts = m.group("ts")
            try:
                dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S,%f")
                self._map("timestamp", dt.isoformat(), base)
            except:
                self._map("timestamp", ts, base)
            self._map("level", m.group("level"), base)
            self._map("logger", m.group("logger"), base)
            msg = m.group("msg")
            req_match = re.search(r"request=(\{.*\})", msg)
            if req_match:
                try:
                    rq = json.loads(req_match.group(1))
                    if isinstance(rq, dict):
                        if "query" in rq:
                            self._map("request_query", rq.get("query"), base)
                        base["request"] = rq
                except:
                    pass
            for k, v in re.findall(r"(\w+)=([^\s\|]+)", msg):
                if k == "cost" and v.endswith("ms"):
                    try:
                        ms = float(v[:-2])
                        self._map("cost_ms", ms, base)
                    except:
                        self._map("cost_ms", v, base)
                else:
                    base[k] = v
            base["message"] = msg
        else:
            base["message"] = line
        base["host"] = self.hostname
        for k, v in (self.metadata or {}).items():
            base[k] = v
        if "@timestamp" not in base and "timestamp" in base:
            base["@timestamp"] = base["timestamp"]
        return base

class ELKHttpSender:
    def __init__(self, url, auth, content_type, logger, max_retries, backoff_ms):
        self.url = url
        self.auth = auth or {"type": "none"}
        self.content_type = content_type or "ndjson"
        self.logger = logger
        self.max_retries = max_retries
        self.backoff_ms = backoff_ms
    def _headers(self):
        headers = {"Content-Type": "application/x-ndjson" if self.content_type == "ndjson" else "application/json"}
        t = (self.auth or {}).get("type", "none")
        if t == "basic":
            u = self.auth.get("username", "")
            p = self.auth.get("password", "")
            token = base64.b64encode(f"{u}:{p}".encode("utf-8")).decode("ascii")
            headers["Authorization"] = f"Basic {token}"
        elif t == "bearer":
            headers["Authorization"] = f"Bearer {self.auth.get('bearer_token','')}"
        return headers
    def send_batch(self, events):
        if not events:
            return True
        data = None
        if self.content_type == "ndjson":
            lines = [json.dumps(e, ensure_ascii=False) for e in events]
            data = ("\n".join(lines) + "\n").encode("utf-8")
        else:
            data = json.dumps(events, ensure_ascii=False).encode("utf-8")
        req = urlrequest.Request(self.url, data=data, headers=self._headers(), method="POST")
        for attempt in range(self.max_retries):
            try:
                with urlrequest.urlopen(req, timeout=10) as resp:
                    code = resp.getcode()
                    if 200 <= code < 300:
                        return True
                    self.logger.error(f"HTTP {code}")
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep((self.backoff_ms / 1000.0) * (attempt + 1))
                else:
                    self.logger.error(f"send failed: {e}")
        return False
    def health_check(self):
        payload = [{"@timestamp": datetime.utcnow().isoformat(), "type": "healthcheck", "host": socket.gethostname()}]
        return self.send_batch(payload)

class LogPushService:
    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger
        self.state_mgr = StateManager(cfg.get("state_file"))
        self.reader = LogReader(cfg.get("input_file"), self.state_mgr, logger)
        self.parser = EventParser(cfg.get("field_map"), cfg.get("metadata"))
        self.sender = ELKHttpSender(cfg.get("logstash_http_url"), cfg.get("auth"), cfg.get("content_type"), logger, cfg.get("max_retries"), cfg.get("retry_backoff_ms"))
        self.batch_size = int(cfg.get("batch_size", 200))
        self.flush_ms = int(cfg.get("batch_flush_ms", 1500))
        self.queue = queue.Queue(maxsize=10000)
        self.stop_event = threading.Event()
        self.last_committed_pos = self.reader.pos
    def _aggregate_and_send(self):
        batch = []
        last_pos = self.last_committed_pos
        last_time = time.time()
        while not self.stop_event.is_set():
            try:
                pos, line = self.queue.get(timeout=0.2)
                event = self.parser.parse(line)
                batch.append(event)
                last_pos = pos
                if len(batch) >= self.batch_size:
                    ok = self.sender.send_batch(batch)
                    if ok:
                        self.reader.commit(last_pos)
                        batch = []
                        last_time = time.time()
                    else:
                        time.sleep(0.5)
                self.queue.task_done()
            except queue.Empty:
                now = time.time()
                if batch and (now - last_time) * 1000 >= self.flush_ms:
                    ok = self.sender.send_batch(batch)
                    if ok:
                        self.reader.commit(last_pos)
                        batch = []
                        last_time = now
                    else:
                        time.sleep(0.5)
        if batch:
            ok = self.sender.send_batch(batch)
            if ok:
                self.reader.commit(last_pos)
    def run_manual(self, dry_run=False, from_start=None):
        if from_start is not None:
            if from_start:
                self.reader.pos = 0
            else:
                pass
        lines = self.reader.snapshot(count=10000000)
        events = [self.parser.parse(line) for _, line in lines]
        if dry_run:
            self.logger.info(f"dry-run events={len(events)}")
            return
        if events:
            ok = self.sender.send_batch(events)
            if ok:
                self.reader.commit(lines[-1][0])
    def run_daemon(self):
        t_reader = threading.Thread(target=self.reader.read_lines, args=(self.stop_event, self.queue), daemon=True)
        t_sender = threading.Thread(target=self._aggregate_and_send, daemon=True)
        t_reader.start()
        t_sender.start()
        try:
            while True:
                time.sleep(self.cfg.get("interval_sec", 2))
        except KeyboardInterrupt:
            self.stop_event.set()
            t_reader.join(timeout=2)
            t_sender.join(timeout=2)
    def health(self):
        ok = self.sender.health_check()
        if ok:
            self.logger.info("healthcheck ok")
        else:
            self.logger.error("healthcheck failed")

def build_logger():
    logger = logging.getLogger("log_push_service")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s')
    ensure_dir("work/logs")
    fh = logging.FileHandler(abs_path("work/logs/log_push_service.log"), encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

def setup_logger():
    logger = logging.getLogger('elk_integration')
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    file_handler = logging.FileHandler('elk_integration.log', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

def generate_test_logs(logger, duration_minutes=1):
    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)
    log_count = 0
    test_messages = ["用户登录成功","数据库查询执行","API 请求处理完成","缓存更新操作","文件上传成功","邮件发送完成","定时任务执行","系统健康检查","配置更新通知","性能监控数据"]
    log_levels = [(logging.INFO, "INFO"),(logging.WARNING, "WARNING"),(logging.ERROR, "ERROR"),(logging.DEBUG, "DEBUG")]
    while time.time() < end_time:
        level, _ = log_levels[log_count % len(log_levels)]
        message = test_messages[log_count % len(test_messages)]
        full_message = f"{message} - 序号: {log_count + 1}, 时间: {datetime.now().strftime('%H:%M:%S')}"
        logger.log(level, full_message)
        log_count += 1
        time.sleep(1)

def test_elk_connection(logger):
    logger.info("=== ELK 系统连接测试开始 ===")
    logger.debug("这是一条调试日志 - DEBUG level")
    logger.info("这是一条信息日志 - INFO level")
    logger.warning("这是一条警告日志 - WARNING level")
    logger.error("这是一条错误日志 - ERROR level")
    logger.critical("这是一条严重错误日志 - CRITICAL level")
    logger.info("测试中文字符和特殊符号: 你好世界! @#$%^&*()")
    try:
        raise ValueError("这是一个测试异常")
    except Exception:
        logger.exception("捕获到异常信息")
    logger.info("=== ELK 系统连接测试完成 ===")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="work/log_push_config.json")
    parser.add_argument("--mode", choices=["manual","daemon","health","test"], default="manual")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--from-start", action="store_true")
    args = parser.parse_args()
    logger = build_logger()
    cfg = load_config(args.config)
    service = LogPushService(cfg, logger)
    if args.mode == "health":
        service.health()
        return
    if args.mode == "test":
        tlogger = setup_logger()
        test_elk_connection(tlogger)
        return
    if args.mode == "manual":
        service.run_manual(dry_run=args.dry_run, from_start=args.from_start or cfg.get("from_start"))
        return
    if args.mode == "daemon":
        service.run_daemon()

if __name__ == "__main__":
    main()