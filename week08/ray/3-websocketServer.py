from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import os

# 禁用Ray的监控功能以避免metrics exporter错误
os.environ["RAY_DISABLE_IMPORT_WARNING"] = "1"
os.environ["RAY_USAGE_STATS_ENABLED"] = "0"

from ray import serve
import ray

# 初始化Ray时禁用监控
if not ray.is_initialized():
    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,  # 禁用dashboard
        _metrics_export_port=None,  # 禁用metrics导出
        _system_config={
            "metrics_report_interval_ms": 0,  # 禁用metrics报告
        }
    )


app = FastAPI()


@serve.deployment
@serve.ingress(app)
class EchoServer:
    @app.websocket("/")
    async def echo(self, ws: WebSocket):
        await ws.accept()

        try:
            while True:
                text = await ws.receive_text()
                await ws.send_text(text)
        except WebSocketDisconnect:
            print("Client disconnected.")


serve_app = serve.run(EchoServer.bind())

# 保持服务器运行
if __name__ == "__main__":
    print("WebSocket服务器已启动，监听地址: http://127.0.0.1:8000/")
    print("WebSocket端点: ws://127.0.0.1:8000/")
    print("按 Ctrl+C 停止服务器")
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("服务器已停止")