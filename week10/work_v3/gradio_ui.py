import sqlite3
from typing import List, Any

try:
    from . import config
except Exception:
    import config as config

import gradio as gr


def _load_orders(tenant_id: str = "default") -> List[List[Any]]:
    p = config.get_orders_db_path(tenant_id)
    if not p:
        return []
    try:
        with sqlite3.connect(p) as conn:
            rows = conn.execute(
                "SELECT order_id, status, amount, updated_at, start_time FROM orders"
            ).fetchall()
    except Exception:
        return []
    return [
        [
            str(oid or ""),
            str(status or ""),
            float(amount) if amount is not None else None,
            str(updated_at or ""),
            str(start_time or ""),
        ]
        for (oid, status, amount, updated_at, start_time) in rows
    ]


def build_orders_ui():
    headers = ["order_id", "status", "amount", "updated_at", "start_time"]
    with gr.Blocks() as demo:
        gr.Markdown("订单数据库")
        tenant = gr.Textbox(label="tenant", value="default")
        df = gr.Dataframe(headers=headers, value=_load_orders("default"), interactive=False)
        btn = gr.Button("刷新")
        btn.click(fn=_load_orders, inputs=tenant, outputs=df, api_name="lambda", show_progress="minimal")
    return demo


def mount_gradio(app):
    demo = build_orders_ui()
    try:
        if hasattr(gr, "mount_gradio_app"):
            gr.mount_gradio_app(app, demo, path="/listdb")
        else:
            from gradio.routes import App as GradioApp
            app.mount("/listdb", GradioApp.create_app(demo))
        return True
    except Exception:
        return False
