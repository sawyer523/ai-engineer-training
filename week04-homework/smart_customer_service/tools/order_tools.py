import random
import time
from langchain.tools import tool


@tool
def query_order(order_id: str) -> dict:
    """
    根据订单号查询订单状态和物流信息。
    当用户想要查询订单时，调用此工具。
    """
    print(f"--- [工具调用] 正在查询订单号: {order_id} ---")
    time.sleep(2)
    mock_db = {
        "SN20240924001": {"status": "已发货", "tracking_number": "SF123456789", "items": ["LangChain入门实战T恤"]},
        "SN20240925001": {"status": "已发货", "tracking_number": "SF987654321", "items": ["AI Agent开发者马克杯"]},
        "SN20240924002": {"status": "待支付", "tracking_number": None, "items": ["LangGraph高级教程贴纸"]},
        "SN20240924003": {"status": "已完成", "tracking_number": "JD987654321", "items": ["AI Agent开发者马克杯"]},
    }
    order_info = mock_db.get(order_id)
    if order_info:
        return {
            "success": True,
            "order_id": order_id,
            "status": order_info["status"],
            "tracking_number": order_info["tracking_number"],
            "details": f"订单中的商品: {', '.join(order_info['items'])}"
        }
    else:
        return {
            "success": False,
            "order_id": order_id,
            "error": "未找到该订单，请检查订单号是否正确。"
        }


@tool
def apply_refund(order_id: str, reason: str) -> dict:
    """
    为指定订单号的订单申请退款。
    需要提供订单号和退款原因。
    """
    print(f"--- [工具调用] 正在为订单号 {order_id} 申请退款，原因: {reason} ---")
    time.sleep(1)
    if "SN" in order_id:
        refund_id = f"REFUND_{random.randint(1000, 9999)}"
        return {
            "success": True,
            "order_id": order_id,
            "refund_id": refund_id,
            "message": "退款申请已提交，审核通过后将原路退回。"
        }
    else:
        return {
            "success": False,
            "order_id": order_id,
            "error": "无效的订单号，无法申请退款。"
        }


@tool
def generate_invoice(order_id: str) -> dict:
    """
    为指定订单号的订单生成发票。
    当用户需要开具发票时调用。
    """
    print(f"--- [工具调用] 正在为订单号 {order_id} 生成发票 ---")
    time.sleep(1)
    if "SN" in order_id:
        invoice_url = f"https://example.com/invoices/{order_id}.pdf"
        return {
            "success": True,
            "order_id": order_id,
            "invoice_url": invoice_url,
            "message": f"发票已生成，您可以从以下链接下载：{invoice_url}"
        }
    else:
        return {
            "success": False,
            "order_id": order_id,
            "error": "无效的订单号，无法生成发票。"
        }
