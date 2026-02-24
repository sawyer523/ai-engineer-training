from datetime import datetime, timedelta
from langchain.tools import tool


@tool
def get_date_for_relative_time(relative_time_str: str) -> str:
    """
    将相对时间描述（如"昨天"、"前天"、"上周三"）转换为"YYYY-MM-DD"格式的具体日期。
    今天的日期是 2025-09-26。
    """
    print(f"--- [工具调用] 正在解析相对时间: {relative_time_str} ---")
    today = datetime(2025, 9, 26)
    relative_time_str = relative_time_str.lower()

    if "昨天" in relative_time_str:
        target_date = today - timedelta(days=1)
    elif "前天" in relative_time_str:
        target_date = today - timedelta(days=2)
    elif "今天" in relative_time_str:
        target_date = today
    # 简单实现"上周X"
    elif "上周" in relative_time_str:
        weekday_map = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6}
        target_weekday = -1
        for day_char, day_index in weekday_map.items():
            if day_char in relative_time_str:
                target_weekday = day_index
                break

        if target_weekday != -1:
            days_ago = (today.weekday() - target_weekday + 7) % 7 + 7
            target_date = today - timedelta(days=days_ago)
        else:
            return "无法识别的星期信息。"
    else:
        return "无法解析该相对时间，请使用更明确的描述。"

    return target_date.strftime("%Y-%m-%d")
