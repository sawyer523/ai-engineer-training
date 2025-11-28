"""提示词配置模块：包含意图路由、RAG 与订单话术生成的模板。"""

# 意图路由提示词：将输出限制为特定标签集合
INTENT_PROMPT = (
    "请只输出一个词：course、presale、postsale、order、human 或 direct。\n"
    "当问题涉及课程咨询、学习建议、FAQ、售前或售后时，输出 course/presale/postsale；\n"
    "当问题涉及订单号、支付、退款、状态、进度时，输出 order；\n"
    "用户明确要求人工或转人工时输出 human；否则输出 direct。\n"
)

# RAG 回答模板：只依据参考资料的 Content 字段作答
RAG_PROMPT_TEMPLATE = (
    "你是一个严谨的客服问答助手。你的回答必须只依据参考资料的 Content 字段。\n"
    "参考资料：\n{context}\n\n问题：{question}\n"
)

# 直答模板：无 KB/订单时的简要回答
DIRECT_PROMPT_TEMPLATE = "请简要回答用户问题：{question}"

# 订单摘要前缀（用于拼接展示）
ORDER_SUMMARY_PREFIX = "订单查询结果："

# 订单话术生成模板：指导 LLM 生成专业客服说明
ORDER_NLG_PROMPT_TEMPLATE = (
    "请用专业客服话术，以中文向用户说明订单进展。\n"
    "订单信息：订单号 {order_id}；状态 {status}；金额 {amount}；最近更新时间 {updated_at}；开课时间 {start_time}。\n"
    "如存在开课时间，则表述为已报名成功并给出具体时间；\n"
    "提醒开课前做好预习；\n"
    "只输出自然语言句子，不要列表或JSON。"
)

ORDER_SQL_PROMPT_TEMPLATE = (
    "请将用户查询转换为安全的 SQL（仅查询表 orders：order_id、status、amount、updated_at、start_time）。\n"
    "必须使用参数化占位符 %s，并同时给出参数列表 params；不要直接拼接值。\n"
    "只输出结构化内容用于解析。\n"
    "用户查询：{question}"
)
