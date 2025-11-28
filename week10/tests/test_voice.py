import os
import dashscope
# 设置地域 API 地址（北京为例）
dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'
messages = [
   {"role": "system", "content": [{"text": "Bulge Bracket、Boutique、Middle Market"}]}, # 上下文增强
   {"role": "user", "content": [{"audio": "https://dashscope.oss-cn-beijing.aliyuncs.com/audios/welcome.mp3"}]}
]
response = dashscope.MultiModalConversation.call(
   api_key=os.getenv("DASHSCOPE_API_KEY"),
   model="qwen3-asr-flash",
   messages=messages,
   result_format="message",
   asr_options={
       "language": "zh", # 可选：指定语种
       "enable_itn": True # 开启逆文本规范化
   }
)

print(response["output"]["choices"][0]["message"]["content"][0]["text"])
