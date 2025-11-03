"""
Gradio界面应用程序，用于与LangGraph工作流进行对话和查询历史。
"""
import logging
import uuid
from datetime import datetime
from typing import List, Tuple, Optional

import gradio as gr
import requests
import pandas as pd

from config import config
from database import db_manager

# 配置日志
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)

# API基础URL
API_BASE_URL = f"http://{config.HOST}:{config.PORT}"


class GradioApp:
    """Gradio应用程序类。"""
    
    def __init__(self):
        self.current_session_id = str(uuid.uuid4())
        logger.info(f"初始化Gradio应用，会话ID: {self.current_session_id}")
    
    def chat_with_ai(self, message: str, history: List[dict]) -> Tuple[str, List[dict]]:
        """
        与AI进行对话。
        
        Args:
            message: 用户输入消息
            history: 对话历史（消息格式）
            
        Returns:
            空字符串和更新后的历史记录
        """
        if not message.strip():
            return "", history
        
        try:
            # 调用API
            response = requests.post(
                f"{API_BASE_URL}/run",
                json={
                    "user_input": message,
                    "session_id": self.current_session_id
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result.get("result", "抱歉，没有收到有效响应。")
                
                # 更新历史记录 - 使用消息格式
                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": ai_response})
                logger.info(f"对话成功，会话ID: {self.current_session_id}")
                
            else:
                error_msg = f"API调用失败，状态码: {response.status_code}"
                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": error_msg})
                logger.error(error_msg)
                
        except requests.exceptions.RequestException as e:
            error_msg = f"网络请求错误: {str(e)}"
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": error_msg})
            logger.error(error_msg)
        except Exception as e:
            error_msg = f"未知错误: {str(e)}"
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": error_msg})
            logger.error(error_msg)
        
        return "", history
    
    def get_conversation_history(self, limit: int = 50, session_filter: str = "") -> pd.DataFrame:
        """
        获取对话历史。
        
        Args:
            limit: 记录数限制
            session_filter: 会话ID过滤
            
        Returns:
            包含对话历史的DataFrame
        """
        try:
            # 构建请求参数
            params = {"limit": limit}
            if session_filter.strip():
                params["session_id"] = session_filter.strip()
            
            # 调用API
            response = requests.get(
                f"{API_BASE_URL}/history",
                params=params,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                history_data = result.get("history", [])
                
                if history_data:
                    # 转换为DataFrame
                    df = pd.DataFrame(history_data)
                    # 重新排列列顺序
                    df = df[["id", "timestamp", "session_id", "user_input", "ai_response"]]
                    # 格式化时间戳
                    if "timestamp" in df.columns:
                        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
                    
                    logger.info(f"成功获取 {len(df)} 条历史记录")
                    return df
                else:
                    logger.info("没有找到历史记录")
                    return pd.DataFrame(columns=["id", "timestamp", "session_id", "user_input", "ai_response"])
            else:
                logger.error(f"获取历史记录失败，状态码: {response.status_code}")
                return pd.DataFrame(columns=["错误"], data=[[f"API调用失败: {response.status_code}"]])
                
        except requests.exceptions.RequestException as e:
            logger.error(f"网络请求错误: {e}")
            return pd.DataFrame(columns=["错误"], data=[[f"网络请求错误: {str(e)}"]])
        except Exception as e:
            logger.error(f"获取历史记录时出错: {e}")
            return pd.DataFrame(columns=["错误"], data=[[f"未知错误: {str(e)}"]])
    
    def new_session(self) -> Tuple[str, List[dict]]:
        """
        开始新的对话会话。
        
        Returns:
            新的会话ID和空的历史记录
        """
        self.current_session_id = str(uuid.uuid4())
        logger.info(f"开始新会话: {self.current_session_id}")
        return f"新会话已开始: {self.current_session_id}", []
    
    def get_current_session_info(self) -> str:
        """
        获取当前会话信息。
        
        Returns:
            当前会话ID
        """
        return f"当前会话ID: {self.current_session_id}"
    
    def create_interface(self) -> gr.Blocks:
        """
        创建Gradio界面。
        
        Returns:
            Gradio Blocks界面
        """
        with gr.Blocks(
            title="LangGraph AI 对话助手",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 1200px !important;
            }
            .chat-container {
                height: 500px !important;
            }
            """
        ) as interface:
            
            gr.Markdown("# 🤖 LangGraph AI 对话助手")
            gr.Markdown("与AI进行智能对话，并查看完整的对话历史记录。")
            
            with gr.Tab("💬 AI对话"):
                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(
                            label="对话窗口",
                            height=500,
                            show_copy_button=True,
                            type="messages"
                        )
                        
                        with gr.Row():
                            msg_input = gr.Textbox(
                                label="输入消息",
                                placeholder="请输入您的问题...",
                                lines=2,
                                scale=4
                            )
                            send_btn = gr.Button("发送", variant="primary", scale=1)
                        
                        with gr.Row():
                            clear_btn = gr.Button("清空对话", variant="secondary")
                            new_session_btn = gr.Button("新建会话", variant="secondary")
                    
                    with gr.Column(scale=1):
                        session_info = gr.Textbox(
                            label="会话信息",
                            value=self.get_current_session_info(),
                            interactive=False
                        )
                        
                        gr.Markdown("### 💡 使用提示")
                        gr.Markdown("""
                        - 输入问题后点击"发送"或按Enter键
                        - 点击"新建会话"开始新的对话
                        - 所有对话都会自动保存到数据库
                        - 可在"历史记录"标签页查看所有对话
                        """)
            
            with gr.Tab("📚 历史记录"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 对话历史查询")
                        
                        with gr.Row():
                            limit_input = gr.Number(
                                label="记录数限制",
                                value=50,
                                minimum=1,
                                maximum=200,
                                step=1
                            )
                            session_filter_input = gr.Textbox(
                                label="会话ID过滤（可选）",
                                placeholder="输入会话ID进行过滤..."
                            )
                            query_btn = gr.Button("查询历史", variant="primary")
                        
                        history_display = gr.Dataframe(
                            label="对话历史",
                            headers=["ID", "时间", "会话ID", "用户输入", "AI回复"],
                            datatype=["number", "str", "str", "str", "str"],
                            wrap=True
                        )
                        
                        refresh_btn = gr.Button("刷新", variant="secondary")
            
            # 事件绑定
            def submit_message(message, history):
                return self.chat_with_ai(message, history)
            
            def clear_chat():
                return []
            
            def new_session():
                info, history = self.new_session()
                return info, history
            
            def query_history(limit, session_filter):
                return self.get_conversation_history(limit, session_filter)
            
            # 绑定事件
            msg_input.submit(
                submit_message,
                inputs=[msg_input, chatbot],
                outputs=[msg_input, chatbot]
            )
            
            send_btn.click(
                submit_message,
                inputs=[msg_input, chatbot],
                outputs=[msg_input, chatbot]
            )
            
            clear_btn.click(
                clear_chat,
                outputs=[chatbot]
            )
            
            new_session_btn.click(
                new_session,
                outputs=[session_info, chatbot]
            )
            
            query_btn.click(
                query_history,
                inputs=[limit_input, session_filter_input],
                outputs=[history_display]
            )
            
            refresh_btn.click(
                query_history,
                inputs=[limit_input, session_filter_input],
                outputs=[history_display]
            )
            
            # 页面加载时自动查询历史记录
            interface.load(
                query_history,
                inputs=[gr.Number(value=50), gr.Textbox(value="")],
                outputs=[history_display]
            )
        
        return interface


def main():
    """主函数，启动Gradio应用。"""
    try:
        # 初始化数据库
        db_manager.init_database()
        logger.info("数据库初始化完成")
        
        # 创建应用实例
        app = GradioApp()
        interface = app.create_interface()
        
        # 启动界面
        logger.info(f"启动Gradio界面，地址: http://localhost:7860")
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True
        )
        
    except Exception as e:
        logger.error(f"启动Gradio应用失败: {e}")
        raise


if __name__ == "__main__":
    main()