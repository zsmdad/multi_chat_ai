import os
import json
import gradio as gr
import openai
import uuid
import time
from typing import List, Dict, Any, Optional
import threading

# 配置文件路径
CONFIG_FILE = "multichat_config.json"

# 默认配置
DEFAULT_CONFIG = {
    "models": []
}

class ModelConfig:
    def __init__(self, name: str, url: str, token: str, model: str, system_prompt: str, temperature: float = 0.7):
        self.name = name
        self.url = url
        self.token = token
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.client = None
        self.update_client()
    
    def update_client(self):
        try:
            self.client = openai.OpenAI(
                base_url=self.url,
                api_key=self.token
            )
        except Exception as e:
            print(f"Error initializing client for {self.name}: {e}")
            self.client = None
    
    def to_dict(self):
        return {
            "name": self.name,
            "url": self.url,
            "token": self.token,
            "model": self.model,
            "system_prompt": self.system_prompt,
            "temperature": self.temperature
        }

class MultiChatAI:
    def __init__(self):
        self.configs: Dict[str, ModelConfig] = {}
        self.load_config()
        self.conversation_history: Dict[str, List[Dict[str, Any]]] = {}
    
    def load_config(self):
        """从配置文件加载模型配置"""
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                for model_config in config.get("models", []):
                    model = ModelConfig(
                        name=model_config.get("name", ""),
                        url=model_config.get("url", ""),
                        token=model_config.get("token", ""),
                        model=model_config.get("model", ""),
                        system_prompt=model_config.get("system_prompt", ""),
                        temperature=model_config.get("temperature", 0.7)
                    )
                    self.configs[model.name] = model
            except Exception as e:
                print(f"Error loading config: {e}")
                # 如果加载失败，创建一个空配置
                self.configs = {}
        else:
            # 配置文件不存在，创建默认配置
            self.save_config()
    
    def save_config(self):
        """保存模型配置到文件"""
        config = {
            "models": [model.to_dict() for model in self.configs.values()]
        }
        try:
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def add_model(self, name: str, url: str, token: str, model: str, system_prompt: str, temperature: float = 0.7) -> str:
        """添加或更新模型配置"""
        if not name:
            return "模型名称不能为空"
        
        if not url:
            return "API URL不能为空"
        
        if not token:
            return "API Token不能为空"
        
        if not model:
            return "模型名称不能为空"
        
        # 创建新的模型配置
        model_config = ModelConfig(name, url, token, model, system_prompt, temperature)
        
        # 测试连接
        try:
            if model_config.client is None:
                return "API连接初始化失败"
            
            # 简单的API测试
            response = model_config.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello!"}
                ],
                max_tokens=10
            )
            
            # 如果没有异常，则保存配置
            self.configs[name] = model_config
            self.save_config()
            return f"模型 {name} 添加成功!"
        
        except Exception as e:
            return f"API连接测试失败: {str(e)}"
    
    def remove_model(self, name: str) -> str:
        """删除模型配置"""
        if name in self.configs:
            del self.configs[name]
            self.save_config()
            return f"模型 {name} 已删除"
        return f"模型 {name} 不存在"
    
    def get_model_names(self) -> List[str]:
        """获取所有配置的模型名称"""
        return list(self.configs.keys())
    
    def chat_with_model(self, model_name: str, message: str, conversation_id: str = None) -> Dict[str, Any]:
        """与指定模型对话"""
        if model_name not in self.configs:
            return {
                "error": f"模型 {model_name} 不存在",
                "response": f"错误: 模型 {model_name} 未配置"
            }
        
        model_config = self.configs[model_name]
        
        # 创建新的对话ID
        if not conversation_id:
            conversation_id = f"{model_name}_{uuid.uuid4()}"
        
        # 确保对话历史存在
        if conversation_id not in self.conversation_history:
            self.conversation_history[conversation_id] = []
            # 添加系统提示
            if model_config.system_prompt:
                self.conversation_history[conversation_id].append(
                    {"role": "system", "content": model_config.system_prompt}
                )
        
        # 添加用户消息到历史
        self.conversation_history[conversation_id].append(
            {"role": "user", "content": message}
        )
        
        try:
            # 创建消息列表
            messages = self.conversation_history[conversation_id].copy()
            
            # 发送请求到API
            start_time = time.time()
            response = model_config.client.chat.completions.create(
                model=model_config.model,
                messages=messages,
                temperature=model_config.temperature,
            )
            end_time = time.time()
            
            # 提取响应文本
            assistant_message = response.choices[0].message.content
            
            # 将助手回复添加到历史记录
            self.conversation_history[conversation_id].append(
                {"role": "assistant", "content": assistant_message}
            )
            
            return {
                "conversation_id": conversation_id,
                "response": assistant_message,
                "model": model_name,
                "time_taken": f"{end_time - start_time:.2f}秒"
            }
            
        except Exception as e:
            error_msg = f"与模型 {model_name} 通信时出错: {str(e)}"
            return {
                "conversation_id": conversation_id,
                "response": error_msg,
                "model": model_name,
                "error": str(e)
            }
    
    def chat_with_multiple_models(self, model_names: List[str], message: str) -> Dict[str, Any]:
        """同时向多个模型发送相同的消息"""
        responses = {}
        threads = []
        
        def thread_chat(model_name):
            response = self.chat_with_model(model_name, message)
            responses[model_name] = response
        
        # 创建并启动线程
        for model_name in model_names:
            thread = threading.Thread(target=thread_chat, args=(model_name,))
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        return responses
    
    def clear_conversation(self, conversation_id: str) -> str:
        """清除特定对话的历史记录"""
        if conversation_id in self.conversation_history:
            # 保留系统消息
            system_messages = [msg for msg in self.conversation_history[conversation_id] if msg["role"] == "system"]
            self.conversation_history[conversation_id] = system_messages
            return f"对话 {conversation_id} 已清除"
        return f"对话 {conversation_id} 不存在"

# 创建MultiChatAI实例
chat_ai = MultiChatAI()

# 创建Gradio界面
def create_interface():
    with gr.Blocks(title="MultiChat AI - 多窗智聊") as app:
        gr.Markdown("# MultiChat AI (多窗智聊)")
        gr.Markdown("基于Gradio的多模型对话平台，支持同时向多个大模型发送请求")
        
        with gr.Tabs() as tabs:
            # 模型配置标签页
            with gr.TabItem("模型配置"):
                with gr.Row():
                    with gr.Column():
                        model_name_input = gr.Textbox(label="模型名称", placeholder="给这个配置起个名字")
                        api_url_input = gr.Textbox(label="API URL", placeholder="例如: https://api.openai.com/v1")
                        api_token_input = gr.Textbox(label="API Token", placeholder="输入API Token", type="password")
                        model_input = gr.Textbox(label="模型ID", placeholder="例如: gpt-4-turbo")
                        system_prompt_input = gr.Textbox(label="系统提示词", placeholder="输入默认系统提示词", lines=3)
                        temperature_input = gr.Number(label="Temperature", value=0.7, minimum=0.0, maximum=1.0, step=0.05)
                        
                        with gr.Row():
                            add_btn = gr.Button("添加/更新模型", variant="primary")
                            test_btn = gr.Button("测试连接")
                        
                        config_output = gr.Textbox(label="配置结果", lines=2)
                    
                    with gr.Column():
                        model_list = gr.Dropdown(label="已配置模型", choices=chat_ai.get_model_names(), interactive=True)
                        remove_btn = gr.Button("删除选中模型", variant="stop")
                        model_details = gr.JSON(label="模型详情")
            
            # 单模型对话标签页
            with gr.TabItem("单模型对话"):
                with gr.Row():
                    with gr.Column(scale=1):
                        single_model_selector = gr.Dropdown(
                            label="选择模型", 
                            choices=chat_ai.get_model_names(),
                            interactive=True
                        )
                        single_chat_clear = gr.Button("清除对话")
                    
                    with gr.Column(scale=3):
                        single_chat_history = gr.Chatbot(label="对话历史")
                        single_chat_input = gr.Textbox(
                            label="发送消息", 
                            placeholder="输入消息...",
                            lines=2
                        )
                        single_chat_info = gr.Textbox(label="信息", interactive=False)
                        single_send_btn = gr.Button("发送", variant="primary")
            
            # 多模型对话标签页
            with gr.TabItem("多模型对比"):
                with gr.Row():
                    with gr.Column(scale=1):
                        multi_model_selector = gr.CheckboxGroup(
                            label="选择模型",
                            choices=chat_ai.get_model_names()
                        )
                        multi_chat_clear = gr.Button("清除所有对话")
                    
                    with gr.Column(scale=3):
                        multi_chat_input = gr.Textbox(
                            label="发送消息到所有选中的模型",
                            placeholder="输入一条消息发送到所有选中的模型...",
                            lines=2
                        )
                        multi_send_btn = gr.Button("发送到所有模型", variant="primary")
                
                # 动态创建多模型对话区域
                multi_chatbots = {}
                multi_infos = {}
                
                with gr.Row():
                    for i, model_name in enumerate(chat_ai.get_model_names()):
                        if i % 2 == 0 and i > 0:  # 每两个模型一行
                            with gr.Row():
                                create_model_chatbox(model_name, multi_chatbots, multi_infos)
                        else:
                            create_model_chatbox(model_name, multi_chatbots, multi_infos)
        
        # 存储当前单聊对话ID
        current_conversation_id = gr.State(None)
        
        # 更新模型列表的函数
        def update_model_lists():
            model_names = chat_ai.get_model_names()
            return {
                model_list: gr.Dropdown(choices=model_names),
                single_model_selector: gr.Dropdown(choices=model_names),
                multi_model_selector: gr.CheckboxGroup(choices=model_names)
            }
        
        # 添加/更新模型按钮功能
        def add_model_config(name, url, token, model, system_prompt, temperature):
            result = chat_ai.add_model(name, url, token, model, system_prompt, temperature)
            updates = update_model_lists()
            return {config_output: result, **updates}
        
        add_btn.click(
            add_model_config,
            inputs=[model_name_input, api_url_input, api_token_input, model_input, system_prompt_input, temperature_input],
            outputs=[config_output, model_list, single_model_selector, multi_model_selector]
        )
        
        # 测试连接按钮功能
        def test_connection(url, token, model):
            try:
                client = openai.OpenAI(base_url=url, api_key=token)
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "Hello!"}],
                    max_tokens=5
                )
                return "连接测试成功!"
            except Exception as e:
                return f"连接测试失败: {str(e)}"
        
        test_btn.click(
            test_connection,
            inputs=[api_url_input, api_token_input, model_input],
            outputs=[config_output]
        )
        
        # 删除模型按钮功能
        def remove_model_config(name):
            result = chat_ai.remove_model(name)
            updates = update_model_lists()
            return {model_details: None, **updates, **{config_output: result}}
        
        remove_btn.click(
            remove_model_config,
            inputs=[model_list],
            outputs=[model_details, model_list, single_model_selector, multi_model_selector, config_output]
        )
        
        # 显示模型详情
        def show_model_details(name):
            if name in chat_ai.configs:
                return chat_ai.configs[name].to_dict()
            return None
        
        model_list.change(
            show_model_details,
            inputs=[model_list],
            outputs=[model_details]
        )
        
        # 模型列表选择时填充配置
        def populate_model_config(name):
            if name in chat_ai.configs:
                model_config = chat_ai.configs[name]
                return {
                    model_name_input: model_config.name,
                    api_url_input: model_config.url,
                    api_token_input: model_config.token,
                    model_input: model_config.model,
                    system_prompt_input: model_config.system_prompt,
                    temperature_input: model_config.temperature
                }
            else:
                return {
                    model_name_input: "",
                    api_url_input: "",
                    api_token_input: "",
                    model_input: "",
                    system_prompt_input: "",
                    temperature_input: 0.7
                }
        
        model_list.change(
            populate_model_config,
            inputs=[model_list],
            outputs=[model_name_input, api_url_input, api_token_input, model_input, system_prompt_input, temperature_input]
        )
        
        # 单模型对话功能
        def single_chat(model_name, message, conversation_id):
            if not model_name:
                return {single_chat_info: "请选择一个模型"}

            if not message.strip():
                return {single_chat_info: "消息不能为空"}
            result = chat_ai.chat_with_model(model_name, message, conversation_id)
            
            if "error" in result:
                return {
                    single_chat_info: f"错误: {result['error']}",
                    current_conversation_id: conversation_id
                }
            
            # 更新聊天历史
            history = []
            conversation_id = result["conversation_id"]
            conv_history = chat_ai.conversation_history[conversation_id]
            
            # 只显示用户和助手消息，跳过系统消息
            for i in range(len(conv_history)):
                if conv_history[i]["role"] == "user":
                    user_msg = conv_history[i]["content"]
                    if i+1 < len(conv_history) and conv_history[i+1]["role"] == "assistant":
                        assistant_msg = conv_history[i+1]["content"]
                        history.append([user_msg, assistant_msg])
                    else:
                        history.append([user_msg, None])
            
            return {
                single_chat_history: history,
                single_chat_input: "",
                single_chat_info: f"模型: {model_name} | 耗时: {result.get('time_taken', 'N/A')}",
                current_conversation_id: conversation_id
            }
        
        single_send_btn.click(
            single_chat,
            inputs=[single_model_selector, single_chat_input, current_conversation_id],
            outputs=[single_chat_history, single_chat_input, single_chat_info, current_conversation_id]
        )
        
        # 清除单聊对话历史
        def clear_single_chat(conversation_id):
            if conversation_id:
                chat_ai.clear_conversation(conversation_id)
            return {
                single_chat_history: [],
                single_chat_info: "对话已清除",
                current_conversation_id: None
            }
        
        single_chat_clear.click(
            clear_single_chat,
            inputs=[current_conversation_id],
            outputs=[single_chat_history, single_chat_info, current_conversation_id]
        )
        
        # 多模型对话功能
        def multi_chat(model_names, message):
            if not model_names:
                return {k: "请至少选择一个模型" for k in multi_infos.values()}
            
            if not message.strip():
                return {k: "消息不能为空" for k in multi_infos.values()}
            
            results = chat_ai.chat_with_multiple_models(model_names, message)
            
            updates = {}
            for model_name in chat_ai.get_model_names():
                if model_name in results:
                    result = results[model_name]
                    
                    # 更新聊天历史
                    history = []
                    if "conversation_id" in result:
                        conversation_id = result["conversation_id"]
                        if conversation_id in chat_ai.conversation_history:
                            conv_history = chat_ai.conversation_history[conversation_id]
                            
                            # 只显示用户和助手消息，跳过系统消息
                            for i in range(len(conv_history)):
                                if conv_history[i]["role"] == "user":
                                    user_msg = conv_history[i]["content"]
                                    if i+1 < len(conv_history) and conv_history[i+1]["role"] == "assistant":
                                        assistant_msg = conv_history[i+1]["content"]
                                        history.append([user_msg, assistant_msg])
                                    else:
                                        history.append([user_msg, None])
                    
                    updates[multi_chatbots[model_name]] = history
                    updates[multi_infos[model_name]] = f"耗时: {result.get('time_taken', 'N/A')}"
                else:
                    updates[multi_infos[model_name]] = "未选择此模型"
            
            updates[multi_chat_input] = ""
            return updates
        
        multi_send_btn.click(
            multi_chat,
            inputs=[multi_model_selector, multi_chat_input],
            outputs=[multi_chat_input] + list(multi_chatbots.values()) + list(multi_infos.values())
        )
        
        # 清除所有多模型对话
        def clear_multi_chats():
            for model_name in chat_ai.get_model_names():
                for conversation_id in list(chat_ai.conversation_history.keys()):
                    if conversation_id.startswith(f"{model_name}_"):
                        chat_ai.clear_conversation(conversation_id)
            
            updates = {}
            for chatbot in multi_chatbots.values():
                updates[chatbot] = []
            for info in multi_infos.values():
                updates[info] = "对话已清除"
            
            return updates
        
        multi_chat_clear.click(
            clear_multi_chats,
            outputs=list(multi_chatbots.values()) + list(multi_infos.values())
        )
        
        return app

def create_model_chatbox(model_name, chatbots_dict, infos_dict):
    with gr.Column():
        gr.Markdown(f"### {model_name}")
        chatbot = gr.Chatbot(label=f"{model_name} 对话", height=400)
        info = gr.Textbox(label="信息", value="", interactive=False)
        chatbots_dict[model_name] = chatbot
        infos_dict[model_name] = info

# 启动应用
if __name__ == "__main__":
    app = create_interface()
    app.launch()
def create_model_chatbox(model_name, chatbots_dict, infos_dict):
    with gr.Column():
        gr.Markdown(f"### {model_name}")
        chatbot = gr.Chatbot(label=f"{model_name} 对话", height=400)
        info = gr.Textbox(label="信息", value="", interactive=False)
        chatbots_dict[model_name] = chatbot
        infos_dict[model_name] = info

# 启动应用
if __name__ == "__main__":
    app = create_interface()
    app.launch()

