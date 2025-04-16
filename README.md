# multi_chat_ai
Multi Chat AI is a Gradio-based platform for interacting with multiple large language models simultaneously. Configure various OpenAI-format APIs with custom parameters and send queries to one or multiple models concurrently. Compare responses across different AI systems in a unified interface.

# 多窗智聊
多窗智聊是基于Gradio框架的多模型对话平台，支持配置多个OpenAI格式的大模型接口参数，可同时向一个或多个AI模型发送请求，在统一界面中比较不同模型的响应。

## How to run
1. Install the required packages:
```bash
pip install -r requirements.txt
```
2. Run the main script:
```bash
python main.py
```
3. Access the Gradio interface at [http://localhost:7860](http://localhost:7860).


## Docker Usage
1. Build the Docker image:
```bash
docker build -t multi_chat_ai .
```
2. Run the Docker container:
```bash
docker run -p 7860:7860 -v $(pwd)/data:/app/data multi_chat_ai
```
3. Access the Gradio interface at [http://localhost:7860](http://localhost:7860).

