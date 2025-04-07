import gradio as gr
import requests

from core.settings import Config


def process_text_gradio(input_text, history):
    history = history or []
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer YOUR_API_KEY"
    }
    data = {
        "model": "deepseek-r1-qwen-1.5",
        "messages": [
            {"role": "system", "content": "你是一个热情的助手"},
            {"role": "user", "content": input_text}
        ],
        "max_tokens": 1000
    }

    response = requests.post(Config.MY_DEEPSEEK_HOST, headers=headers, json=data)
    history.append((input_text, response.json()["choices"][0]['message']['content']))
    return history, history


# 输入接口组件，label:显示框的标签,lines 行数为五行，placeholder
input = gr.Text(label="输入文字", lines=5, placeholder="请在这里输入...", )

# 输出接口组件，label:显示框的标签

chatbot = gr.Chatbot(label="聊天记录", height=400, min_width=500)
chatbot.style = {"background-color": "green"}

# 创建 Gradio 界面
interface = gr.Interface(
    fn=process_text_gradio,
    inputs=[input, "state"],
    outputs=[chatbot, "state"],
    live=False,
)

# 启动界面
interface.launch()
