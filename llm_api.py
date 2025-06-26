import tkinter as tk
from tkinter import scrolledtext, messagebox
import os
from openai import OpenAI


class ChatApp:
    def __init__(self, root, api_key):
        self.root = root
        self.api_key = api_key
        self.history = []
        
        # 构造 client
        self.client = OpenAI(
            api_key=api_key,  # 混元 APIKey
            base_url="https://api.hunyuan.cloud.tencent.com/v1",  # 混元 endpoint
        )
        
        root.title("大模型对话助手")
        root.geometry("600x400")
        root.resizable(True, True)
        
        # 创建聊天显示区域
        self.chat_display = scrolledtext.ScrolledText(
            root, wrap=tk.WORD, state='disabled', font=("微软雅黑", 10))
        self.chat_display.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # 创建输入区域
        input_frame = tk.Frame(root)
        input_frame.pack(padx=0, pady=0, fill=tk.X)
        
        self.user_input = tk.Entry(
            input_frame, font=("微软雅黑", 20))
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=3)
        self.user_input.bind("<Return>", self.send_message)
        
        send_button = tk.Button(
            input_frame, text="发送", command=self.send_message, 
            bg="#4CAF50", fg="white", font=("微软雅黑", 10))
        send_button.pack(side=tk.RIGHT, padx=(0, 10))
        
        # 添加清空按钮
        clear_button = tk.Button(
            root, text="清空对话", command=self.clear_chat,
            bg="#f44336", fg="white", font=("微软雅黑", 9))
        clear_button.pack(pady=5)
        
        # 初始问候
        self.display_message("助手", "你好！请问有什么可以帮您？")
    
    def display_message(self, sender, message):
        self.chat_display.configure(state='normal')
        self.chat_display.insert(tk.END, f"{sender}: {message}\n\n")
        self.chat_display.configure(state='disabled')
        self.chat_display.see(tk.END)  # 自动滚动到底部
    
    def send_message(self, event=None):
        user_text = self.user_input.get().strip()
        if not user_text:
            return
            
        # 显示用户消息
        self.display_message("您", user_text)
        self.user_input.delete(0, tk.END)
        
        # 添加到历史记录
        self.history.append({"role": "user", "content": user_text})
        
        try:
            # 调用API
            response = self.call_api(user_text)
            
            # 显示助手回复
            self.display_message("助手", response)
            self.history.append({"role": "assistant", "content": response})
        except Exception as e:
            self.display_message("系统", f"API调用失败: {str(e)}")
    
    def call_api(self, message):
        try:
            completion = self.client.chat.completions.create(
                model="hunyuan-turbos-latest",
                messages=self.history,
                extra_body={
                    "enable_enhancement": True,  # 自定义参数
                },
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            raise Exception(f"API调用错误: {str(e)}")
    
    def clear_chat(self):
        self.chat_display.configure(state='normal')
        self.chat_display.delete(1.0, tk.END)
        self.chat_display.configure(state='disabled')
        self.history = []
        self.display_message("助手", "对话已重置，请问有什么可以帮您？")


if __name__ == "__main__":
    # 配置你的API信息
    API_KEY = "sk-RKzGuK3W7kwxfvGcnILRTwqQFCe8Rz74v86qlMMC7fyAKJl2"  # 替换为你的API密钥
    
    root = tk.Tk()
    app = ChatApp(root, API_KEY)
    root.mainloop()
