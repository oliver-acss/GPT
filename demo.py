import gradio as gr

import torch
from transformers import BertTokenizer

from GPT import GPT
from dataHandle import data_loader


def id2String(Tokenizer:BertTokenizer,Input_id):
    result=""
    for Id in Input_id:
        result=result+' '+Tokenizer.decode(Id)
    return result


def algorithm(x):
    tokenizer=BertTokenizer.from_pretrained(r'D:\AIGCpractice\Day1\GPT\bert-base-multilingual-uncased')
    model = GPT(768, tokenizer.vocab_size, 12, 3072, 128, 0.4)
    model.load_state_dict(torch.load("./last_model.pt"))
    model.eval() #

    device = torch.device("cpu")
    model.to(device)

    x= tokenizer(x, max_length=128, truncation=True, padding='max_length', return_tensors='pt')
    x = x['input_ids'].reshape(-1)[None,]
    #print(x.shape)
    x=torch.tensor(x).to(device)

    output = model(x)
    Input_id=output.argmax(dim=-1)
    for x_ids,y_ids in zip(Input_id,x):
        #eng=id2String(tokenizer,y_ids)
        chin=id2String(tokenizer,x_ids).replace('[PAD]','').replace('[SEP]','')

    return chin

# 构建界面
demo = gr.Interface(
    fn=algorithm,
    inputs=gr.Textbox(lines=2, placeholder="输入文字 x"),
    outputs=gr.Textbox(label="输出文字 y"),
    title="翻译",
    description="请输入一段文字，算法将输出结果",
    examples=["你好世界", "Gradio", "Python"],
)

# 启动
demo.launch()
