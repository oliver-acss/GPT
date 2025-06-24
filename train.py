import torch
from torch import nn
from transformers import BertTokenizer

import dataHandle
from GPT import GPT
from Generator import Generator
from dataHandle import data_loader


def main():
    file_path=r"./textData/train.txt"
    tokenizer=BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
    loader,dataset=data_loader(file_path,16,tokenizer,128)
    print("数据加载完成")
    model=GPT(768,tokenizer.vocab_size,1,3072,128,0.4)
    print("模型加载完成")
    optimizer1=torch.optim.Adam(model.parameters(),lr=0.0001)
    loss_fn=nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_fn.to(device)
    model.to(device)
    print("开始训练循环")
    for epoch in range(1,100):
        total_loss=0
        best_loss=100000
        for data in loader:
            x,target=data
            x=x.to(device,dtype=torch.long)
            target=target.to(device,dtype=torch.long)
            output=model(x)
            loss=loss_fn(output.view(-1,tokenizer.vocab_size),target.view(-1))
            total_loss+=loss.item()
            optimizer1.zero_grad()
            loss.backward()
            optimizer1.step()
            print(f"当前损失为{loss.item()}")
        print(f"第{epoch}轮训练已完成,总损失为{total_loss}")
        torch.save(model.state_dict(), 'last_model.pt')
if __name__=="__main__":
    main()