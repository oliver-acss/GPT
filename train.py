import torch
from torch import nn
from transformers import BertTokenizer

import dataHandle
from GPT import GPT
from Generator import Generator
from dataHandle import data_loader

from torch.utils.tensorboard import SummaryWriter


def main():
    file_path = r"./textData/train.txt"
    pre_model = None  # 设置为None表示不使用预训练模型
    print("数据加载完成")

    # 日志
    writer = SummaryWriter('runs/experiment_1')

    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
    loader, dataset = data_loader(file_path, 16, tokenizer, 128)

    model = GPT(768, tokenizer.vocab_size, 12, 3072, 128, 0.4)
    print("模型加载完成")

    # 加载预训练参数
    if pre_model is not None:
        try:
            state_dict = torch.load(pre_model)
            model.load_state_dict(state_dict)
            print(f"成功加载预训练模型: {pre_model}")
        except FileNotFoundError:
            print(f"预训练模型文件不存在: {pre_model}")
        except Exception as e:
            print(f"加载预训练模型时出错: {e}")

    optimizer1 = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_fn = nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    loss_fn.to(device)
    model.to(device)

    print("开始训练循环")
    for epoch in range(1, 100):  # 100
        total_loss = 0
        best_loss = 100000
        for data in loader:
            x, target = data
            x = x.to(device, dtype=torch.long)
            target = target.to(device, dtype=torch.long)
            output = model(x)
            loss = loss_fn(output.view(-1, tokenizer.vocab_size), target.view(-1))
            total_loss += loss.item()
            optimizer1.zero_grad()
            loss.backward()
            optimizer1.step()
            print(f"当前损失为{loss.item()}")

            writer.add_scalar('train_Loss', loss, epoch)

        print(f"第{epoch}轮训练已完成,总损失为{total_loss}")

        if total_loss < best_loss:
            best_loss = total_loss
            print(f"第{epoch}轮训练损失降低,保存模型")
            # 保存模型
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            print(f"第{epoch}轮训练损失未降低,不保存模型")
        torch.save(model.state_dict(), 'last_model.pt')

    writer.add_graph(model, x)  # 可视化网络图
    writer.close()  # 关闭会话


if __name__ == "__main__":
    main()