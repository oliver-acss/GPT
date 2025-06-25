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
    # 减小batch size以提高训练稳定性
    loader, dataset = data_loader(file_path, 8, tokenizer, 128)

    # 降低dropout以提高训练效果
    model = GPT(768, tokenizer.vocab_size, 12, 3072, 128, 0.1)
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

    # 降低学习率以提高训练稳定性
    optimizer1 = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=1e-5)
    # 忽略padding token以提高损失计算准确性
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    loss_fn.to(device)
    model.to(device)

    print("开始训练循环")
    best_loss = float('inf')
    patience = 10  # 早停耐心值
    patience_counter = 0
    
    for epoch in range(1, 150):  # 增加训练轮数
        model.train()
        total_loss = 0
        num_batches = 0
        
        for data in loader:
            x, target = data
            x = x.to(device, dtype=torch.long)
            target = target.to(device, dtype=torch.long)
            
            output = model(x)
            
            # 改进损失计算：使用shifted target
            target_shifted = target[:, 1:]  # 去掉第一个token
            output_shifted = output[:, :-1, :]  # 去掉最后一个输出
            
            loss = loss_fn(output_shifted.reshape(-1, tokenizer.vocab_size), target_shifted.reshape(-1))
            total_loss += loss.item()
            num_batches += 1
            
            optimizer1.zero_grad()
            loss.backward()
            
            # 添加梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer1.step()
            
            # 减少打印频率，每20个batch打印一次
            if num_batches % 20 == 0:
                print(f"Epoch {epoch}, Batch {num_batches}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / num_batches
        writer.add_scalar('train_Loss', avg_loss, epoch)
        
        print(f"第{epoch}轮训练已完成,平均损失为{avg_loss:.4f}")

        # 改进模型保存逻辑
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            print(f"第{epoch}轮训练损失降低,保存最佳模型")
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            patience_counter += 1
            print(f"第{epoch}轮训练损失未降低,耐心计数: {patience_counter}")
            
        # 每5轮保存一次最新模型
        if epoch % 5 == 0:
            torch.save(model.state_dict(), 'last_model.pt')
            
        # 早停机制
        if patience_counter >= patience:
            print(f"连续{patience}轮损失未降低，提前停止训练")
            break

    writer.add_graph(model, x)  # 可视化网络图
    writer.close()  # 关闭会话


if __name__ == "__main__":
    main()