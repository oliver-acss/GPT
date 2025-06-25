import torch
from transformers import BertTokenizer
from GPT import GPT


def id2String(tokenizer: BertTokenizer, input_ids):
    return tokenizer.decode(input_ids, skip_special_tokens=True)


def generate_translation(model, tokenizer, input_text, max_length=128, temperature=0.7):
    """
    生成翻译
    """
    model.eval()
    device = next(model.parameters()).device
    
    # 编码输入文本
    inputs = tokenizer(input_text, return_tensors='pt', max_length=max_length, 
                      truncation=True, padding=True)
    input_ids = inputs['input_ids'].to(device)
    
    with torch.no_grad():
        # 生成翻译
        generated_ids = input_ids.clone()
        
        for _ in range(max_length - input_ids.shape[1]):
            # 获取模型输出
            outputs = model(generated_ids)
            next_token_logits = outputs[:, -1, :] / temperature
            
            # 采样下一个token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # 如果生成了结束符，停止生成
            if next_token.item() == tokenizer.eos_token_id:
                break
                
            # 将新token添加到序列中
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
    
    # 解码生成的文本
    generated_text = id2String(tokenizer, generated_ids[0])
    return generated_text


def main():
    # 加载模型和tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
    model = GPT(768, tokenizer.vocab_size, 12, 3072, 128, 0.1)
    
    # 加载训练好的模型权重
    try:
        model.load_state_dict(torch.load("./best_model.pt", weights_only=True))
        print("成功加载最佳模型")
    except:
        try:
            model.load_state_dict(torch.load("./last_model.pt", weights_only=True))
            print("成功加载最新模型")
        except:
            print("无法加载模型权重，请先训练模型")
            return
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 测试句子
    test_sentences = [
        "duravit combines the global resources, but all factories follow the same high - standard, consummate craft advanced technology, which.",
        "anyway, your pre - existing condition won't be covered under most corporate plans for at least a year.",
        "if you hadn t lent me some money, i couldn t have bought the new house and most likely i would be still living in the dangerous house now.",
        "has miss lin ever talked such disgusting nonsense?"
    ]
    
    print("=" * 60)
    print("模型翻译测试结果")
    print("=" * 60)
    
    for i, sentence in enumerate(test_sentences, 1):
        print(f"\n测试 {i}:")
        print(f"输入: {sentence}")
        
        try:
            translation = generate_translation(model, tokenizer, sentence)
            print(f"输出: {translation}")
        except Exception as e:
            print(f"生成失败: {e}")
        
        print("-" * 40)


if __name__ == "__main__":
    main() 