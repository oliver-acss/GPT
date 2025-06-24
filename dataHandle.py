import json

from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer


def read_file(path):
    """数据文件格式为txt,每一行为一个json文件, 按行读取并添加至data"""
    data = []
    try:
        # 打开文件
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                # 去除可能的空行或额外字符
                clean_line = line.strip()
                if clean_line:
                    try:
                        # 将字符串转换为字典
                        data_dict = json.loads(clean_line)
                        data.append(data_dict)
                    except json.JSONDecodeError as e:
                        print(f"JSON解析错误: {e}")
    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
    except IOError as e:
        print(f"文件读取错误: {e}")
    return data


class TranslationDataset(Dataset):
    """自定义数据集"""

    def __init__(self, data, tokenizer, max_length):
        self.data = data  # 数据
        self.English = [item['english'].lower() for item in data]  # 将英文文本添加到 self.English 列表，编码需要小写化
        self.Chinese = [item['chinese'] for item in data]  # 将中文文本添加到 self.Chinese 列表
        self.tokenizer = tokenizer  # token化工具
        self.max_length = max_length  # 最大序列长度

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        English_encoded = self.tokenizer(self.English[idx], max_length=self.max_length, truncation=True,
                                         padding='max_length', return_tensors='pt')
        Chinese_encoded = self.tokenizer(self.Chinese[idx], max_length=self.max_length, truncation=True,
                                         padding='max_length', return_tensors='pt')
        return English_encoded['input_ids'].reshape(-1), Chinese_encoded['input_ids'].reshape(-1)
def data_loader(data_path,batch_size,tokenizer,max_length):
    # 从路径读取数据
    data = read_file(data_path)
    # 创建测试数据集
    dataset = TranslationDataset(data, tokenizer, max_length)
    # 创建数据加载器
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader, dataset

def capitalize_first_letter_of_sentences(text):
    # 使用split方法将文本分割成句子列表，这里假设句子以点号、问号或感叹号结尾
    sentences = text.split('. ')
    sentences = [sentence.strip() for sentence in sentences if sentence.strip() != '']

    # 对每个句子进行处理，使其第一个单词的首字母大写
    capitalized_sentences = [sentence[0].upper() + sentence[1:] if sentence else '' for sentence in sentences]

    # 将处理后的句子列表重新组合成一个文本
    capitalized_text = '. '.join(capitalized_sentences) + ('.' if text.endswith('.') else '')

    return capitalized_text
