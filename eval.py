import torch
from transformers import BertTokenizer

from GPT import GPT
from dataHandle import data_loader

def id2String(Tokenizer:BertTokenizer,Input_id):
    result=""
    for Id in Input_id:
        result=result+' '+Tokenizer.decode(Id)
    return result

def main():
    tokenizer=BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
    model=GPT(768,tokenizer.vocab_size,12,3072,128,0.4)
    model.load_state_dict(torch.load("./last_model.pt"))
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    file_path = r"./textData/test.txt"
    loader, dataset = data_loader(file_path, 16, tokenizer, 128)
    for data in loader:
        x,y = data
        x = x.to(device)
        y = y.to(device)
        output = model(x)
        Input_id=output.argmax(dim=-1)
        for x_ids,y_ids in zip(Input_id,x):
            print(id2String(tokenizer,y_ids))
            print(id2String(tokenizer,x_ids))
if __name__ == '__main__':
    main()

