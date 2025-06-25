import torch
from transformers import BertTokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from GPT import GPT
from dataHandle import data_loader


def id2String(tokenizer: BertTokenizer, input_ids):
    return tokenizer.decode(input_ids, skip_special_tokens=True)


def calculate_bleu(reference, candidate):
    reference = [reference.split()]
    candidate = candidate.split()
    score = sentence_bleu(reference, candidate, smoothing_function=SmoothingFunction().method4)
    return score


def calculate_rouge(reference, candidate):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
    scores = scorer.score(reference, candidate)
    return scores


def main():
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
    model = GPT(768, tokenizer.vocab_size, 12, 3072, 128, 0.4)

    # Load model weights securely
    model.load_state_dict(torch.load("./last_model.pt", weights_only=True))
    model.eval()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    file_path = r"./textData/test.txt"
    loader, dataset = data_loader(file_path, 16, tokenizer, 128)

    with open('output_results.txt', 'w', encoding='utf-8') as output_file:
        for data in loader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            input_ids = output.argmax(dim=-1)

            for x_ids, y_ids in zip(input_ids, x):
                eng = id2String(tokenizer, y_ids).strip()
                chin = id2String(tokenizer, x_ids).strip()

                # Calculate BLEU score
                bleu_score = calculate_bleu(eng, chin)

                # Calculate ROUGE scores
                rouge_scores = calculate_rouge(eng, chin)

                print(f"Input: {eng}")
                print(f"Output: {chin}")
                print(f"BLEU Score: {bleu_score:.4f}")
                print(f"ROUGE-1: {rouge_scores['rouge1'].fmeasure:.4f}")
                print(f"ROUGE-2: {rouge_scores['rouge2'].fmeasure:.4f}")
                print(f"ROUGE-L: {rouge_scores['rougeL'].fmeasure:.4f}")

                output_file.write(f"Input: {eng}\n")
                output_file.write(f"Output: {chin}\n")
                output_file.write(f"BLEU Score: {bleu_score:.4f}\n")
                output_file.write(f"ROUGE-1: {rouge_scores['rouge1'].fmeasure:.4f}\n")
                output_file.write(f"ROUGE-2: {rouge_scores['rouge2'].fmeasure:.4f}\n")
                output_file.write(f"ROUGE-L: {rouge_scores['rougeL'].fmeasure:.4f}\n\n")


if __name__ == '__main__':
    main()
