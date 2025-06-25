from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from requests.packages import target
from rouge_score import rouge_scorer

reference = "this is a test sentence"
candiate = "this is a test sentence"

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])

soures = scorer.score(target=reference,prediction=candiate)
print(f"ROUGE-1: {soures['rouge1'].fmeasure:.4f}")