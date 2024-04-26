from models_utils import t5_model, gpt2_model
from utils import chat_answer
import pandas as pd
from rouge_score import rouge_scorer

# Path to the CSV file
csv_file_path = '../data/qa/llm_test_1000.csv'

# Load the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'])

llm = t5_model(temp = 0,rep = 1.5)

scores = []

for i in range(5):
    reference = df.iloc[i]['Answers']
    prompt_question = df.iloc[i]['Question']
    candidate_full = chat_answer(prompt_question, llm)
    candidate = candidate_full['answer']
    score = scorer.score(reference, candidate)
    scores.append(score)
    
def Average(list): 
    return sum(list) / len(list)

precisions = recall = fmeasure = []
for sco in scores:
    precisions.append(sco['rouge1'][0])
    recall.append(sco['rouge1'][1])
    fmeasure.append(sco['rouge1'][2])

pre_avg = Average(precisions)
recall_avg = Average(recall)
f_avg = Average(fmeasure)
overall_avg = (pre_avg + recall_avg + f_avg) / 3
print(f"ROUGE-1 scores:\nAverage Precision: {pre_avg}\nAverage Recall: {recall_avg}\nAverage fmeasure: {f_avg}\nAverage overall: {overall_avg}")