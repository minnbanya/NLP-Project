import pandas as pd
from rouge_score import rouge_scorer
from models_utils import load_model
from utils import chat_answer
import mlflow

# Configure MLflow
mlflow.set_experiment("NLP Model Comparison")

def calculate_rouge_scores(references, candidates):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'])
    scores = [scorer.score(ref, cand) for ref, cand in zip(references, candidates)]
    return scores

def run_experiment(model_name, temp, rep, data, num_samples=5):
    # Load model
    llm = load_model(model_name, temp=temp, rep=rep)
    
    # Select a subset of data
    sampled_data = data.sample(n=num_samples)
    
    # Generate answers and compute scores
    references = sampled_data['Answers']
    candidates = [chat_answer(row['Question'], llm)['answer'] for _, row in sampled_data.iterrows()]
    scores = calculate_rouge_scores(references, candidates)
    
    return scores

def log_scores(scores):
    for score_dict in scores:
        for key, (precision, recall, fmeasure) in score_dict.items():
            mlflow.log_metric(f"{key}_precision", precision)
            mlflow.log_metric(f"{key}_recall", recall)
            mlflow.log_metric(f"{key}_fmeasure", fmeasure)

def main():
    # Path to the CSV file
    csv_file_path = '../data/qa/llm_test_1000.csv'
    
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)
    
    # Define models and parameters to test
    models_params = [
        # ("t5", 0, 1.5),
        # ("t5", 0, 1),
        # ("gpt2", 0, 1.5),
        # ("gpt2", 0.5, 1.2),
        ("t5", 0.5, 1.5)
    ]
    
    # Run experiments
    for model_name, temp, rep in models_params:
        with mlflow.start_run(run_name=f"{model_name}_{temp}_{rep}"):
            mlflow.log_params({"model": model_name, "temperature": temp, "repetition_penalty": rep})
            scores = run_experiment(model_name, temp, rep, df)
            log_scores(scores)
            mlflow.end_run()

if __name__ == "__main__":
    main()
