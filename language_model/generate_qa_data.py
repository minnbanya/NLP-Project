import json
import csv

# Path to your JSONL file
file_path = '../data/qa/test-qar_all.jsonl'
output_csv_path = '../data/qa/llm_test_1000.csv'

# Define a function to load data and extract question and answers, then save to CSV
def save_questions_and_answers_to_csv(input_file_path, output_file_path, nrows=1000):
    with open(input_file_path, 'r') as file, open(output_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write header to CSV file
        csvwriter.writerow(['Question', 'Answers'])

        for i, line in enumerate(file):
            if i >= nrows:
                break
            # Parse the JSON line
            entry = json.loads(line)
            question = entry.get('questionText', '')
            # Extract answers; assumes answers are in a list of dicts under 'answers' key
            answers = [ans['answerText'] for ans in entry.get('answers', []) if 'answerText' in ans]
            # Convert list of answers to a single string separated by '|'
            answers_str = '|'.join(answers)
            # Write to CSV
            csvwriter.writerow([question, answers_str])

# Call function to save data to CSV
save_questions_and_answers_to_csv(file_path, output_csv_path)

