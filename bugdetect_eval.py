import os
import json
import argparse
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report


argparser = argparse.ArgumentParser()
argparser.add_argument('--model', type=str, default='gpt-4.1-nano')
args = argparser.parse_args()

#label: 1--satisfied; 2--unsatisfied;3--bug-triggered
def count_dataset_labels(labels):
    total_count = 0
    count_sat = 0
    count_unsat = 0
    count_bug = 0
    for problem_labels in labels:
        for label in problem_labels:
            total_count += 1
            if label == 1:
                count_sat += 1
            elif label == 2:
                count_unsat += 1
            elif label == 3:
                count_bug += 1
    print(f'total paths: {total_count}')
    print(f'satisfied: {count_sat}')
    print(f'unsatisfied: {count_unsat}')
    print(f'bug-triggered: {count_bug}')


def count_paths(paths):
    total_count = 0
    total_code_block_count = 0
    for problem_paths in paths:
        for path in problem_paths:
            total_count += 1
            # Count path length by occurrences of 'Code block' in the current path
            code_block_count = path.count('Code block')
            total_code_block_count += code_block_count
    avg_path_length_blocks = total_code_block_count / total_count
    print(f'avg path length: {avg_path_length_blocks}')


if __name__ == '__main__':
    dataset_path = 'bugdetect-data/bug_paths_dataset.jsonl'
    #load labels
    labels = []
    paths = []
    with open(dataset_path, 'r') as f:
        for line in f.readlines():
            data = json.loads(line)
            labels.append(data['label'])
            paths.append(data['paths'])
    count_dataset_labels(labels)
    count_paths(paths)

    res_dir = 'bugdetect-data/res'
    res_file = os.path.join(res_dir, f'{args.model}.jsonl')
    predicted_labels = []
    with open(res_file, 'r') as f:
        for line in f.readlines():
            data = json.loads(line)
            problem_prdicted_labels = []
            for prediction in data['prediction']:
                if 'VALID' in prediction and 'INVALID' not in prediction:
                    problem_prdicted_labels.append(1)
                elif 'INVALID' in prediction:
                    problem_prdicted_labels.append(2)
                elif 'ZERODIVISION' in prediction:
                    problem_prdicted_labels.append(3)
                else:
                    problem_prdicted_labels.append(0)
            predicted_labels.append(problem_prdicted_labels)
    
    #compute metrics
    # Flatten labels and predicted_labels into 1-dimensional lists (one-liner approach)
    flat_labels = [label for problem_labels in labels for label in problem_labels]
    flat_predicted_labels = [label for problem_predicted_labels in predicted_labels for label in problem_predicted_labels]
    
    #compute acc, precision, recall, f1
    
    # Compute overall accuracy
    accuracy = accuracy_score(flat_labels, flat_predicted_labels)
    
    # Compute precision, recall, F1 for class '1' specifically
    p_valid, r_valid, f1_valid, sup_valid = precision_recall_fscore_support(
        flat_labels, flat_predicted_labels, labels=[1], average=None
    )
    
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Class '1' Precision: {p_valid[0]:.4f}")
    print(f"Class '1' Recall: {r_valid[0]:.4f}")
    print(f"Class '1' F1-score: {f1_valid[0]:.4f}")
    print(f"Class '1' Support: {sup_valid[0]}")

    # Compute precision, recall, F1 for class '2' specifically
    p_invalid, r_invalid, f1_invalid, sup_invalid = precision_recall_fscore_support(
        flat_labels, flat_predicted_labels, labels=[2], average=None
    )
    print(f"Class '2' Precision: {p_invalid[0]:.4f}")
    print(f"Class '2' Recall: {r_invalid[0]:.4f}")
    print(f"Class '2' F1-score: {f1_invalid[0]:.4f}")
    print(f"Class '2' Support: {sup_invalid[0]}")

    # Compute precision, recall, F1 for class '3' specifically
    p_bug, r_bug, f1_bug, sup_bug = precision_recall_fscore_support(
        flat_labels, flat_predicted_labels, labels=[3], average=None
    )
    print(f"Class '3' Precision: {p_bug[0]:.4f}")
    print(f"Class '3' Recall: {r_bug[0]:.4f}")
    print(f"Class '3' F1-score: {f1_bug[0]:.4f}")
    print(f"Class '3' Support: {sup_bug[0]}")
