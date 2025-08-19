#evaluate experiment results from coverage reports
import os
import json
import argparse
from tqdm import tqdm

from datasets import load_dataset
dataset = load_dataset("kjain14/testgenevallite")
test_set = dataset["test"]


argparser = argparse.ArgumentParser()
argparser.add_argument('--setting', type=str, default='path', choices=['path', 'baseline']) #whether to use execution paths
argparser.add_argument('--model', type=str, default='gpt-4.1-nano')
args = argparser.parse_args()


data_path = 'sym_data/paths_data.jsonl'
baseline_cov_dir = 'results/testgenevallite/data_logs/baseline'
baseline_cov_results_path = 'sym_data/baseline_cov.jsonl'  #write baseline cov results to this file


def load_jsonl_dataset(data_path):
    dataset = []
    with open(data_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            dataset.append(data)
    return dataset

def analyze_baseline():
    """
    Analyze the coverage results of the baseline test files.
    """
    dataset = load_jsonl_dataset(data_path)
    total_funcs = 0
    total_covered_lines = 0
    total_missing_lines = 0
    all_func_line_coverage = []  #line coverages for each function
    for data in tqdm(dataset):
        instance_id = data['instance_id']
        data_sample = data['data']
        task_instance = next((sample for sample in test_set if sample['instance_id'] == instance_id), None)
        assert task_instance is not None, f'Test set sample not found for instance_id: {instance_id}'
        test_sample_id = task_instance['id']  #the coverage file use id instead of instance_id
        
        #print(data_sample.keys())
        file_covdata = {}

        baseline_cov_file = os.path.join(baseline_cov_dir, f'{test_sample_id}.baseline.last.coverage.json')
        if os.path.exists(baseline_cov_file):
            with open(baseline_cov_file, 'r') as f:
                baseline_cov_data = json.load(f)
            all_func_covdata = baseline_cov_data['functions']
            #print(all_func_covdata.keys())
            for func_name in data_sample:
                if func_name in all_func_covdata:
                    func_covdata = all_func_covdata[func_name]
                    #print(func_covdata)
                    total_funcs += 1
                    total_covered_lines += func_covdata['summary']['covered_lines']
                    total_missing_lines += func_covdata['summary']['missing_lines']
                    all_func_line_coverage.append(func_covdata['summary']['percent_covered'])

                    file_covdata[func_name] = {
                        'covered_lines': func_covdata['executed_lines'],
                        'missing_lines': func_covdata['missing_lines'],
                        'num_covered_lines': func_covdata['summary']['covered_lines'],
                        'num_missing_lines': func_covdata['summary']['missing_lines'],
                        'num_total_lines': func_covdata['summary']['covered_lines'] + func_covdata['summary']['missing_lines'],
                        'percent_covered': func_covdata['summary']['percent_covered']
                    }

                else:
                    print(f"Function {func_name} not found in coverage data")
        else:
            print(f"Coverage file not found: {baseline_cov_file} for instance {instance_id}")
        
        
        with open(baseline_cov_results_path, 'a') as f:
            f.write(json.dumps({'instance_id': instance_id, 'id': test_sample_id, 'coverage_data': file_covdata}) + '\n')
        

    print(f'analyzed {total_funcs} functions, in {len(dataset)} files')
    print(f'total covered lines: {total_covered_lines}, total missing lines: {total_missing_lines}')
    print(f'total line coverage: {total_covered_lines / (total_covered_lines + total_missing_lines)}')
    print(f'average line coverage: {sum(all_func_line_coverage) / len(all_func_line_coverage)}%')


def analyze_preds(reports_dir, baseline_coverage_data):
    """analyze the coverage results of the LLM-generated test cases"""
    total_covered_lines = 0
    total_missing_lines = 0
    all_func_line_coverage = []  #line coverages for each function
    num_tests_passed = 0

    passed_funcs_covered_lines_base = 0
    passed_funcs_covered_lines_pred = 0
    passed_funcs_total_lines = 0
    for line in baseline_coverage_data:
        instance_id = line['instance_id']
        test_id = line['id']
        coverage_data = line['coverage_data']
        for func_name, func_covdata_baseline in coverage_data.items():
            cov_report_path = f'{reports_dir}/{test_id}.{args.model}.-{func_name}-.coverage.json'
            baseline_covered_lines = func_covdata_baseline['covered_lines']
            baseline_missing_lines = func_covdata_baseline['missing_lines']
            baseline_num_covered_lines = func_covdata_baseline['num_covered_lines']
            baseline_num_missing_lines = func_covdata_baseline['num_missing_lines']
            num_total_lines = func_covdata_baseline['num_total_lines']
            baseline_percent_covered = func_covdata_baseline['percent_covered']
            
            
            if os.path.exists(cov_report_path):
                num_tests_passed += 1
                print(f'Found coverage report for {func_name}')
                covdata_pred = json.load(open(cov_report_path, 'r'))
                func_covdata_pred = covdata_pred['functions'][func_name]
                #print(func_covdata_pred)
                #print(func_covdata_baseline)
                covered_lines_pred = func_covdata_pred['executed_lines']
                missing_lines_pred = func_covdata_pred['missing_lines']
                num_covered_lines_pred = func_covdata_pred['summary']['covered_lines']
                num_missing_lines_pred = func_covdata_pred['summary']['missing_lines']
                percent_covered_pred = func_covdata_pred['summary']['percent_covered']
                print(f'coverage changed from {baseline_percent_covered}% to {percent_covered_pred}%')
                total_covered_lines += num_covered_lines_pred
                total_missing_lines += num_missing_lines_pred
                all_func_line_coverage.append(percent_covered_pred)

                passed_funcs_covered_lines_base += baseline_num_covered_lines
                passed_funcs_covered_lines_pred += num_covered_lines_pred
                passed_funcs_total_lines += num_total_lines
            else: #no coverage report found (generated test case failed)
                total_covered_lines += baseline_num_covered_lines
                total_missing_lines += baseline_num_missing_lines
                all_func_line_coverage.append(baseline_percent_covered)
                
    print(f'num tests passed: {num_tests_passed}, pass rate: {num_tests_passed / 605}')
    print(f'total covered lines: {total_covered_lines}, total missing lines: {total_missing_lines}')
    print(f'total line coverage: {total_covered_lines / (total_covered_lines + total_missing_lines)}')
    print(f'average line coverage: {sum(all_func_line_coverage) / len(all_func_line_coverage)}%')

    print(f'Generated test cases covered {passed_funcs_covered_lines_pred} lines, {passed_funcs_covered_lines_pred / passed_funcs_total_lines * 100}% of total lines, in {passed_funcs_total_lines} lines')
    print(f'Previously covered {passed_funcs_covered_lines_base} lines, {passed_funcs_covered_lines_base / passed_funcs_total_lines * 100}% of total lines, in {passed_funcs_total_lines} lines')
        
        


if __name__ == "__main__":
    baseline_coverage_data = load_jsonl_dataset(baseline_cov_results_path)
    print(baseline_coverage_data[0])
    if args.setting == 'path':
        log_dir = f'results/testgenevallite/data_logs/{args.model}'
    elif args.setting == 'baseline':
        log_dir = f'results/testgenevallite/data_logs/{args.model}_baseline'
    os.chmod(log_dir, 0o777)

    analyze_preds(log_dir, baseline_coverage_data)
            