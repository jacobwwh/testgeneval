#parse the coverage logs of the baseline, find functions that have branches not covered.

import json
import os
from pathlib import Path

#read the testgenevallite dataset
from datasets import load_dataset
dataset = load_dataset("kjain14/testgenevallite")
test_set = dataset["test"]
test_set_instances = set(test_set['instance_id'])
print(f'number of test set instances: {len(test_set_instances)}')
'''for data_sample in test_set:
    #print(data_sample.keys())
    print(data_sample['instance_id'])
    print(data_sample['code_file'], data_sample['test_file'])
    with open(os.path.join('test_files', f'{data_sample["instance_id"]}.py'), 'w') as f:
        f.write(data_sample['code_src'])'''
#quit()



def find_log_files(baseline_dir="results/testgenevallite/data_logs/baseline", setting="last"):
    """
    Traverse the baseline directory and find all .log files.
    
    Args:
        baseline_dir (str): Path to the baseline directory
        
    Returns:
        list: List of paths to all .log files found
    """
    log_files = []
    
    # Convert to Path object for easier handling
    baseline_path = Path(baseline_dir)
    
    # Check if directory exists
    if not baseline_path.exists():
        print(f"Directory {baseline_dir} does not exist!")
        return log_files
    
    if not baseline_path.is_dir():
        print(f"{baseline_dir} is not a directory!")
        return log_files
    
    # Walk through the directory
    for file_path in baseline_path.iterdir():
        if file_path.is_file() and str(file_path).endswith(f'.{setting}.eval.log'):
            log_files.append(str(file_path))
    
    return log_files


"""Main function to demonstrate finding log files"""
log_files = find_log_files(setting="last")
print(f'Found {len(log_files)} .log files:')

all_branchnotcovered_files = []
all_branchnotcovered_files_instance_ids = []
num_branchnotcovered_funcs = 0
all_branchnotcovered_metadata = {}
for log_file in log_files:
    log_instance_id = log_file.split('/')[-1].split('.')[0].split('-')[:-1]
    log_instance_id = '-'.join(log_instance_id)
    #print(log_instance_id)
    #quit()
    if log_instance_id not in test_set_instances:
        continue
    with open(log_file, 'r') as f:
        log_lines = f.readlines()
        for line in log_lines:
            if 'Find function with branch not completely covered!' in line:
                all_branchnotcovered_files.append(log_file)
                all_branchnotcovered_files_instance_ids.append(log_instance_id)

                #find the coverage.json file
                #print(log_file)
                coverage_file_path = log_file.replace('.eval.log', '.coverage.json')
                #print(coverage_file_path)
                # Check if coverage file exists
                if not os.path.exists(coverage_file_path):
                    print(f"Coverage file does not exist: {coverage_file_path}")
                    quit()
                print(f'instance_id: {log_instance_id}')
                #parse the coverage.json file
                with open(coverage_file_path, 'r') as f:
                    coverage_data = json.load(f)
                    funcs_coverage_data = coverage_data['functions']
                    #print(funcs_coverage_data.keys())
                    partially_covered_funcdata = {}
                    partially_covered_func_metadata = {}  #only contains instance id and func name
                    for func_fullname, func_coverage_data in funcs_coverage_data.items():
                        func_names = func_fullname.split('.')
                        func_name = func_names[-1]
                        func_class = func_names[0] if len(func_names) > 1 else ''
                        func_executed_lines = func_coverage_data['executed_lines']
                        func_missing_lines = func_coverage_data['missing_lines']
                        if len(func_missing_lines) > 0 and len(func_executed_lines) > 0: #with branches not covered
                            func_all_lines = sorted(func_executed_lines + func_missing_lines)
                            func_start_line = func_all_lines[0]
                            func_end_line = func_all_lines[-1]
                            #print(func_all_lines)
                            print(f'func not fully covered: {func_fullname}')
                            num_branchnotcovered_funcs += 1
                            partially_covered_funcdata[func_fullname] = {
                                'func_name': func_name,
                                'class': func_class,
                                'executed_lines': func_executed_lines,
                                'missing_lines': func_missing_lines,
                                'all_lines': func_all_lines
                            }
                            partially_covered_func_metadata[func_fullname] = {
                                'func_name': func_name,
                                'class': func_class,
                                'start_line': func_start_line,
                                'end_line': func_end_line
                            }
                    with open(os.path.join('func_partial_covdata', f'{log_instance_id}.json'), 'w') as f:
                        json.dump(partially_covered_funcdata, f, indent=4)
                            #quit()
                    #quit()
                    all_branchnotcovered_metadata[log_instance_id] = partially_covered_func_metadata
                break
        '''for i, line in enumerate(log_lines):
            if 'Find function with branch not completely covered!' in line:
                func_cover_details = log_lines[i+2]
                print(func_cover_details)
                quit()'''

with open(os.path.join('sym_data', 'metadata.json'), 'w') as f: #dump the metadata of all partially covered functions
    json.dump(all_branchnotcovered_metadata, f, indent=4)

print(f'{len(all_branchnotcovered_files)} files with not covered branches')
#for log_file in all_branchnotcovered_files:
    #print(log_file)
print(f'number of functions not fully covered: {num_branchnotcovered_funcs}')


