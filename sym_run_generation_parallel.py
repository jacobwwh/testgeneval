# run test case generation in the symbolic execution setting (PARALLEL VERSION)
import json
import os
import argparse
import time
from tqdm import tqdm
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue

from sym_prompts import InstructPrompt
from sym_gen_utils import get_response

from datasets import load_dataset
dataset = load_dataset("kjain14/testgenevallite")
test_set = dataset["test"]

argparser = argparse.ArgumentParser()
argparser.add_argument('--setting', type=str, default='path', choices=['path', 'baseline']) #whether to use execution paths
argparser.add_argument('--model', type=str, default='gpt-4.1-nano')
argparser.add_argument('--platform', type=str, choices=['openai', 'openrouter', 'deepseek', 'anthropic'], default=None, 
                      help='Platform to use. If not specified, will be inferred from model name.')
argparser.add_argument('--temperature', type=float, default=0.0)
argparser.add_argument('--top_p', type=float, default=1.0)
argparser.add_argument('--reasoning_effort', type=str, choices=['minimal', 'low', 'medium', 'high'], default='medium')
argparser.add_argument('--num_workers', type=int, default=4, help='Number of parallel workers for processing')
args = argparser.parse_args()

dataset_path = 'sym_data/paths_data.jsonl'
dataset_with_paths = []
with open(dataset_path, 'r') as f:
    for line in f:
        data = json.loads(line)
        dataset_with_paths.append(data)

print(f'Dataset with {len(dataset_with_paths)} files')

prompt_template = InstructPrompt()

preds_dir = 'results/testgenevallite/preds'

# Thread-safe file writing
file_write_lock = threading.Lock()
processed_counter_lock = threading.Lock()
functions_processed_this_session = 0
functions_skipped_this_session = 0


def get_safe_filename(model_name):
    """Convert model name to a safe filename by replacing '/' with '_'"""
    return model_name.replace('/', '_')


def get_response_with_retry(prompt, system_message, args, model_name, platform=None, max_retries=3):
    """Get response with retry mechanism"""
    for attempt in range(max_retries):
        try:
            response = get_response(prompt, system_message, args, model_name, platform)
            return response
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = attempt  # Exponential backoff: 1, 2, 4 seconds
                time.sleep(wait_time)
            else:
                print(f"💥 Failed after {max_retries} attempts for function: {e}")
                return "FAILED"


def load_existing_predictions(output_file_path):
    """Load existing predictions from jsonl file to enable checkpoint/resume functionality"""
    existing_predictions = set()  # Now we track (instance_id, func_name) tuples
    if os.path.exists(output_file_path):
        print(f"Loading existing predictions from {output_file_path}")
        with open(output_file_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    instance_id = data.get('instance_id')
                    func_name = data.get('func_name')
                    pred = data.get('pred')
                    # Check if this function has a non-empty prediction
                    if instance_id and func_name and pred:
                        existing_predictions.add((instance_id, func_name))
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed JSON line in {output_file_path}")
                    continue
        print(f"Found {len(existing_predictions)} existing function predictions")
    return existing_predictions


def save_result(output_file_path, result):
    """Thread-safe function to save results to file"""
    with file_write_lock:
        with open(output_file_path, 'a') as f:
            f.write(json.dumps(result) + '\n')


def process_function(task_data):
    """Worker function to process a single function"""
    global functions_processed_this_session, functions_skipped_this_session
    
    instance_id = task_data['instance_id']
    func_name = task_data['func_name']
    func_data = task_data['func_data']
    code_src = task_data['code_src']
    test_src = task_data['test_src']
    output_file_path = task_data['output_file_path']
    existing_predictions = task_data['existing_predictions']
    
    # Check if this function already has predictions
    if (instance_id, func_name) in existing_predictions:
        with processed_counter_lock:
            functions_skipped_this_session += 1
        return {'status': 'skipped', 'instance_id': instance_id, 'func_name': func_name}
    
    try:
        func_shortname = func_data['func']
        all_paths_txt = func_data.get('paths_txt', '')  # execution paths in text form
        
        if args.setting == 'path':
            prompt = prompt_template.create_prompt(code_src, test_src, func_name, 
                                                  with_paths=True, all_paths_txt=all_paths_txt)
        elif args.setting == 'baseline':  # baseline: do not use execution paths
            prompt = prompt_template.create_prompt(code_src, test_src, func_name, with_paths=False)
        
        generated_test_for_func = get_response_with_retry(
            prompt, 
            system_message=prompt_template.SYSTEM_MESSAGE, 
            args=args, 
            model_name=args.model, 
            platform=args.platform
        )
        
        if generated_test_for_func == "FAILED":
            status = 'failed'
        else:
            status = 'success'
        
        # Save each function result
        func_result = {
            'instance_id': instance_id,
            'func_name': func_name,
            'pred': generated_test_for_func
        }
        
        # Save to file (thread-safe)
        save_result(output_file_path, func_result)
        
        with processed_counter_lock:
            functions_processed_this_session += 1
        
        return {
            'status': status,
            'instance_id': instance_id,
            'func_name': func_name,
            'result': func_result
        }
        
    except Exception as e:
        print(f"❌ Error processing {instance_id}/{func_name}: {e}")
        return {
            'status': 'error',
            'instance_id': instance_id,
            'func_name': func_name,
            'error': str(e)
        }


def prepare_tasks(dataset_with_paths, test_set, existing_predictions, output_file_path):
    """Prepare all tasks for parallel processing"""
    tasks = []
    
    for i, data in enumerate(dataset_with_paths):
        instance_id = data['instance_id']
        instance_data = data['data']
        
        # get test set sample by instance_id
        test_set_sample = next((sample for sample in test_set if sample['instance_id'] == instance_id), None)
        assert test_set_sample is not None, f'Test set sample not found for instance_id: {instance_id}'
        
        # get file-level data
        code_src = test_set_sample['code_src']
        test_src = test_set_sample['preds_context']['last']  # follow the 'extra' setting in testgeneval
        
        # Create tasks for each function under test
        for func_name, func_data in instance_data.items():
            task = {
                'instance_id': instance_id,
                'func_name': func_name,
                'func_data': func_data,
                'code_src': code_src,
                'test_src': test_src,
                'output_file_path': output_file_path,
                'existing_predictions': existing_predictions
            }
            tasks.append(task)
    
    return tasks


if __name__ == '__main__':
    # Determine output file path
    safe_model_name = get_safe_filename(args.model)
    if args.setting == 'path':
        output_file_path = os.path.join(preds_dir, f'{safe_model_name}.jsonl')
    elif args.setting == 'baseline':
        output_file_path = os.path.join(preds_dir, f'{safe_model_name}_baseline.jsonl')
    
    # Load existing predictions for checkpoint/resume functionality
    existing_predictions = load_existing_predictions(output_file_path)
    
    # Prepare all tasks
    print("📋 Preparing tasks...")
    all_tasks = prepare_tasks(dataset_with_paths, test_set, existing_predictions, output_file_path)
    
    # Filter out already processed tasks
    tasks_to_process = [task for task in all_tasks 
                        if (task['instance_id'], task['func_name']) not in existing_predictions]
    
    print(f"📊 Total tasks: {len(all_tasks)}")
    print(f"⏭️  Already processed: {len(all_tasks) - len(tasks_to_process)}")
    print(f"🔄 To process: {len(tasks_to_process)}")
    print(f"👷 Number of workers: {args.num_workers}")
    
    if tasks_to_process:
        # Process tasks in parallel
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            # Submit all tasks
            future_to_task = {executor.submit(process_function, task): task 
                             for task in tasks_to_process}
            
            # Process completed tasks with progress bar
            with tqdm(total=len(tasks_to_process), desc="Processing functions") as pbar:
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        result = future.result()
                        if result['status'] == 'success':
                            pbar.set_postfix_str(f"✅ {result['instance_id']}/{result['func_name']}")
                        elif result['status'] == 'failed':
                            pbar.set_postfix_str(f"❌ Failed: {result['instance_id']}/{result['func_name']}")
                        elif result['status'] == 'error':
                            pbar.set_postfix_str(f"💥 Error: {result['instance_id']}/{result['func_name']}")
                    except Exception as e:
                        pbar.set_postfix_str(f"💥 Exception: {task['instance_id']}/{task['func_name']}")
                        print(f"\n❌ Exception processing task: {e}")
                    pbar.update(1)
    
    print(f'\n📊 Final Summary:')
    print(f'   Skipped: {len(all_tasks) - len(tasks_to_process)} functions (already processed)')
    print(f'   Processed: {functions_processed_this_session} functions')
    print(f'   Total: {len(all_tasks)} functions')
    print(f'   Results saved to: {output_file_path}')