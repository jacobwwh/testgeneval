# run test case generation in the symbolic execution setting
import json
import os
import argparse
import time
from tqdm import tqdm
from collections import OrderedDict

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


def get_safe_filename(model_name):
    """Convert model name to a safe filename by replacing '/' with '_'"""
    return model_name.replace('/', '_')


def get_response_with_retry(prompt, system_message, args, model_name, platform=None, max_retries=3):
    """Get response with retry mechanism"""
    for attempt in range(max_retries):
        try:
            print(f"🔄 Attempt {attempt + 1}/{max_retries}...")
            response = get_response(prompt, system_message, args, model_name, platform)
            print(f"✅ Success on attempt {attempt + 1}")
            return response
        except Exception as e:
            print(f"❌ Attempt {attempt + 1} failed: {type(e).__name__}: {str(e)}")
            if attempt < max_retries - 1:
                wait_time = attempt  # Exponential backoff: 1, 2, 4 seconds
                print(f"⏳ Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"💥 All {max_retries} attempts failed. Returning 'FAILED'")
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


if __name__ == '__main__':
    # Determine output file path
    safe_model_name = get_safe_filename(args.model)
    if args.setting == 'path':
        output_file_path = os.path.join(preds_dir, f'{safe_model_name}.jsonl')
    elif args.setting == 'baseline':
        output_file_path = os.path.join(preds_dir, f'{safe_model_name}_baseline.jsonl')
    
    # Load existing predictions for checkpoint/resume functionality
    existing_predictions = load_existing_predictions(output_file_path)
    
    # Track functions processed in this session
    functions_processed_this_session = 0
    functions_skipped_this_session = 0
    
    for i, data in enumerate(tqdm(dataset_with_paths)):
        if i < 7:
            continue
            
        instance_id = data['instance_id']
        instance_data = data['data']
        
        # get test set sample by instance_id
        test_set_sample = next((sample for sample in test_set if sample['instance_id'] == instance_id), None)
        assert test_set_sample is not None, f'Test set sample not found for instance_id: {instance_id}'
        
        # get file-level data
        code_src = test_set_sample['code_src']
        test_src = test_set_sample['preds_context']['last'] #follow the 'extra' setting in testgeneval
        
        print(f'🔄 Processing instance_id: {instance_id}')
        
        # start processing each function under test separately
        for func_name, func_data in instance_data.items():
            # Check if this function already has predictions
            if (instance_id, func_name) in existing_predictions:
                print(f'⏭️  Skipping {instance_id}/{func_name} (already processed)')
                functions_skipped_this_session += 1
                continue
                
            print(f'Generating test case for function: {func_name}')
            func_shortname = func_data['func']
            all_paths_txt = func_data['paths_txt'] #execution paths in text form

            if args.setting == 'path':
                prompt = prompt_template.create_prompt(code_src, test_src, func_name, with_paths=True, all_paths_txt=all_paths_txt)
            elif args.setting == 'baseline': #baseline: do not use execution paths
                prompt = prompt_template.create_prompt(code_src, test_src, func_name, with_paths=False)

            print(f"Prompt length: {len(prompt)} characters")
            print(f"Prompt preview: {prompt[:200]}{'...' if len(prompt) > 200 else ''}")
            
            generated_test_for_func = get_response_with_retry(prompt, system_message=prompt_template.SYSTEM_MESSAGE, args=args, model_name=args.model, platform=args.platform)

            if generated_test_for_func == "FAILED":
                print(f"❌ Failed to generate test for function: {func_name}")
            else:
                # Handle case where get_response returns a tuple (output, reasoning_content)
                if isinstance(generated_test_for_func, tuple):
                    output_content = generated_test_for_func[0]
                    reasoning_content = generated_test_for_func[1]
                    print(f"Generated test length: {len(output_content)} characters")
                    print(f"Generated test preview: {output_content[:200]}{'...' if len(output_content) > 200 else ''}")
                    if reasoning_content:
                        print(f"Reasoning content length: {len(reasoning_content)} characters")
                    else:
                        print("No reasoning content")
                else:
                    print(f"Generated test length: {len(generated_test_for_func)} characters")
                    print(f"Generated test preview: {generated_test_for_func[:200]}{'...' if len(generated_test_for_func) > 200 else ''}")

            # Save each function result as a separate line
            func_result = {
                'instance_id': instance_id,
                'func_name': func_name,
                'pred': generated_test_for_func
            }
            
            # Append to the output file
            with open(output_file_path, 'a') as f:
                f.write(json.dumps(func_result) + '\n')
            
            print(f'✅ Saved result for {instance_id}/{func_name}')
            functions_processed_this_session += 1
    
    print(f'\n📊 Summary:')
    print(f'   Skipped: {functions_skipped_this_session} functions (already processed)')
    print(f'   Processed: {functions_processed_this_session} functions')
    print(f'   Total: {functions_skipped_this_session + functions_processed_this_session} functions')
    print(f'   Results saved to: {output_file_path}')


