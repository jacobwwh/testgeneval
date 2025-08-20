# run test case generation in the symbolic execution setting
import json
import os
import argparse
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


def load_existing_predictions(output_file_path):
    """Load existing predictions from jsonl file to enable checkpoint/resume functionality"""
    existing_predictions = {}
    if os.path.exists(output_file_path):
        print(f"Loading existing predictions from {output_file_path}")
        with open(output_file_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    instance_id = data.get('instance_id')
                    preds = data.get('preds', {})
                    # Check if preds is non-empty (has at least one function with non-empty predictions)
                    if instance_id and preds and any(preds.values()):
                        existing_predictions[instance_id] = preds
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed JSON line in {output_file_path}")
                    continue
        print(f"Found {len(existing_predictions)} existing predictions")
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
    
    # Count skipped and processed instances
    skipped_count = 0
    processed_count = 0
    
    for i, data in enumerate(tqdm(dataset_with_paths)):
        if i < 7:
            continue
            
        instance_id = data['instance_id']
        
        # Check if this instance already has predictions
        if instance_id in existing_predictions:
            print(f'⏭️  Skipping {instance_id} (already processed)')
            skipped_count += 1
            continue
            
        print(f'🔄 Processing instance_id: {instance_id}')
        processed_count += 1
        
        generated_results = OrderedDict()
        instance_data = data['data']
        
        # get test set sample by instance_id
        test_set_sample = next((sample for sample in test_set if sample['instance_id'] == instance_id), None)
        assert test_set_sample is not None, f'Test set sample not found for instance_id: {instance_id}'
        
        # get file-level data
        code_src = test_set_sample['code_src']
        test_src = test_set_sample['preds_context']['last'] #follow the 'extra' setting in testgeneval
        
        # start processing each function under test separately
        for func_name, func_data in instance_data.items():
            print(f'Generating test case for function: {func_name}')
            func_shortname = func_data['func']
            all_paths_txt = func_data['paths_txt'] #execution paths in text form

            if args.setting == 'path':
                prompt = prompt_template.create_prompt(code_src, test_src, func_name, with_paths=True, all_paths_txt=all_paths_txt)
            elif args.setting == 'baseline': #baseline: do not use execution paths
                prompt = prompt_template.create_prompt(code_src, test_src, func_name, with_paths=False)

            print(prompt)
            
            generated_test_for_func = get_response(prompt, system_message=prompt_template.SYSTEM_MESSAGE, args=args, model_name=args.model, platform=args.platform)

            print(generated_test_for_func)

            generated_results[func_name] = generated_test_for_func

        # save generated results
        generated_results_for_file = {'instance_id': instance_id, 'preds': generated_results}
        
        # Append to the output file
        with open(output_file_path, 'a') as f:
            f.write(json.dumps(generated_results_for_file) + '\n')
        
        print(f'✅ Saved results for {instance_id}')
    
    print(f'\n📊 Summary:')
    print(f'   Skipped: {skipped_count} instances (already processed)')
    print(f'   Processed: {processed_count} instances')
    print(f'   Total: {skipped_count + processed_count} instances')
    print(f'   Results saved to: {output_file_path}')


