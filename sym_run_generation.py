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


if __name__ == '__main__':
    for i,data in enumerate(tqdm(dataset_with_paths)):
        if i<7:
            continue
        generated_results = OrderedDict()
        instance_id = data['instance_id']
        instance_data = data['data']
        #get test set sample by instance_id
        test_set_sample = next((sample for sample in test_set if sample['instance_id'] == instance_id), None)
        assert test_set_sample is not None, f'Test set sample not found for instance_id: {instance_id}'
        #print(test_set_sample['instance_id'])
        print(f'Processing instance_id: {instance_id}')

        #get file-level data
        code_src = test_set_sample['code_src']
        test_src = test_set_sample['preds_context']['last'] #follow the 'extra' setting in testgeneval
        
        #start processing each function under test separately
        for func_name, func_data in instance_data.items():
            print(f'Generating test case for function: {func_name}')
            #print(func_data.keys())
            func_shortname = func_data['func']
            all_paths_txt = func_data['paths_txt'] #execution paths in text form

            if args.setting == 'path':
                prompt = prompt_template.create_prompt(code_src, test_src, func_name, with_paths=True, all_paths_txt=all_paths_txt)
            elif args.setting == 'baseline': #baseline: do not use execution paths
                prompt = prompt_template.create_prompt(code_src, test_src, func_name, with_paths=False)

            #print(prompt)
            

            generated_test_for_func = get_response(prompt, system_message=prompt_template.SYSTEM_MESSAGE, args=args, model_name=args.model)

            #print(generated_test_for_func)

            generated_results[func_name] = generated_test_for_func
            #quit()

        #save generated results
        generated_results_for_file = {'instance_id': instance_id, 'preds': generated_results}
        if args.setting == 'path':
            with open(os.path.join(preds_dir, f'{args.model}.jsonl'), 'a') as f:
                f.write(json.dumps(generated_results_for_file) + '\n')
        elif args.setting == 'baseline':
            with open(os.path.join(preds_dir, f'{args.model}_baseline.jsonl'), 'a') as f:
                f.write(json.dumps(generated_results_for_file) + '\n')

        #quit()


