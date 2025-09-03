#run binary classification: valid/invalid or valid/zerodivision
import openai
from openai import OpenAI
import json
import jinja2
import os
import argparse
from tqdm import tqdm

openai_api_key = os.getenv('OPENAI_API_KEY')
deepseek_api_key = os.getenv('DEEPSEEK_KEY')
client = OpenAI(api_key=openai_api_key)
deepseek_client = OpenAI(api_key=deepseek_api_key, base_url='https://api.deepseek.com')




# Set up Jinja2 environment to load templates from prompt_templates/ directory
template_dir = 'prompt_templates'
jinja_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(template_dir),
    autoescape=jinja2.select_autoescape(['html', 'xml'])
)

# Load the bugdetect template
#bugdetect_template = jinja_env.get_template('bugdetect.jinja2')
print('using new prompt template')
bugdetect_template_12 = jinja_env.get_template('bugdetect_12.jinja2')  #valid/invalid
bugdetect_template_13 = jinja_env.get_template('bugdetect_13.jinja2')  #valid/zerodivision


# Example usage:
# prompt = render_bugdetect_prompt("your execution path here")
# print(prompt)

bugdetect_dataset_path = 'bugdetect-data/bug_paths_dataset.jsonl'
bugdetect_dataset = []
with open(bugdetect_dataset_path, 'r') as f:
    for line in f:
        bugdetect_dataset.append(json.loads(line))

argparser = argparse.ArgumentParser()
argparser.add_argument('--model', type=str, default='gpt-4.1-nano')
argparser.add_argument('--temperature', type=float, default=0.0)
argparser.add_argument('--top_p', type=float, default=1.0)
argparser.add_argument('--reasoning_effort', type=str, choices=['minimal', 'low', 'medium', 'high'], default='medium')
argparser.add_argument('--task', type=str, choices=['12', '13'], default='12')
#argparser.add_argument('--reasoning_summary', type=str, choices=['auto', 'concise', 'detailed'], default='detailed')
args = argparser.parse_args()


def get_response(prompt, model_name='gpt-4.1-nano'):
    if model_name == 'deepseek-reasoner':
        response = deepseek_client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[{"role": "user", "content": prompt}]
        )

        reasoning_content = response.choices[0].message.reasoning_content
        output = response.choices[0].message.content
        return output, reasoning_content
    if model_name in ['deepseek-chat']:
        response = deepseek_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=args.temperature,
        )
        output = response.choices[0].message.content
        return output
    elif model_name in ['o3', 'o3-mini', 'o4-mini']:
        response = client.responses.create(
            model=model_name,
            input=prompt,
            instructions="",  #system message?
            reasoning={"effort": args.reasoning_effort, "summary": None}
        )
        output = response.output_text
        return output
    else:
        response = client.responses.create(
            model=model_name,
            input=prompt,
            instructions="",  #system message?
            temperature=args.temperature,
            top_p=args.top_p
        )

        #return response.output[0].content[0].text
        return response.output_text


if __name__ == '__main__':
    print(f'Bug detection dataset: {len(bugdetect_dataset)}')
    res_dir = 'bugdetect-data/res'
    os.makedirs(res_dir, exist_ok=True)
    if args.task == '12':
        print('classification task: valid/invalid')
        bugdetect_template = bugdetect_template_12
    elif args.task == '13':
        print('classification task: valid/zerodivision')
        bugdetect_template = bugdetect_template_13

    res = []
    res_file = os.path.join(res_dir, f'{args.model}_{args.task}.jsonl')

    for i, data in tqdm(enumerate(bugdetect_dataset)):
        if i < 96:
            continue
        data_src = data['code']
        execution_paths = data['paths']
        print(data_src)
        #print(f'{len(execution_paths)} execution paths')
        labels = data['label']
        #print(labels)
        if args.task == '12':
            task_indices = [i for i, label in enumerate(labels) if label in [1, 2]]
        elif args.task == '13':
            task_indices = [i for i, label in enumerate(labels) if label in [1, 3]]
        #print(f'task indices: {task_indices}')
        labels = [labels[i] for i in task_indices]
        execution_paths = [execution_paths[i] for i in task_indices]  #filter exexcution paths by labels
        assert len(labels) == len(execution_paths)
        print(f'labels for binary classification: {labels}')
        print(f'{len(labels)} execution paths')
        
        code_res = []
        reasoning_cot = []
        for exec_path in execution_paths:
            #print(exec_path)
            bugdetect_prompt = bugdetect_template.render(execution_path_txt=exec_path)
            #print(bugdetect_prompt)
            if args.model == 'deepseek-reasoner':
                response, reasoning_content = get_response(bugdetect_prompt, args.model)
                reasoning_cot.append(reasoning_content)
            else:
                response = get_response(bugdetect_prompt, args.model)
            print(response)
            code_res.append(response)
        print('-'*10)
        if args.model == 'deepseek-reasoner':
            res_problem = {
                'problem_id': data['problem_id'],
                'submission_id': data['submission_id'],
                'prediction': code_res,
                'reasoning_cot': reasoning_cot
            }
        else:
            res_problem = {
                'problem_id': data['problem_id'],
                'submission_id': data['submission_id'],
                'prediction': code_res
            }
        res.append(res_problem)
        with open(res_file, 'a') as res_f:
            res_f.write(json.dumps(res_problem) + '\n')
