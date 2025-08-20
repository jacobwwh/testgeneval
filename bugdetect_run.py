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
bugdetect_template = jinja_env.get_template('bugdetect.jinja2')


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
argparser.add_argument('--platform', type=str, choices=['openai', 'openrouter', 'deepseek', 'anthropic'], default=None, 
                      help='Platform to use. If not specified, will be inferred from model name.')
argparser.add_argument('--temperature', type=float, default=0.0)
argparser.add_argument('--top_p', type=float, default=1.0)
argparser.add_argument('--reasoning_effort', type=str, choices=['minimal', 'low', 'medium', 'high'], default='medium')
#argparser.add_argument('--reasoning_summary', type=str, choices=['auto', 'concise', 'detailed'], default='detailed')
args = argparser.parse_args()


def get_response(prompt, model_name='gpt-4.1-nano', platform=None):
    # Determine platform from model_name if not explicitly provided
    if platform is None:
        if model_name.startswith('openrouter/'):
            platform = 'openrouter'
        elif model_name in ['deepseek-reasoner', 'deepseek-chat']:
            platform = 'deepseek'
        elif model_name in ['o3', 'o3-mini', 'o4-mini']:
            platform = 'anthropic'
        else:
            platform = 'openai'
    
    # Handle different platforms
    if platform == 'openrouter':
        actual_model = model_name.replace('openrouter/', '') if model_name.startswith('openrouter/') else model_name
        response = openrouter_client.chat.completions.create(
            model=actual_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=getattr(args, 'temperature', 0.7),
            top_p=getattr(args, 'top_p', 1.0)
        )
        output = response.choices[0].message.content
        return output
    elif platform == 'deepseek':
        if model_name == 'deepseek-reasoner':
            response = deepseek_client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[{"role": "user", "content": prompt}]
            )
            reasoning_content = response.choices[0].message.reasoning_content
            output = response.choices[0].message.content
            return output, reasoning_content
        else:  # deepseek-chat
            response = deepseek_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=args.temperature,
            )
            output = response.choices[0].message.content
            return output
    elif platform == 'anthropic':
        response = client.responses.create(
            model=model_name,
            input=prompt,
            instructions="",  #system message?
            reasoning={"effort": args.reasoning_effort, "summary": None}
        )
        output = response.output_text
        return output
    else:  # openai platform
        response = client.responses.create(
            model=model_name,
            input=prompt,
            instructions="",  #system message?
            temperature=args.temperature,
            top_p=args.top_p
        )
        return response.output_text


def get_safe_filename(model_name):
    """Convert model name to a safe filename by replacing '/' with '_'"""
    return model_name.replace('/', '_')


if __name__ == '__main__':
    print(f'Bug detection dataset: {len(bugdetect_dataset)}')
    res_dir = 'bugdetect-data/res'
    os.makedirs(res_dir, exist_ok=True)

    res = []
    safe_model_name = get_safe_filename(args.model)
    res_file = os.path.join(res_dir, f'{safe_model_name}.jsonl')

    for i, data in tqdm(enumerate(bugdetect_dataset)):
        #if i<78:
            #continue
        data_src = data['code']
        execution_paths = data['paths']
        print(data_src)
        print(f'{len(execution_paths)} execution paths')
        code_res = []
        reasoning_cot = []
        for exec_path in execution_paths:
            #print(exec_path)
            bugdetect_prompt = bugdetect_template.render(execution_path_txt=exec_path)
            #print(bugdetect_prompt)
            if args.model == 'deepseek-reasoner':
                response, reasoning_content = get_response(bugdetect_prompt, args.model, args.platform)
                reasoning_cot.append(reasoning_content)
            else:
                response = get_response(bugdetect_prompt, args.model, args.platform)
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

#with open(res_file, 'w') as f:
    #for data in res:
        #f.write(json.dumps(data) + '\n')
