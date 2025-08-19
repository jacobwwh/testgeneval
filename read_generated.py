#New: read generated tests and results
import json

def read_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def read_jsonl(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]


preds_path = 'results/testgenevallite/preds/gpt-4.1-nano__testgenevallite__0.2__dev.jsonl'

preds = read_jsonl(preds_path)
print(len(preds))

for pred in preds:
    print(pred.keys())  #['id', 'instance_id', 'model_name_or_path', 'preds_prompts', 'preds']
    print(f'id: {pred["id"]}')
    print(f'instance_id: {pred["instance_id"]}')
    print(f'model_name_or_path: {pred["model_name_or_path"]}')
    print(pred['preds_prompts'].keys())  #['extra', 'first', 'full', 'last']
    #print('first', pred['preds_prompts']['first'])  #...write the Python code the next test
    #print('--------------------------------')
    #print('last', pred['preds_prompts']['last'])  #...write the Python code the next test
    #print(pred['preds_prompts']['full'])   #Generate the full test files. Given import example
    #print(pred['preds_prompts']['extra'])  #...write the Python code the next test
    print('--------------------------------')
    print(pred['preds'].keys())  #['extra', 'first', 'full', 'last']
    print(pred['preds']['extra'][0])
    print('--------------------------------')
    print(pred['preds']['extra'][1])
    #print('--------------------------------')
    #print(pred['preds']['first'], len(pred['preds']['first']))
    #print('--------------------------------')
    #print(pred['preds']['full'], len(pred['preds']['full']))
    #print('--------------------------------')
    #print(pred['preds']['last'], len(pred['preds']['last']))
    break