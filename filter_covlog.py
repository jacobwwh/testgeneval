#filter the partially-covered function coverage log by parsing the functions with scalpel cfg
import json
import os
from myscalpel.builder import get_signature
from cfg_util import path_find_from_cfg

def gen_prompt_with_path(template_obj, data_obj, no_import=False, tokenizer=None):  
    """Modified from InstructPrompt.add_prompts_to_dataset()"""

    code_src = data_obj["code_src"]

    #extract paths
    all_paths = path_find_from_cfg(code_src)
    #print(all_paths)
    print(f'Number of functions in {data_obj["code_file"]}: {len(all_paths)}')
    for func_data in all_paths:
        print(f'number of paths in function {func_data["signature"]} in class {func_data["class"]}: {len(func_data["pathdata"])}')
    print('--------------------------------')


def find_pathdata_byname(all_paths, func_covdata, func_fullname):
    func_name = func_covdata['func_name']
    func_class = func_covdata['class']
    for func_pathdata in all_paths:
        if func_pathdata["func_fullname"] == func_fullname:
            return func_pathdata
        '''if func_pathdata["func"] == func_name:
            if func_pathdata["class"] == func_class or func_class == '':
                if func_pathdata['start_line'] <= func_covdata['all_lines'][0] and func_pathdata['end_line'] >= func_covdata['all_lines'][-1]:  #make sure that the line numbers match
                    return func_pathdata'''
    
    print(f'no CFG paths data found for {func_fullname}')
    return None


cov_data_dir = 'func_partial_covdata'
src_file_dir = 'test_files'
for file in os.listdir(cov_data_dir):
    if file.endswith('.json'):
        instance_id = file.split('.')[0]
        src_file = os.path.join(src_file_dir, f'{instance_id}.py')
        with open(os.path.join(cov_data_dir, file), 'r') as f:
            with open(src_file, 'r') as src_f:
                print(f'Reading coverage data for {instance_id}')
                cov_data = json.load(f)
                src_code = src_f.read()
                all_paths = path_find_from_cfg(src_code)

                for func_fullname, func_covdata in cov_data.items():
                    func_name = func_covdata['func_name']
                    func_class = func_covdata['class']

                    #find the func_pathdata with the same class and func name
                    path_data = find_pathdata_byname(all_paths, func_covdata, func_fullname)
                    #print(f'Found CFG paths data for {func_fullname}')
                    #print(path_data['start_line'], path_data['end_line'])
                    #print(func_covdata['all_lines'])
                    #print(len(path_data['pathdata']))
                    #quit()
                    if path_data is not None:
                        all_lines = func_covdata['all_lines']
                        covered_lines = func_covdata['executed_lines']
                        missing_lines = func_covdata['missing_lines']
                        print(all_lines, covered_lines, missing_lines)
                        print(path_data['pathdata'])
                        quit()

                #quit()