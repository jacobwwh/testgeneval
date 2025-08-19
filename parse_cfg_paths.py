#parse the cfg paths of the partially covered functions in testgenevallite (for symbolic execution)

import json
import os
from datasets import load_dataset
dataset = load_dataset("kjain14/testgenevallite")
test_set = dataset["test"]

from myscalpel.builder import CFGBuilder
#from myscalpel.model import Block
#from scalpel.core.mnode import MNode
#from scalpel.SSA.const import SSA
#from scalpel_utils import get_block_source, get_block_branchtype, get_block_lineno
from cfg_util import extract_paths
from path_utils import path2txt

#load the metadata of the partially covered functions
funcs_metadata = json.load(open('sym_data/metadata.json'))

test_set_filtered = [e for e in test_set if e['instance_id'] in funcs_metadata]
print(f'number of test set instances: {len(test_set_filtered)}') #83


def get_all_cfgs(src):
    """
    Extract all function CFGs from the source code.
    Args:
        src: The source code to extract CFGs from.
    Returns:
        A list of function CFGs.
    """
    cfg_builder = CFGBuilder()
    cfg = cfg_builder.build_from_src("cfg", src)
    
    all_func_cfgs = {}  #store all function cfgs parsed from the file under test
    all_parent_cfgs = []  #store all parent cfgs, including the root cfg and class cfgs, and parent cfgs for nested functions
    all_parent_cfgs.append({'class': '', 'cfg': cfg})
    for class_name, class_cfg in cfg.class_cfgs.items():
        all_parent_cfgs.append({'class': class_name, 'cfg': class_cfg})
    
    #recursively traverse all parent cfgs to find nested functions
    def traverse_cfg(cfg, class_name='', parent_hierarchy=''):
        """
        Traverse the CFG to find all function CFGs (including functions in classes and nested ones).
        Args:
            cfg: The CFG to traverse.
            parent_hierarchy: The hierarchy of the current function (the list from the root to the current function, may include class names, outer functions).
        """
        for (block_id, func_name), func_cfg in cfg.functioncfgs.items():
            if parent_hierarchy == '':
                func_fullname = func_name
            else:
                func_fullname = parent_hierarchy + '.' + func_name
            #print(func_cfg.start_line, func_cfg.end_line)
            all_func_cfgs[func_fullname] = {'class': class_name, 'func': func_name, 'cfg': func_cfg, 'start_line': func_cfg.start_line, 'end_line': func_cfg.end_line}
            traverse_cfg(func_cfg, class_name, func_fullname)
    for parent_cfg in all_parent_cfgs:
        traverse_cfg(parent_cfg['cfg'], parent_cfg['class'], parent_cfg['class'])
    return all_func_cfgs

print(test_set_filtered[0].keys())

all_paths_data = []
num_parsed_funcs = 0
for test_data in test_set_filtered:
    instance_id = test_data['instance_id']
    instance_metadata = funcs_metadata[instance_id] #contains functions partially covered in this instance
    instance_cfg_data = {}

    print(f'Processing instance {instance_id}')
    #if instance_id != 'django__django-15252':
        #continue
    code_src = test_data['code_src']
    code_file = test_data['code_file']
    try:
        all_func_cfgs = get_all_cfgs(code_src)
        for func_fullname in instance_metadata:
            print(func_fullname)
            if func_fullname in all_func_cfgs: #found the cfg for the function
                parsed_cfg_data = all_func_cfgs[func_fullname]
                if parsed_cfg_data['start_line'] <= instance_metadata[func_fullname]['start_line'] and parsed_cfg_data['end_line'] >= instance_metadata[func_fullname]['end_line']:
                    func_cfg = parsed_cfg_data['cfg']
                    num_parsed_funcs += 1
                    #max_paths=10, max_path_len=30, max_loop_iter=2
                    func_cfg_paths = extract_paths(func_cfg, max_paths=10, max_path_len=30, max_loop_iter=2)
                    print(f'extracted {len(func_cfg_paths)} paths')
                    func_cfg_paths = func_cfg_paths[:10]
                    #print(func_cfg_paths[-1])
                    parsed_cfg_data['paths'] = func_cfg_paths
                    paths_txt = []
                    for path in func_cfg_paths:
                        path_txt = path2txt(path)
                        paths_txt.append(path_txt)
                    parsed_cfg_data['paths_txt'] = paths_txt
                    #quit()
                    parsed_cfg_data.pop('cfg')  #remove the cfg, it cannot be written to json
                    instance_cfg_data[func_fullname] = parsed_cfg_data
                else:
                    print(f'CFG for {func_fullname} does not match the line numbers')
            else: #did not find the cfg for the function
                print(f'{func_fullname} not found in CFGs')
        all_paths_data.append({
            'instance_id': instance_id,
            'data': instance_cfg_data})
    except Exception as e:
        print(f'Error parsing CFGs for {instance_id}: {e}')
    #quit()

print(f'number of functions to be tested: {num_parsed_funcs}')

with open('sym_data/paths_data.json', 'w') as f:
    for data in all_paths_data:
        f.write(json.dumps(data) + '\n')




