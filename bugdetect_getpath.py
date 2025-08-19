#generate target execution paths for the bug detection experiment
import json
from cfg_util import path_find_from_cfg, extract_paths, get_path_from_ids
from myscalpel.builder import CFGBuilder


def path2txt(pathlist):
    """
    Convert an execution path (in list form) into text for prompt
    """
    path_txt = ''
    for i, node in enumerate(pathlist):
        #print(node)
        node_txt = ''
        block_id = node['id']
        node_src = node['src'].rstrip('\n')
        node_startline = node['start_line']
        node_endline = node['end_line']
        node_branch_type = node['branch_type']
        node_branch_condition = node['branch_condition']

        node_txt = f'Code block {block_id}: \n```\n{node_src}\n```\n'

        if node_branch_type == 'if' or node_branch_type == 'while':
            branch_txt = f'Branch condition: {node_branch_condition}'
        elif node_branch_type == 'for':
            node_forloop_iter = node['forloop_iter']
            if node_branch_condition != '': #loop is running
                branch_txt = f'Loop iteration: {node_forloop_iter}'
            else:
                branch_txt = '`for` loop condition not satisfied, exit loop'
        else:
            branch_txt = ''
        if branch_txt != '':
            node_txt += f'{branch_txt}'
        path_txt += f'{node_txt}\n\n'
    return path_txt



dataset_path = 'bugdetect-data/test-filtered.json'
cfg_builder = CFGBuilder()

bug_dataset = json.load(open(dataset_path))
bug_path_dataset = []
bug_path_dataset_raw = []  #store the path in list, not in text format
for i, data in enumerate(bug_dataset):
    data_src = data['code']
    bug_lineno = data['lineno']
    #print(data_src)
    #cfg_paths = path_find_from_cfg(data_src, max_paths=10, max_path_len=20)
    cfg = cfg_builder.build_from_src("cfg", data_src)
    #block_ids = [1, 2, 3]
    #cfg_paths = get_path_from_ids(block_ids, cfg)
    #print(cfg_paths)
    #quit()
    cfg_paths = extract_paths(cfg, max_paths=50, max_path_len=100, max_loop_iter=2)  #old version: max_path_len=30
    print(f'extracted {len(cfg_paths)} paths')
    cfg_paths.sort(key=len)
    if len(cfg_paths) > 50:
        cfg_paths = cfg_paths[:50]
    paths_txt_data = []
    for path in cfg_paths:
        #print(path)
        path_txt = path2txt(path)
        #print(path_txt)
        #print('-'*10)
        paths_txt_data.append(path_txt)
    bug_path_dataset.append({
        'problem_id': data['problem_id'],
        'submission_id': data['submission_id'],
        'code': data_src,
        'paths': paths_txt_data
    })
    bug_path_dataset_raw.append({
        'problem_id': data['problem_id'],
        'submission_id': data['submission_id'],
        'code': data_src,
        'paths': cfg_paths
    })
    #if i >= 0:
        #break

with open('bugdetect-data/test-filtered-paths.jsonl', 'w') as f:
    for bug_path_data in bug_path_dataset:
        f.write(json.dumps(bug_path_data) + '\n')
with open('bugdetect-data/test-filtered-paths-raw.jsonl', 'w') as f:
    for bug_path_data in bug_path_dataset_raw:
        f.write(json.dumps(bug_path_data) + '\n')