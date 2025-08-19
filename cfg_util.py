import argparse
import os
import subprocess
import json
import ast
import math
from collections import defaultdict

#from scalpel.cfg import CFGBuilder
from myscalpel.builder import CFGBuilder
#from scalpel.cfg.model import Block
from myscalpel.model import Block
from scalpel.core.mnode import MNode
from scalpel.SSA.const import SSA
import graphviz
from scalpel_utils import get_block_source, get_block_branchtype, get_block_lineno

def draw_cfg(cfg, name, format='png'):
    generated_graph = cfg.build_visual('png')

    output_path = generated_graph.render(
        format='png',
        filename=name,
        directory='../../output',  # Creates this directory if it doesn't exist
        cleanup=True
    )
    print(f"PNG saved to: {output_path}")



def path_find_from_cfg(src, max_paths=15, max_path_len=20):
    cfg_builder = CFGBuilder()
    #print(src)
    cfg = cfg_builder.build_from_src("cfg", src)
    #print(cfg.function_args)  #TODO why is empty? Shold contain all args for all funcs. Solved. Do not need this to extract function signatures.
    #print(cfg.class_cfgs)
    #print(cfg.functioncfgs)
    
    all_func_cfgs = []  #store all function cfgs parsed from the file under test
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
            all_func_cfgs.append({'class': class_name, 'func': func_name, 'cfg': func_cfg, 'func_fullname': func_fullname})
            traverse_cfg(func_cfg, class_name, func_fullname)
    for parent_cfg in all_parent_cfgs:
        traverse_cfg(parent_cfg['cfg'], parent_cfg['class'], parent_cfg['class'])
    

    '''#with open(path_file_name, "w") as f:
    for (block_id, func_name), func_cfg in cfg.functioncfgs.items():  #traverse function cfgs in the file, out of classes
        cfg_str = f"func_cfg:{block_id},{func_name}\n"
        #print(cfg_str)
        #f.write(cfg_str)
        func_entrypoint = func_cfg.entryblock
        cfg_str = f"func_entrypoint:{func_entrypoint}\n"
        #print(cfg_str)
        #f.write(cfg_str)
        all_func_cfgs.append({'class': '', 'func': func_name, 'cfg': func_cfg})

    for class_name, class_cfg in cfg.class_cfgs.items():  #traverse class cfgs in the file
        #print(class_name)
        #f.write(class_name)
        for (block_id, func_name), func_cfg in class_cfg.functioncfgs.items():  #traverse function cfgs inside the class
            #cfg_str = f"func_cfg:, {block_id},{func_name}\n"
            #print(cfg_str)
            #f.write(cfg_str)
            func_entrypoint = func_cfg.entryblock
            #cfg_str = f"func_entrypoint:{func_entrypoint}\n"
            #print(cfg_str)
            #f.write(cfg_str)
            all_func_cfgs.append({'class': class_name, 'func': func_name, 'cfg': func_cfg})'''


    #print('number of func cfgs:', len(all_func_cfgs))
    '''for func_cfg in all_func_cfgs: #deprecated
        print(func_cfg['cfg'], func_cfg['cfg'].functioncfgs)
        if len(func_cfg['cfg'].functioncfgs) > 0:
            print(f'Find nested functions in {func_cfg["class"]}.{func_cfg["func"]}')
            quit()'''
    #Now extract paths from all the function cfgs
    all_func_cfg_pathdata = []  #store the extracted paths and metadata for all cfgs
    for cfg_obj in all_func_cfgs:
        func_classname = cfg_obj['class']
        func_name = cfg_obj['func']
        func_cfg = cfg_obj['cfg']
        func_fullname = cfg_obj['func_fullname']

        #used only for debugging: draw the cfg
        #draw_cfg(func_cfg, f'{func_name}__{func_classname}')

        func_cfg_block_ids = func_cfg.get_all_block_ids()
        #print(f'Function name: {func_name}, block ids: {func_cfg_block_ids}')

        func_entrypoint = func_cfg.entryblock
        func_cfg_pathdata = extract_paths(func_cfg, max_paths=max_paths, max_path_len=max_path_len)
        #print(f'Number of paths extracted for {func_name}: {len(func_cfg_pathdata)}')

        #Experimental: minimizing the path set
        func_cfg_minpathdata = get_min_set_paths(func_cfg_pathdata)
        #print(f'Number of paths in the minimum path set for {func_name}: {len(func_cfg_minpathdata)}')
        #if func_name == 'isotonic_regression':
            #quit()

        func_cfg_signature = func_cfg.func_signature
        func_cfg_start_line = func_cfg.start_line
        func_cfg_end_line = func_cfg.end_line
        

        func_cfg_pathdata_obj = {'class': func_classname, 'func': func_name, 'func_fullname': func_fullname, 'pathdata': func_cfg_minpathdata, 'signature': func_cfg_signature, 'start_line': func_cfg_start_line, 'end_line': func_cfg_end_line}
        all_func_cfg_pathdata.append(func_cfg_pathdata_obj)
    
    return all_func_cfg_pathdata


def analyze_for_loop(for_node, iteration):
    """
    Analyze whether the loop condition is definite and can be satisfied.
    Args:
        for_node: The for loop node.
        iteration: The current iteration number (starting from 1).
    Returns:
    """
    # target must be simple
    if not isinstance(for_node.target, ast.Name):
        #raise NotImplementedError("Only simple 'for x in ...:' targets supported")
        #print("Only simple 'for x in ...:' targets supported")
        return 'UNK'
    var_name = for_node.target.id
    it = for_node.iter

    # Case 1: range()
    if isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == "range":
        args = it.args
        # normalize start, end, step
        if len(args) == 1:
            start, end, step = ast.Constant(0), args[0], ast.Constant(1)
        elif len(args) == 2:
            start, end, step = args[0], args[1], ast.Constant(1)
        elif len(args) == 3:
            start, end, step = args
        else:
            raise ValueError("Unsupported range(...) signature")
        #print('start, end, step:', start, end, step)

        # only handle constant start/end/step for bounds check
        if all(isinstance(x, ast.Constant) for x in (start, end, step)):
            s, e, st = start.value, end.value, step.value
            #print('s, e, st:', s, e, st)
            # compute number of iterations
            if st > 0:
                count = max(0, math.ceil((e - s) / st))  #e.g., for i in range(10), count = 10
            else:
                count = max(0, math.ceil((s - e) / abs(st)))
            # out-of-bounds?
            if iteration > count:  #e.g., for i in range(10), iteration = 11, count = 10, so it is out of bounds
                return False
        else:
            # cannot determine bounds for non-constant range
            #raise NotImplementedError("Iteration validity unknown for symbolic range")
            return 'UNK'

        # now compute value and definiteness
        is_definite = isinstance(start, ast.Constant) and isinstance(step, ast.Constant)
        if is_definite:
            value = s + (iteration-1) * st
            #return var_name, repr(value), True
            return True
        else:
            mul = ast.BinOp(left=step, op=ast.Mult(), right=ast.Constant(iteration-1))
            add = ast.BinOp(left=start, op=ast.Add(), right=mul)
            #return var_name, ast.unparse(add), False
            return False

    # Case 2: literal list
    if isinstance(it, ast.List):
        elems = it.elts
        if iteration >= len(elems):
            return False
        else:
            return True
        elem = elems[iteration]
        is_definite = isinstance(elem, ast.Constant)
        #return var_name, repr(elem.value) if is_definite else ast.unparse(elem), is_definite

    #raise NotImplementedError("Only range(...) or literal list supported")
    return 'UNK'


def extract_paths(cfg, max_paths=10, max_path_len=10, covered_lineno=0, max_loop_iter=3):
    """
    Extract all paths from the CFG starting from the entry block.
    Args:
        cfg: The input CFG.
        max_paths: The maximum number of paths to extract.
        max_path_len: The maximum length of a path.
        covered_lineno(optional): The line number that need to be covered by the paths. TODO
    Returns:
        A list of paths, where each path is a list of block metadata.
    """
    loop_iter= defaultdict(int)  #global variable to control the iteration for each loop
    entry_block = cfg.entryblock
    all_block_ids = cfg.get_all_block_ids()
    #print(f'all block ids: {all_block_ids}')
    covered_block_ids = set()  #store the block ids covered by the coollected paths
    all_paths = []

    def visit(block, visited=[], calls = True, max_path_len=10, forloop_stat={}, max_loop_iter=max_loop_iter, covered_lineno=covered_lineno):
        """
        Recursively visit all the blocks of the CFG, and collect execution paths. When a path is complete or reaches the max_path_len, add it to the all_paths list.
        Args:
            block: the current block to visit.
            visited: the partial execution path of visited blocks.
            calls: whether to collect function calls.
            max_path_len: the maximum length of a path.
            forloop_stat: store the number of iterations for all for loops in the current path.
            max_loop_iter: the maximum number of iterations for a for loop.
        """
        global loop_iter
        if len(all_paths) >= max_paths and len(covered_block_ids) >= len(all_block_ids):  #reach maximum number of paths and covered all blocks
            return

        if len(visited) >= max_path_len:
            #print(visited)
            if visited not in all_paths:
                all_paths.append(visited)
                for node in visited:  #add the block ids to the covered_block_ids set
                    covered_block_ids.add(node['id'])
            return
        
        #build node
        node_obj = {}
        #nodelabel = block.get_source()
        node_txt = get_block_source(block)
        node_obj['id'] = block.id
        node_obj['src'] = node_txt
        node_startline, node_endline = get_block_lineno(block)
        node_obj['start_line'] = node_startline
        node_obj['end_line'] = node_endline
        node_obj['branch_type'] = get_block_branchtype(block)
        
        #visited_new = visited + [node_obj]

        if not block.exits:  # Natural end of path
            #print(f'path end: {node_obj}')
            node_obj['branch_condition'] = ''
            visited_new = visited.copy() + [node_obj]
            if visited_new not in all_paths:
                all_paths.append(visited_new)
                for node in visited_new:  #add the block ids to the covered_block_ids set
                    covered_block_ids.add(node['id'])
        
        if calls and block.func_calls:  #TODO: in the original scalpel code, this deals with function calls.
            calls_node = str(block.id)+"_calls"
            calls_label = block.get_calls().strip()
            #print('calls:', calls_node, calls_label)

        
        #validate the status of `for` loop condition
        forloop_valid = None
        if node_obj['branch_type'] == 'for':
            for_node = block.statements[-1]
            #loop_var_name = for_node.target.id
            #loop_iter = for_node.iter
            for_eval = analyze_for_loop(for_node, iteration=3)
            #print('for_eval:', for_eval)
            #quit()
            if for_eval == False:
                #print(f'for loop condition cannot be satisfied: {node_obj}')
                forloop_valid = False
            elif for_eval == True:
                #if node_obj['src'].find('3501')==-1:
                    #print(f'for loop condition should be satisfied: {node_obj}')
                forloop_valid = True
        
        # recursively visit all the blocks of the CFG.
        for i, exit in enumerate(block.exits):  #use i to distinguish between branch conditions. Assume that 0 means take true branch, 1 means false.
        #for i in reversed(range(len(block.exits))):  #reversed traversal. Encourage exiting loops first
            #exit = block.exits[i]
            edgelabel = exit.get_exitcase().strip()
            node_withcond = node_obj.copy()
            node_withcond['branch_condition'] = edgelabel
            #print('branch number:', i, 'edge:', edgelabel, 'exit node:', exit)
            #print(f'node: {node_withcond}, edge: {edgelabel}')
            #if node_withcond['branch_type'] in ['if', 'for', 'while']:
                #print('branch number:', i, node_withcond['src'])
                #print('branch type', node_withcond['branch_type'], 'branch condition', node_withcond['branch_condition'])

            #Deal with for loop iteration number
            if node_withcond['branch_type'] in ['for', 'while']:  #Also count iterations for `while` loops, but only use them too control the traversal depth.
                if i ==0:  #Take the true branch, enter the loop.
                    forloop_stat[node_withcond['id']] = forloop_stat.get(node_withcond['id'], 0) + 1  #loop iteration starts from 1
                elif i ==1:  #Take the false branch, exit the loop.
                    forloop_stat[node_withcond['id']] = 0  #0 means exit the loop
                #print(forloop_stat[node_withcond['id']])
                

                #control the loop traversal with the forloop_valid status
                if forloop_valid == False and i == 0:  #should only take the false branch
                    return
                elif forloop_valid == True and i == 1:  #should only take the true branch
                    return

                node_withcond['forloop_iter'] = forloop_stat[node_withcond['id']]
                #print(f'For loops iteration status: {forloop_stat}')
                
                #if i == 0 and forloop_stat[node_withcond['id']] > max_loop_iter:  #limit the number of iterations for 'for' or 'while loops
                    #continue
                if forloop_stat[node_withcond['id']] > max_loop_iter: #if the number of iterations is greater than the max_loop_iter, stop the path and add it to the all_paths list
                    if visited not in all_paths:
                        all_paths.append(visited)
                        for node in visited:  #add the block ids to the covered_block_ids set
                            covered_block_ids.add(node['id'])
                    return #end the traversal of the current path


            visited_new = visited.copy() + [node_withcond]  #append the currect block node to the path AFTER we generate the branch condition label
            visit(exit.target, visited_new.copy(), max_path_len=max_path_len, forloop_stat=forloop_stat.copy(), max_loop_iter=max_loop_iter)

    visit(entry_block, visited=[], calls=True, max_path_len=max_path_len, max_loop_iter=max_loop_iter)
    #print(f'all blocks: {all_block_ids}')
    #print(f'covered blocks: {covered_block_ids}')
    return all_paths
    
#Deprecated: this function cannot be used for selecting minimum path set. Shouldelect the path set by the CFG edges.
def if_node_in_paths(node, paths):  #check if the node in the minimum path set
    for path in paths:
        #print([nd['id'] for nd in path])
        for nd in path:
            #print(node['id'], nd['id'])
            if node['id'] == nd['id']:
                return True
    return False

#Deprecated: this function cannot be used for selecting minimum path set. Shouldelect the path set by the CFG edges.
def get_min_set_paths_old(all_paths): #get the minimum path set
    minset_paths=[]
    for path in all_paths:
        print([nd['id'] for nd in path])
        for node in path:
            if not if_node_in_paths(node, minset_paths): # if there is a node of the path is not in the minimum set , then add the path to the minimum set.
                minset_paths.append(path)
    return minset_paths


def get_min_set_paths(all_paths):
    """
    Get the minimum path set by selecting the path set by the CFG edges.
    Args:
        all_paths: The list of all paths.
    Returns:
        The list ofminimum path set.
    """
    minset_paths = []
    edges_set = set()
    for path in all_paths:
        if len(path) <= 1:
            minset_paths.append(path)
        else:
            add_edge = False
            for i in range(len(path)-1):
                cfg_edge = (path[i]['id'], path[i+1]['id'])
                if cfg_edge not in edges_set:
                    edges_set.add(cfg_edge)
                    add_edge = True
                    break
            if add_edge:
                minset_paths.append(path)
    #print(edges_set)
    return minset_paths


def get_path_from_ids(block_ids, cfg):
    """Generate a CFG path from a sequence of block ids.
    Args:
        block_ids: The list of block ids.
        cfg: The CFG.
    Returns:
        The execution path in list format (the same as extract_paths).
    """
    path = []
    forloop_stat = {}
    
    # Get all blocks from the CFG
    all_blocks = cfg.get_all_blocks()
    # Create a mapping from block id to block for efficient lookup
    block_map = {block.id: block for block in all_blocks}

    
    for i, block_id in enumerate(block_ids):
        if block_id in block_map:
            block = block_map[block_id]
            
            # Create node object similar to the pattern used in extract_paths
            node_obj = {}
            node_txt = get_block_source(block)
            node_obj['id'] = block.id
            node_obj['src'] = node_txt
            node_startline, node_endline = get_block_lineno(block)
            node_obj['start_line'] = node_startline
            node_obj['end_line'] = node_endline
            node_obj['branch_type'] = get_block_branchtype(block)
            node_obj['branch_condition'] = ''
            

            # Find the exit that leads to the next block in the sequence
            next_block_id = block_ids[i + 1] if i + 1 < len(block_ids) else None
            if next_block_id:
                found_next = False
                for exit in block.exits:
                    if exit.target.id == next_block_id:
                        found_next = True
                        node_obj['branch_condition'] = exit.get_exitcase().strip()
                        break
                if not found_next:  #next block id not found in the CFG
                    print(f'Cannot find CFG edge for next block {next_block_id} from block {block_id}')
                    return None
                
            #deal with loops, add loop iteration number
            if node_obj['branch_type'] in ['for']:  #only for 'for' loops this time, not 'while' loops
                if node_obj['branch_condition'] != '':
                    forloop_stat[node_obj['id']] = forloop_stat.get(node_obj['id'], 0) + 1
                else:
                    forloop_stat[node_obj['id']] = 0
                node_obj['forloop_iter'] = forloop_stat[node_obj['id']]
            
            
            path.append(node_obj)
        else:  #block id not found in the CFG
            print(f'Block {block_id} not found in CFG')
            return None
    
    return path
