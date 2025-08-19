

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