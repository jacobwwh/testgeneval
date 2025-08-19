#include self-implemented functions modified from the original Scalpel library
from scalpel.cfg.model import Block
import ast
import astor


def get_block_source(block, return_list=False):
    """
    Get a string containing the Python source code corresponding to the
    statements in the block.
    Returns:
        A string containing the source code of the statements.
    Modified from `get_source` in `scalpel.cfg.model.Block`
    """
    src = ''
    src_lines = []
    for statement in block.statements:  #block.statements is a list of ast statements
        if type(statement) in [ast.If, ast.For, ast.While, ast.With]:
            src_stmt = (astor.to_source(statement)).split('\n')[0] + "\n"
        elif type(statement) == ast.Try:
            src_stmt = (astor.to_source(statement)).split('\n')[0] + "\n"
        #elif type(statement) == ast.If:
        #    src += (astor.to_source(statement)).split('\n')[0] + "\n"
        elif type(statement) in [ast.FunctionDef,ast.AsyncFunctionDef,
                ast.ClassDef]:
            src_stmt = (astor.to_source(statement)).split('\n')[0] + "...\n"
        elif type(statement) == ast.ClassDef:
            src_stmt = (astor.to_source(statement)).split('\n')[0] + "...\n"
        else:
            src_stmt = astor.to_source(statement)

        src += src_stmt
        src_lines.append(src_stmt)
    if return_list:
        return src_lines
    else:
        return src


def get_block_branchtype(block):
    """
    In Scalpel, the last statement in a cfg block is often a branch statement.
    This function returns the statement type (if, for, while) of the last statement in a cfg block.
    """
    block_laststmt = block.statements[-1]  
    if type(block_laststmt) == ast.If:
        return 'if'
    elif type(block_laststmt) == ast.For:
        return 'for'
    elif type(block_laststmt) == ast.While:
        return 'while'
    else:
        return ''
    

def get_block_lineno(block):
    """
    Get the line number of the first and last statement in the block.
    """
    first_lineno = block.statements[0].lineno
    last_lineno = block.statements[-1].lineno
    return first_lineno, last_lineno