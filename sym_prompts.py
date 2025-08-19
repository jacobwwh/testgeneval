class InstructPrompt:
    """Modified from https://github.com/facebookresearch/testgeneval/blob/main/inference/configs/instruct_prompt.py"""
    def __init__(self):
        self.SYSTEM_MESSAGE = "You are an expert Python software testing assistant, specializing in analyzing program execution paths for  symbolic execution. Your job is to generate a test case for a function to cover given execution paths."
        self.SYSTEM_MESSAGE_FULL = "You are an expert Python automated testing assistant. Your job is to generate a test file given a code file."

        self.PROMPT_COMPLETION = """Below is a code file:
```python
{code_src}
```

And the current unit test file
```python
{test_src}
```

Your job is to generate a unit test case for function `{func_name}`, the generated test case will be written to the test file. Your generated test case should cover the execution paths for the function, which are given below:

[Begin of execution paths]
{all_paths_txt}
[End of execution paths]

You should analyze all given execution paths, and generate test inputs that cover all the satisfiable paths. If a given path is not satisfiable, you should ignore it.
Ideally your generated test case should improve the coverage of the existing test file for the code file.

Only output the generated unit test, preserve indentation and formatting. Do not output anything else. Format like this:

```python
Next unit test Python code
```
"""
        self.PROMPT_COMPLETION_BASELINE = """Below is a code file:
```python
{code_src}
```

And the current unit test file
```python
{test_src}
```

Your job is to generate a unit test case for function `{func_name}`, the generated test case will be written to the test file.
Ideally your generated test case should improve the coverage of the existing test file for the code file.

Only output the generated unit test, preserve indentation and formatting. Do not output anything else. Format like this:

```python
Next unit test Python code
```
"""

    def combine_paths(self, paths_txt):
        """
        Combine the execution paths into a single string.
        """
        combined_txt = ''
        for i, path_txt in enumerate(paths_txt):
            combined_txt += f'Path {i+1}:\n{path_txt}\n----------\n'
        return combined_txt


    def create_prompt(self, code_src, test_src, func_name, with_paths=True, all_paths_txt=None):
        """
        Create a prompt for the function.
        """
        if with_paths:
            assert all_paths_txt is not None, "all_paths_txt is required when with_paths is True"

            all_paths_txt = self.combine_paths(all_paths_txt)
            prompt = self.PROMPT_COMPLETION.format(code_src=code_src, test_src=test_src, func_name=func_name, all_paths_txt=all_paths_txt)
        else:
            prompt = self.PROMPT_COMPLETION_BASELINE.format(code_src=code_src, test_src=test_src, func_name=func_name)
        return prompt