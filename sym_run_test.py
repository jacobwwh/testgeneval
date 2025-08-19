#execute the generated test cases, get logs and coverage reports
# Adapted from: https://github.com/facebookresearch/testgeneval/blob/main/swebench_docker/run_docker.py
# and https://github.com/facebookresearch/testgeneval/blob/main/run_evaluation.py
import os
import json
import argparse
import logging
import dotenv
import tempfile
import time
import asyncio
import subprocess
from tqdm import tqdm

from swebench_docker.constants import (
    MAP_VERSION_TO_INSTALL, 
    KEY_ID,
    KEY_INSTANCE_ID,
    KEY_MODEL,
    KEY_PREDICTIONS,
    MAP_REPO_TO_TEST_FRAMEWORK,
)

from swebench_docker.swebench_utils import get_instances, get_test_directives

from datasets import load_dataset
dataset = load_dataset("kjain14/testgenevallite")
test_set = dataset["test"]

argparser = argparse.ArgumentParser()
argparser.add_argument('--setting', type=str, default='path', choices=['path', 'baseline']) #whether to use execution paths
argparser.add_argument('--model', type=str, default='gpt-4.1-nano')
args = argparser.parse_args()

def postprocess_output(text):
    """Modified from https://github.com/facebookresearch/testgeneval/blob/main/inference/configs/instruct_prompt.py"""
    text = text.replace("```python", "```")
    if "```" not in text:  #may be compilation error, but may also be the LLM did not generate '```'
        return text
    text_cleaned = text.split("```")[1].split("```")[0]
    return text_cleaned



logger = logging.getLogger(__name__)
dotenv.load_dotenv()


if __name__ == '__main__':
    # Configure logging after argument parsing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()  # This will output to console
        ]
    )

    
    if args.setting == 'path':
        preds_path = f'results/testgenevallite/preds/{args.model}.jsonl'
    elif args.setting == 'baseline':
        preds_path = f'results/testgenevallite/preds/{args.model}_baseline.jsonl'

    namespace = "aorwall"
    if args.setting == 'path':
        log_dir = os.path.join(os.path.dirname(__file__), f'results/testgenevallite/data_logs/{args.model}')
    elif args.setting == 'baseline':
        log_dir = os.path.join(os.path.dirname(__file__), f'results/testgenevallite/data_logs/{args.model}_baseline')
    print(f'log_dir: {log_dir}')
    os.makedirs(log_dir, exist_ok=True)
    os.chmod(log_dir, 0o777)  #set the permission so that the logs can be written

    with open(preds_path, 'r') as f:
        for i, line in tqdm(enumerate(f.readlines())):
            data = json.loads(line)
            instance_id = data['instance_id']
            preds = data['preds']
            print(f'Processing instance_id: {instance_id}')

            task_instance = next((sample for sample in test_set if sample['instance_id'] == instance_id), None)
            assert task_instance is not None, f'Test set sample not found for instance_id: {instance_id}'


            repo_name = task_instance["repo"].replace("/", "_")

            specifications = MAP_VERSION_TO_INSTALL[task_instance["repo"]][
                task_instance["version"]
            ]
            image_prefix = "swe-bench"
            print(f'repo name: {repo_name}')
            print(f'specifications: {specifications}')

            # TODO: Change this when deciding
            if "packages" in specifications and specifications["packages"] == "environment.yml":
                container_log_dir = "/home/swe-bench/logs"
            else:
                container_log_dir = "/opt/logs"
            print(f'container_log_dir: {container_log_dir}')

            if specifications.get("instance_image", False):
                docker_image = f"{namespace}/{image_prefix}-{repo_name}-instance:{task_instance['instance_id']}"
            else:
                docker_image = (
                    f"{namespace}/{image_prefix}-{repo_name}-testbed:{task_instance['version']}"
                )
            print(f'docker_image: {docker_image}')

            swebench_docker_fork_dir = os.environ.get("SWEBENCH_DOCKER_FORK_DIR") #should be the testgeneval root fir
            #swebench_docker_fork_dir = "/home/wwh/testgeneval/"
            print(swebench_docker_fork_dir)

            

            for func_name, func_pred in preds.items():
                print(f'Running test for function: {func_name}')
                func_test = postprocess_output(func_pred)
                #print(func_test)

                test_type = MAP_REPO_TO_TEST_FRAMEWORK[task_instance["repo"]]
                #print(f'test_type: {test_type}')
                test_directives = get_test_directives(task_instance)  #path of the test files
                #print(f'test_directives: {test_directives}')
                test_cmd = f"{test_type} {' '.join(test_directives)}"
                #print(f'test command: {test_cmd}')
                
                #create a new task_instance for docker run (will be written to a temp file)
                new_task_instance = {
                    "instance_id": task_instance["instance_id"],
                    "id": task_instance["id"],
                    "repo": task_instance["repo"],
                    "version": task_instance["version"],
                    "base_commit": task_instance["base_commit"],
                    "preds_context": task_instance["preds_context"]["last"], #use the 'extra' setting for context
                    "test_patch": task_instance["test_patch"],
                    "test_file": task_instance["test_file"],
                    "code_file": task_instance["code_file"],
                    "patch": task_instance["patch"],
                    "test_func_name": func_name, #the name of the func under test
                    "pred": func_test,  #the generated test case
                    KEY_MODEL: args.model,  #model_name_or_path: model name
                    "test_directives": test_directives,
                    "test_cmd": test_cmd,
                }

                if swebench_docker_fork_dir:
                    # Create a temporary file to store the task_instance JSON
                    tmpfile_path = tempfile.mktemp(suffix=".json")
                    print(f'tmpfile_path: {tmpfile_path}')
                    with open(tmpfile_path, "w+") as f:
                        json.dump(new_task_instance, f)

                    docker_command = [
                        "docker",
                        "run",
                        "--rm",
                        "--network",
                        "host",
                        "-v",
                        f"{log_dir}:{container_log_dir}",
                        # Map the swebench_docker fork dir to the container
                        # for some reason, swebench_docker has different locations for the different containers :(
                        # so we need to map all of them to make it work
                        "-v",
                        f"{swebench_docker_fork_dir}/swebench_docker:/opt/swebench_docker:ro",
                        "-v",
                        f"{swebench_docker_fork_dir}/swebench_docker:/home/swe-bench/swebench_docker:ro",
                        "-v",
                        f"{swebench_docker_fork_dir}/swebench_docker:/home/swe-bench/swebench:ro",
                        # =======
                        # Map file instead pass the instance as env var to avoid "Argument list too long" error
                        "-v",
                        f"{tmpfile_path}:/home/swe-bench/task_instance.json:ro",
                        "-e",
                        f"LOG_DIR={container_log_dir}",
                        "-e",
                        f"IND=60",  #unknown param, default to 60 TODO: remove it
                        "-e",
                        f"TIMEOUT=60",  #set timeout to 60s
                        "-e",
                        f"SKIP_MUTATION=True",  #skip mutation TODO: changeable
                        docker_image,
                    ]

                    cmd_string = " ".join(docker_command)
                    #print(f'cmd_string: {cmd_string}')
                    
                    
                    #start running docker command
                    logger.info(f"Running docker command: {cmd_string}")

                    start_time = time.time()
                    
                    try:
                        result = subprocess.run(
                            docker_command,
                            capture_output=True,
                            text=True,
                            timeout=120  # 2 minute timeout for the entire docker run
                        )
                            
                        end_time = time.time()
                        execution_time = end_time - start_time
                        
                        logger.info(f"Docker command completed in {execution_time:.2f} seconds")
                        logger.info(f"Return code: {result.returncode}")
                        
                        if result.stdout:
                            logger.info(f"STDOUT: {result.stdout}")
                        if result.stderr:
                            logger.info(f"STDERR: {result.stderr}")
                    except Exception as e:
                        logger.error(f"[{task_instance['id']}][{docker_image}]  Error running container: {e}")
                            


            #quit()