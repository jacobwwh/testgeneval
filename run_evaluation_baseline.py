# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Run evaluation"""
import argparse
import asyncio
import logging
import os

from swebench_docker.constants import KEY_BASELINES, KEY_ID
from swebench_docker.run_docker import run_docker_evaluation
from swebench_docker.utils import get_eval_refs

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("run_evaluation_baseline")


async def main(
    swe_bench_tasks: str,
    namespace: str,
    log_dir: str,
    skip_existing: bool = False,
    timeout: int = 60,
    num_processes: int = -1,
):
    """
    Runs evaluation on predictions for each model/repo/version combination.
    """
    if not os.path.exists(log_dir) or not os.path.isdir(log_dir):
        raise ValueError("--log_dir must exist and point at a directory")
    os.chmod(log_dir, 0o777)

    tasks = list(get_eval_refs(swe_bench_tasks).values())
    if not isinstance(tasks, list):
        raise ValueError(f"{swe_bench_tasks} must contain an array of tasks")

    filtered_tasks = []
    for task_instance in tasks:
        all_settings_exist = True
        for setting in ["first", "last_minus_one", "last"]:
            #Only evaluate 'last' setting (complete test cases)
            #if setting != "last": (not working)
                #continue
            log_file_name = f"{task_instance[KEY_ID]}.baseline.{setting}.eval.log"
            log_file = os.path.join(log_dir, log_file_name)
            if not os.path.exists(log_file):
                all_settings_exist = False
                break
        if not all_settings_exist or not skip_existing:
            filtered_tasks.append(task_instance)

    if len(filtered_tasks) == 0:
        logger.info(f"All logs already exist, skipping")
        return
    else:
        logger.info(
            f"# of baselines to evaluate: {len(filtered_tasks)} "
            + f"({len(tasks) - len(filtered_tasks)} already evaluated)"
        )
        tasks = filtered_tasks

    sem = asyncio.Semaphore(num_processes if num_processes > 0 else len(tasks))
    asyncio_tasks = []
    for task_instance in tasks:
        for setting in task_instance[KEY_BASELINES]:
            if setting in ["preamble", "none"]:
                continue

            async def run_docker_throttled(task_instance, setting):
                async with sem:
                    return await run_docker_evaluation(
                        task_instance,
                        namespace,
                        log_dir,
                        setting,
                        timeout,
                        only_baseline=True,
                        verbose=True,
                        skip_mutation=True,
                    )

            task = asyncio.create_task(run_docker_throttled(task_instance, setting))
            asyncio_tasks.append(task)

    results = await asyncio.gather(*asyncio_tasks)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--swe_bench_tasks", type=str, required=True)
    parser.add_argument("--namespace", type=str, default="aorwall")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--num_processes", type=int, default=-1)
    args = parser.parse_args()
    asyncio.run(main(**vars(args)))
