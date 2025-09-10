import importlib
import logging
import os
import warnings
from dataclasses import dataclass
from itertools import repeat
from multiprocessing import Pool
from pathlib import Path
from typing import Literal

import click
from dotenv import load_dotenv
from rich import print
from rich.live import Live
from rich.logging import RichHandler

from agentdojo.agent_pipeline.agent_pipeline import DEFENSES, AgentPipeline, PipelineConfig
from agentdojo.attacks.attack_registry import ATTACKS, load_attack
from agentdojo.benchmark import SuiteResults, benchmark_suite_with_injections, benchmark_suite_without_injections
from agentdojo.logging import OutputLogger
from agentdojo.models import ModelsEnum
from agentdojo.task_suite.load_suites import get_suite, get_suites
from agentdojo.task_suite.task_suite import TaskSuite

from agentdojo.scripts.vllm_utils import start_vllm, unload_lora_modules


def benchmark_suite(
    suite_name: str,
    model: ModelsEnum,
    logdir: Path,
    benchmark_version: str,
    user_tasks: tuple[str, ...] = (),
    injection_tasks: tuple[str, ...] = (),
    model_id: str | None = None,
    attack: str | None = None,
    defense: str | None = None,
    tool_delimiter: str = "tool",
    system_message_name: str | None = None,
    system_message: str | None = None,
    force_rerun: bool = False,
    live: Live | None = None,
    tool_output_format: Literal["yaml", "json"] | None = None,
) -> SuiteResults:
    if not load_dotenv(".env"):
        warnings.warn("No .env file found")

    suite = get_suite(benchmark_version, suite_name)

    print(f"Running benchmark for suite: '{suite.name}'")
    print(f"Using model: '{model}'")
    if attack is not None:
        print(f"Using attack: '{attack}'")
    if defense is not None:
        print(f"Using defense: '{defense}'")
    if len(user_tasks) > 0:
        print(f"Using user tasks: {', '.join(user_tasks)}")

    pipeline = AgentPipeline.from_config(
        PipelineConfig(
            llm=model,
            model_id=model_id,
            defense=defense,
            tool_delimiter=tool_delimiter,
            system_message_name=system_message_name,
            system_message=system_message,
            tool_output_format=tool_output_format,
        )
    )
    with OutputLogger(str(logdir), live=live):
        if attack is None:
            results = benchmark_suite_without_injections(
                pipeline,
                suite,
                user_tasks=user_tasks if len(user_tasks) != 0 else None,
                logdir=logdir,
                force_rerun=force_rerun,
                benchmark_version=benchmark_version,
            )
        else:
            attacker_ = load_attack(attack, suite, pipeline)
            results = benchmark_suite_with_injections(
                pipeline,
                suite,
                attacker_,
                user_tasks=user_tasks if len(user_tasks) != 0 else None,
                injection_tasks=injection_tasks if len(injection_tasks) != 0 else None,
                logdir=logdir,
                force_rerun=force_rerun,
                benchmark_version=benchmark_version,
            )
    print(f"Finished benchmark for suite: '{suite.name}'")

    return results


def show_results(suite_name: str, results: SuiteResults, show_security_results: bool):
    utility_results = results["utility_results"].values()
    avg_utility = sum(utility_results) / len(utility_results)

    print(f"Results for suite {suite_name}")
    print(f"Average utility: {avg_utility * 100:.2f}%")

    if show_security_results:
        passed_injection_tasks = sum(results["injection_tasks_utility_results"].values())
        total_injection_tasks = len(results["injection_tasks_utility_results"])
        print(f"\nPassed injection tasks as user tasks: {passed_injection_tasks}/{total_injection_tasks}")

        security_results = results["security_results"].values()
        avg_security = sum(security_results) / len(security_results)
        print(f"Average security: {avg_security * 100:.2f}%")


def main(
    suites: tuple[str, ...],
    model: ModelsEnum,
    benchmark_version: str = "v1.2.1",
    logdir: Path = Path("./runs"),
    user_tasks: tuple[str, ...] = (),
    injection_tasks: tuple[str, ...] = (),
    model_id: str | None = None,
    attack: str | None = None,
    defense: str | None = None,
    tool_delimiter: str = "tool",
    system_message_name: str | None = None,
    system_message: str | None = None,
    max_workers: int = 1,
    force_rerun: bool = False,
    modules_to_load: tuple[str, ...] = (),
):
    for module in modules_to_load:
        importlib.import_module(module)

    if len(suites) == 0:
        suites = tuple(get_suites(benchmark_version).keys())

    if len(suites) != 1:
        print(f"Benchmarking suites {', '.join(suites)}")

    if len(suites) != 1 and len(user_tasks) != 0:
        raise ValueError("A user task can be specified only when one suite is being executed")

    if max_workers == 1:
        results = {}
        for suite_name in suites:
            results[suite_name] = benchmark_suite(
                suite_name,
                model,
                logdir,
                benchmark_version=benchmark_version,
                user_tasks=user_tasks,
                injection_tasks=injection_tasks,
                model_id=model_id,
                attack=attack,
                defense=defense,
                tool_delimiter=tool_delimiter,
                system_message_name=system_message_name,
                system_message=system_message,
                force_rerun=force_rerun,
            )
        combined_results = SuiteResults(utility_results={}, security_results={}, injection_tasks_utility_results={})
        for suite_name, result in results.items():
            show_results(suite_name, result, attack is not None)
            for user_task, injection_task in result["utility_results"]:
                new_user_task = suite_name + "_" + user_task
                combined_results["utility_results"][(new_user_task, injection_task)] = result["utility_results"][
                    (user_task, injection_task)
                ]
            for user_task, injection_task in result["security_results"]:
                new_user_task = suite_name + "_" + user_task
                combined_results["security_results"][(new_user_task, injection_task)] = result["security_results"][
                    (user_task, injection_task)
                ]
            for injection_task in result["injection_tasks_utility_results"]:
                new_injection_task = suite_name + "_" + injection_task
                combined_results["injection_tasks_utility_results"][new_injection_task] = result[
                    "injection_tasks_utility_results"
                ][injection_task]
        show_results("combined", combined_results, attack is not None)
    else:
        with Pool(max_workers) as p:
            arguments = zip(
                suites,
                repeat(model),
                repeat(logdir),
                repeat(benchmark_version),
                repeat(user_tasks),
                repeat(injection_tasks),
                repeat(model_id),
                repeat(attack),
                repeat(defense),
                repeat(tool_delimiter),
                repeat(system_message_name),
                repeat(system_message),
                repeat(force_rerun),
            )
            results = p.starmap(benchmark_suite, arguments)

        for suite_name, result in zip(suites, results):
            show_results(suite_name, result, attack is not None)


@dataclass
class VllmConfig:
    host: str = "localhost"
    port: int = 8000
    api_key: str = "EMPTY"
    max_model_len: int = 32768

    session_name: str ="vllm_server"

    model_path_base: Path = Path.home() / "spag_optim"

    def __post_init__(self):
        self.url = f"http://{self.host}:{self.port}"

    def to_string_list(self):
        # helper function to gather commands for launching vLLM server
        return [
            "--host", self.host,
            "--port", self.port,
            "--api-key", self.api_key,
            "--max-model-len", str(self.max_model_len),
            "--gpu-memory-utilization", "0.9",
            "--generation-config", "auto",              # use vllm's generation config
            "--model-impl", "vllm",
        ]


def evaluate_model_sequentially():

    # vllm server variables
    vllm_config = VllmConfig()
    os.environ["LOCAL_LLM_PORT"] = str(vllm_config.port)

    # if the model path has iteration checkpoints, add all of them to vllms
    vllm_models = [
        # "pretrained/gemma-3-12b-it",
        # "imitation/gemma-3-12b-it-agent-lora_r_128_alpha_256_modules_all-linear",
        "grpo_general_prompt/grpo_gemma_3_12b_it_self_play_agent_population_attacker_iterative_use_sampling_iw_kl_0.05_clip_0.1_0.3_group_normalization/rl/iter_10",
        # "grpo_general_prompt/grpo_gemma_3_12b_it_agent_iterative_attacker_iterative_use_sampling_iw_kl_0.05_clip_0.1_0.3_group_normalization/rl/agent/iter_10",
        # "grpo_general_prompt/grpo_gemma_3_12b_it_agent_population_attacker_iterative_use_sampling_iw_kl_0.05_clip_0.1_0.3_group_normalization_asymmetric_kl/rl/agent/iter_10",
    ]

    # start evaluation
    for llm_name in vllm_models:
        loaded_lora_module_names = start_vllm(vllm_config, llm_name)

        if "pretrained" in llm_name or "imitation" in llm_name:
            short_llm_name = llm_name.split("/")[1]
        elif "rl" in llm_name:
            exp_name = llm_name.split("/")[1]
            iter_name = llm_name.split("/")[-1]
            short_llm_name = f"{exp_name}/{iter_name}"
        else:
            raise ValueError(f"Unknown model name: {llm_name}")
        main(
            suites=tuple(),
            model=ModelsEnum.LOCAL,
            benchmark_version="v1.2.1",
            model_id=str(vllm_config.model_path_base / llm_name),
            attack="tool_knowledge",
            defense=None,
            tool_delimiter="user",
            max_workers=32,
            logdir=Path("./runs") / short_llm_name,
        )
        unload_lora_modules(vllm_config.url, vllm_config.api_key, loaded_lora_module_names)


if __name__ == "__main__":
    format = "%(message)s"
    logging.basicConfig(
        format=format,
        level=logging.INFO,
        datefmt="%H:%M:%S",
        handlers=[RichHandler(show_path=False, markup=True)],
    )
    evaluate_model_sequentially()
