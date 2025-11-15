import importlib
import json
import logging
import os
import warnings
from dataclasses import dataclass, field
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


def show_results(log_fname: Path, suite_name: str, results: SuiteResults, show_security_results: bool):
    utility_values = list(results["utility_results"].values())
    avg_utility = (sum(utility_values) / len(utility_values)) if len(utility_values) > 0 else 0.0

    payload: dict = {
        "suite_name": suite_name,
        "average_utility": avg_utility,
    }

    if show_security_results:
        passed_injection_tasks = int(sum(results["injection_tasks_utility_results"].values()))
        total_injection_tasks = int(len(results["injection_tasks_utility_results"]))
        security_values = list(results["security_results"].values())
        avg_security = (sum(security_values) / len(security_values)) if len(security_values) > 0 else 0.0
        payload.update(
            {
                "passed_injection_tasks": passed_injection_tasks,
                "total_injection_tasks": total_injection_tasks,
                "average_security": avg_security,
            }
        )

    log_fname.parent.mkdir(parents=True, exist_ok=True)
    with open(log_fname, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


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
        results = {suite_name: result for suite_name, result in zip(suites, results)}

    combined_results = SuiteResults(utility_results={}, security_results={}, injection_tasks_utility_results={})
    results_log = logdir / f"attack_{'none' if attack is None else attack}_defense_{'none' if defense is None else defense}_results.json"
    for suite_name, result in results.items():
        show_results(results_log, suite_name, result, attack is not None)
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
    show_results(results_log, "combined", combined_results, attack is not None)


@dataclass
class VllmConfig:
    host: str = "localhost"
    port: int = 8000
    api_key: str = "EMPTY"
    max_model_len: int = 32768

    session_name: str ="vllm_server"

    model_path_base: Path = Path.home() / "spag_ckpt"

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

@dataclass
class EvalConfig:
    base_models: list[str] = field(
        default_factory=lambda: [
            "pretrained/gemma-3-12b-it",
            # "pretrained/qwen3-14b",
        ],
    )
    exp_models: list[str] = field(
        default_factory=lambda: [
            # "iclr_exp/grpo_gemma_3_12b_it_self_play_agent_iterative_attacker_iterative",
            # "iclr_exp/grpo_gemma_3_12b_it_self_play_agent_population_attacker_iterative",
            # "iclr_exp/grpo_gemma_3_12b_it_agent_iterative",
            # "iclr_exp/grpo_gemma_3_12b_it_art_agent_iterative",
            # "iclr_exp/spag_gemma_3_12b_it_self_play_agent_iterative_attacker_iterative",
            # "iclr_exp/grpo_qwen3_14b_self_play_agent_population_attacker_iterative",
        ],
    )

    # "workspace", "travel", "banking", "slack"
    suites: tuple[str, ...] = field(
        default_factory=lambda: (
            "workspace",
            # "travel",
            # "banking",
            # "slack",
        )
    )

    attacks: list[str | None] = field(
        default_factory=lambda: [
            "adaptive_gemini_no_defense",
            "adaptive_gemini_spotlighting",
            "adaptive_gemini_prompt_sandwich",
            "adaptive_gemini_pi_guard",
            "adaptive_gpt_model_armor",
        ]
    )

    defenses: list[str | None] = field(
        default_factory=lambda: [
            None,
            # "tool_filter",
            # "transformers_pi_detector",
            # "spotlighting_with_delimiting",
        ]
    )

    log_path = Path.home() / "bucket" / "agentdojo" / "iclr_rebuttal"


def evaluate_model_sequentially():
    eval_config = EvalConfig()

    # vllm server variables
    vllm_config = VllmConfig()
    os.environ["LOCAL_LLM_PORT"] = str(vllm_config.port)

    # if the model path has iteration checkpoints, add all of them to vllms
    vllm_models = eval_config.base_models
    for exp_name in eval_config.exp_models:
        for seed in [0, 1, 2]:
            for iteration in [20, 15, 10, 5]:
                llm_name_candidate1 = f"{exp_name}_seed_{seed}/rl/agent/iter_{iteration}"
                llm_name_candidate2 = f"{exp_name}_seed_{seed}/rl/iter_{iteration}"
                if (vllm_config.model_path_base / llm_name_candidate1).exists():
                    vllm_models.append(llm_name_candidate1)
                    break
                elif (vllm_config.model_path_base / llm_name_candidate2).exists():
                    vllm_models.append(llm_name_candidate2)
                    break

    # start evaluation
    for llm_name in vllm_models:
        if "qwen" in llm_name:
            vllm_config.max_model_len = 40960
        elif "gemma" in llm_name:
            vllm_config.max_model_len = 128000
        else:
            raise ValueError(f"Unknown model name: {llm_name}")

        loaded_lora_module_names = start_vllm(vllm_config, llm_name)

        if "pretrained" in llm_name or "imitation" in llm_name:
            short_llm_name = llm_name.split("/")[1]
        elif "rl" in llm_name:
            exp_name = llm_name.split("/")[1]
            iter_name = llm_name.split("/")[-1]
            short_llm_name = f"{exp_name}/{iter_name}"
        else:
            raise ValueError(f"Unknown model name: {llm_name}")

        for attack in eval_config.attacks:
            for defense in eval_config.defenses:
                main(
                    suites=eval_config.suites,
                    model=ModelsEnum.LOCAL,
                    benchmark_version="v1.2.1",
                    model_id=str(vllm_config.model_path_base / llm_name),
                    attack=attack,
                    defense=defense,
                    tool_delimiter="user",
                    max_workers=32,
                    logdir=eval_config.log_path / short_llm_name,
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
