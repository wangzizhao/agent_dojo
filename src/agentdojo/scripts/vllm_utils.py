import json
import libtmux
import os
import requests
import tempfile
import time
import torch
from transformers import AutoConfig
from typing import Dict, List, Optional
from pathlib import Path


def get_vllm_available_models(url: str, api_key: str):
    """Waits for the vLLM server to become ready."""

    check_url = f"{url}/v1/models"
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        response = requests.get(check_url, headers=headers, timeout=5)
        if response.status_code == 200:
            model_info = response.json().get('data', [])
            available_model_names = [info["id"] for info in model_info]
            return available_model_names
        else:
            return []
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.RequestException):
        return []


def wait_for_vllm(url: str, api_key: str, timeout: int) -> None:
    """Waits for the vLLM server to become ready."""

    print(f"Waiting for vLLM server at {url} (timeout: {timeout}s)")
    start_time = time.time()

    while True:
        available_model_names = get_vllm_available_models(url, api_key)
        if len(available_model_names):
            return

        current_time = time.time()
        elapsed_time = current_time - start_time

        if elapsed_time >= timeout:
            raise Exception(f"ERROR: vLLM server at {url} failed to become ready. Stopping iteration.")

        print(".", end="", flush=True)
        time.sleep(5)


def close_vllm_session(session_name: str):
    # Attempts to kill the specified tmux session

    try:
        server = libtmux.Server()
        session = server.find_where({"session_name": session_name})

        if session:
            session.kill_session()
            print(f"Tmux session '{session_name}' killed successfully using libtmux.")
        else:
            print(f"Tmux session '{session_name}' does not exist or was already closed.")

    except Exception as e:
        print(f"An unexpected error occurred while using libtmux: {e}")


def post_request(url: str, headers: Dict, data: Optional[str | Dict] = None):
    if data is not None:
        assert isinstance(data, (str, dict))
        if isinstance(data, dict):
            data = json.dumps(data)

    response = requests.post(url, headers=headers, data=data, timeout=10)

    if response.status_code == 200:
        return response
    else:
        error_message = f"Request failed with status code: {response.status_code}"
        try:
            error_body = json.dumps(response.json(), indent=4)
            error_message += f"\nResponse Body: {error_body}"
        except json.JSONDecodeError:
            error_body = response.text
            error_message += f"\nResponse Body: {error_body}"
        raise requests.exceptions.RequestException(error_message, response=response)


def load_lora_modules(url: str, api_key: str, lora_module_names: List[str]) -> None:
    available_model_names = get_vllm_available_models(url, api_key)
    for lora_module_name in lora_module_names:
        if lora_module_name in available_model_names:
            print(f"LoRA Module {lora_module_name} already available, skipping loading")
            continue

        load_url = f"{url}/v1/load_lora_adapter"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        data = {
            "lora_name": lora_module_name,
            "lora_path": lora_module_name
        }

        post_request(load_url, headers, data)
        print(f"Load LoRA Modules {lora_module_name} successful! Status Code: 200")


def unload_lora_modules(url: str, api_key: str, lora_module_names: List[str]) -> None:
    available_model_names = get_vllm_available_models(url, api_key)
    print("to unload")
    for lora_module_name in lora_module_names:
        print(lora_module_name)
    print("available")
    for lora_module_name in available_model_names:
        print(lora_module_name)
    for lora_module_name in lora_module_names:
        if lora_module_name not in available_model_names:
            print(f"LoRA Module {lora_module_name} not found, skipping unloading")
            continue

        unload_url = f"{url}/v1/unload_lora_adapter"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        data = {
            "lora_name": lora_module_name,
        }

        post_request(unload_url, headers, data)
        print(f"Unload LoRA Modules {lora_module_name} successful! Status Code: 200")


def get_tensor_parallel_size(model_path: str, num_gpus: int):
    config = AutoConfig.from_pretrained(model_path)

    assert len(config.architectures) == 1, "Only one architecture is supported"
    architecture = config.architectures[0]

    if architecture.endswith("ForCausalLM"):
        num_attention_heads = config.num_attention_heads
    elif architecture.endswith("ForConditionalGeneration"):
        num_attention_heads = config.text_config.num_attention_heads
    else:
        raise ValueError(f"Invalid architecture: {architecture}")

    if num_attention_heads > num_gpus:
        assert num_attention_heads % num_gpus == 0, "Number of attention heads must be divisible by number of GPUs"
        return num_gpus
    else:
        assert num_gpus % num_attention_heads == 0, "Number of GPUs must be divisible by number of attention heads"
        return num_attention_heads


def get_tensor_parallel_size(model_path: str, num_gpus: int) -> int:
    config = AutoConfig.from_pretrained(model_path)

    assert len(config.architectures) == 1, "Only one architecture is supported"
    architecture = config.architectures[0]

    if architecture.endswith("ForCausalLM"):
        num_attention_heads = config.num_attention_heads
    elif architecture.endswith("ForConditionalGeneration"):
        num_attention_heads = config.text_config.num_attention_heads
    else:
        raise ValueError(f"Invalid architecture: {architecture}")

    if num_attention_heads > num_gpus:
        assert num_attention_heads % num_gpus == 0, "Number of attention heads must be divisible by number of GPUs"
        return num_gpus
    else:
        assert num_gpus % num_attention_heads == 0, "Number of GPUs must be divisible by number of attention heads"
        return num_attention_heads


def start_vllm(
    vllm_config,
    llm_name: str,
):

    llm_path = vllm_config.model_path_base / llm_name

    # check if a LoRA checkpoint
    base_model_path = llm_path
    adapter_config_path = llm_path / 'adapter_config.json'
    if adapter_config_path.exists():
        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)
        base_model_path = Path(adapter_config['base_model_name_or_path'])
        if not base_model_path.is_absolute():
            base_model_path = vllm_config.model_path_base / base_model_path

    tensor_parallel_size = get_tensor_parallel_size(base_model_path, torch.cuda.device_count())
    pipeline_parallel_size = torch.cuda.device_count() // tensor_parallel_size

    dtype = AutoConfig.from_pretrained(base_model_path).torch_dtype
    if dtype == torch.bfloat16:
        dtype = "bfloat16"
    elif dtype == torch.float16:
        dtype = "float16"
    elif dtype == torch.float32:
        dtype = "float32"

    available_model_names = get_vllm_available_models(vllm_config.url, vllm_config.api_key)
    should_start_vllm_server = str(base_model_path) not in available_model_names

    if len(available_model_names) > 0 and should_start_vllm_server:
        close_vllm_session(vllm_config.session_name)

    if should_start_vllm_server:
        # vLLM server is not running, need to start it

        vllm_command = [
            f"VLLM_ALLOW_RUNTIME_LORA_UPDATING=True",
            f"VLLM_SERVER_DEV_MODE=1",
            "vllm", "serve", base_model_path,
            "--pipeline-parallel-size", pipeline_parallel_size,
            "--tensor-parallel-size", tensor_parallel_size,
            "--dtype", dtype,
            "--enable-lora",
            "--max-lora-rank", "128",
        ]
        vllm_command += vllm_config.to_string_list()

        # write vllm_command to a file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".sh") as tmp:
            tmp.write("#!/bin/bash\n")
            tmp.write(" ".join([str(ele) for ele in vllm_command]) + '\n')
            script_path = tmp.name
            os.chmod(script_path, 0o755)

        # start vLLM server in a tmux session
        server = libtmux.Server()
        session = server.find_where({"session_name": vllm_config.session_name})

        if not session:
            session = server.new_session(session_name=vllm_config.session_name)

        pane = session.attached_pane
        pane.send_keys(f"bash {script_path}")

        wait_for_vllm(vllm_config.url, vllm_config.api_key, timeout=7200)

    # Wait for vLLM server to become ready and load LoRA modules
    lora_module_names = []
    if llm_path != base_model_path:
        lora_module_names.append(str(llm_path))
        load_lora_modules(vllm_config.url, vllm_config.api_key, lora_module_names)

    return lora_module_names
