from .llamacpp import LlamaCppAdapter
from .sglang import SglangAdapter
from .vllm import VllmAdapter

ADAPTERS = {"vllm": VllmAdapter, "sglang": SglangAdapter, "llamacpp": LlamaCppAdapter}


def adapter_for(engine: str, base_url: str):
    return ADAPTERS[engine](base_url)
