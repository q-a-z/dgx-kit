from .atlas import AtlasAdapter
from .llamacpp import LlamaCppAdapter
from .sglang import SglangAdapter
from .trtllm import TrtllmAdapter
from .vllm import VllmAdapter

ADAPTERS = {"vllm": VllmAdapter, "sglang": SglangAdapter, "llamacpp": LlamaCppAdapter, "atlas": AtlasAdapter,
            "trtllm": TrtllmAdapter}


def adapter_for(engine: str, base_url: str):
    return ADAPTERS[engine](base_url)
