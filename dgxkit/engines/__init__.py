from .atlas import AtlasAdapter
from .llamacpp import LlamaCppAdapter
from .sglang import SglangAdapter
from .trtllm import TrtllmAdapter
from .vllm import VllmAdapter
from .velogb10 import VeloGB10Adapter

ADAPTERS = {"vllm": VllmAdapter, "sglang": SglangAdapter, "llamacpp": LlamaCppAdapter, "atlas": AtlasAdapter,
            "trtllm": TrtllmAdapter, "velogb10": VeloGB10Adapter}


def adapter_for(engine: str, base_url: str):
    return ADAPTERS[engine](base_url)
