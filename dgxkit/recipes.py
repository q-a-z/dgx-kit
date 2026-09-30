"""Recipes: everything needed to run one model, built from a Hugging Face repo.

A recipe is plain data (saved as YAML). inspect_repo() reads a repo's config
and file list and fills in engine, quantization and draft method; the user can
edit anything afterwards.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

WEIGHT_SUFFIXES = (".safetensors", ".bin", ".gguf")


@dataclass
class Recipe:
    name: str
    repo: str
    engine: str = "vllm"            # vllm | sglang | llamacpp
    image: str | None = None        # set by the image manager
    revision: str | None = None
    quantization: str | None = None  # nvfp4 | fp8 | gguf | None
    gguf_file: str | None = None
    draft_repo: str | None = None
    draft_method: str | None = None  # eagle3 | dflash | mtp | draft_model | auto (the engine reads it from the draft)
    num_speculative_tokens: int = 3
    kv_cache_dtype: str = "auto"
    kv_cache_bytes: int | None = None     # fixed KV cache size; None lets DGX-kit size it from free memory
    gpu_memory_utilization: float | None = None  # fixed vLLM share of memory; None lets DGX-kit work it out
    max_context: int | None = None
    min_context: int = 4096
    min_concurrency: float = 2.0
    extra_args: list[str] = field(default_factory=list)
    speculative_extra: dict = field(default_factory=dict)  # more --speculative-config keys, e.g. draft_sample_method
    env: dict = field(default_factory=dict)  # extra environment for the engine container
    docker: dict = field(default_factory=dict)  # container settings: mem_limit, shm_size, volumes ["src:dst[:mode]"]
    notes: str = ""                  # the user's own notes (imported from llmctl's OPTS)
    path: str | None = None          # weights already on disk (from the library) instead of a download
    draft_path: str | None = None
    publish: bool = True             # list it on the LiteLLM gateway while it runs
    weights_bytes: int = 0
    draft_weights_bytes: int = 0
    config: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def detect_quantization(config: dict, files: list[str]) -> str | None:
    if any(f.endswith(".gguf") for f in files):
        return "gguf"
    q = config.get("quantization_config") or {}
    blob = json.dumps(q).lower()
    if "nvfp4" in blob or "fp4" in blob:
        return "nvfp4"
    if "fp8" in blob:
        return "fp8"
    return q.get("quant_method")


def detect_draft_method(draft_config: dict, name: str = "") -> str:
    archs = " ".join(draft_config.get("architectures") or []).lower()
    mtype = str(draft_config.get("model_type", "")).lower()
    name = name.lower()
    for method in ("eagle3", "dflash", "dspark", "mtp"):
        if method in archs or method in mtype:
            return method
    # EAGLE checkpoints often reuse the base model's architecture; the name is all that says so.
    if "eagle3" in name:
        return "eagle3"
    if "eagle" in archs or "eagle" in name:
        return "eagle"
    return "draft_model"


def pick_engine(quantization: str | None) -> str:
    return "llamacpp" if quantization == "gguf" else "vllm"


def weight_bytes(siblings: list[tuple[str, int | None]], gguf_file: str | None = None) -> int:
    if gguf_file:
        return sum(size or 0 for name, size in siblings if name == gguf_file)
    return sum(size or 0 for name, size in siblings if name.endswith(WEIGHT_SUFFIXES))


def build_recipe(repo: str, config: dict, siblings: list[tuple[str, int | None]], *,
                 draft_repo: str | None = None, draft_config: dict | None = None,
                 draft_siblings: list[tuple[str, int | None]] | None = None,
                 gguf_file: str | None = None) -> Recipe:
    """Pure part of inspect_repo(), so it can be tested without the network."""
    files = [n for n, _ in siblings]
    quant = detect_quantization(config, files)
    ggufs = sorted(f for f in files if f.endswith(".gguf"))
    if quant == "gguf" and not gguf_file:
        gguf_file = ggufs[0] if len(ggufs) == 1 else None  # several quants: the user picks
    r = Recipe(
        name=repo.split("/")[-1].lower(),
        repo=repo,
        engine=pick_engine(quant),
        quantization=quant,
        gguf_file=gguf_file,
        draft_repo=draft_repo,
        draft_method=detect_draft_method(draft_config, draft_repo or "") if draft_config else None,
        # Several GGUF quants and none picked yet: size unknown until the user chooses.
        weights_bytes=0 if quant == "gguf" and not gguf_file else weight_bytes(siblings, gguf_file),
        draft_weights_bytes=weight_bytes(draft_siblings or []),
        config=config,
    )
    if quant in ("nvfp4", "fp8"):
        r.kv_cache_dtype = "fp8"
    return r


def inspect_repo(repo: str, draft_repo: str | None = None, token: str | None = None) -> Recipe:
    """Read config.json and the file list from the Hub and build a recipe."""
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=token)

    def load(r):
        info = api.model_info(r, files_metadata=True)
        siblings = [(s.rfilename, s.size) for s in info.siblings or []]
        cfg = {}
        if any(n == "config.json" for n, _ in siblings):
            with open(hf_hub_download(r, "config.json", token=token)) as f:
                cfg = json.load(f)
        return cfg, siblings

    config, siblings = load(repo)
    draft_config, draft_siblings = load(draft_repo) if draft_repo else (None, None)
    return build_recipe(repo, config, siblings, draft_repo=draft_repo, draft_config=draft_config,
                        draft_siblings=draft_siblings)
