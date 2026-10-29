"""vLLM config setup for Clover: the Qwen3.5 hybrid-cache contract plus a diffusion canvas of one Clover block."""
from vllm.config.diffusion import DiffusionConfig
from vllm.logger import init_logger
from vllm.model_executor.models.config import Qwen3_5ForCausalLMConfig
from vllm.v1.attention.backends.registry import AttentionBackendEnum

logger = init_logger(__name__)


class CloverConfig(Qwen3_5ForCausalLMConfig):
    @classmethod
    def verify_and_update_config(cls, vllm_config):
        Qwen3_5ForCausalLMConfig.verify_and_update_config(vllm_config)
        hf = vllm_config.model_config.hf_config
        if vllm_config.diffusion_config is None:
            vllm_config.diffusion_config = DiffusionConfig(canvas_length=int(getattr(hf, "block_size", 32)))
        if getattr(hf, "canvas_length", None) is None:
            hf.canvas_length = vllm_config.diffusion_config.canvas_length
        att = vllm_config.attention_config
        if att.backend == AttentionBackendEnum.FLASHINFER:
            raise ValueError("Clover mixes causal and bidirectional attention in one batch: use --attention-backend "
                             "FLASH_ATTN or TRITON_ATTN, not FLASHINFER")
        if vllm_config.cache_config.enable_prefix_caching:
            logger.warning("Clover keeps one mamba state slot per request: turning prefix caching off")
            vllm_config.cache_config.enable_prefix_caching = False
