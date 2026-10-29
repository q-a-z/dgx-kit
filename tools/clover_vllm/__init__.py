def register():
    from vllm import ModelRegistry
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP
    from vllm.sampling_params import SamplingParams, _SAMPLING_EPS
    from vllm.exceptions import VLLMValidationError

    from .config_hook import CloverConfig
    ModelRegistry.register_model("CloverMoeModelLM", "clover_vllm.model:CloverMoeModelLM")
    MODELS_CONFIG_MAP["CloverMoeModelLM"] = CloverConfig

    orig = SamplingParams._validate_diffusion

    def _validate_diffusion(self, model_config):
        # vLLM insists on temperature == 1.0 for diffusion models. Clover decodes greedily whatever the client sends (its
        # generation_config says do_sample=false), and agent clients always send a temperature, so accept it; keep the rest.
        if not model_config.is_diffusion or "clover" not in str(getattr(model_config.hf_config, "model_type", "")):
            return orig(self, model_config)
        if (self.min_p > _SAMPLING_EPS or self.seed is not None or self.min_tokens > 0 or self.logit_bias
                or self.bad_words or self.allowed_token_ids):
            raise VLLMValidationError("Clover does not support min_p, seed, min_tokens, logit_bias, bad_words or allowed_token_ids")

    SamplingParams._validate_diffusion = _validate_diffusion
