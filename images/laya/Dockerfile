# Laya (a decision model: typed questions in, calibrated probabilities out) served over HTTP on the GB10.
# Built on the vLLM image DGX-kit already uses, since that one has a PyTorch that runs on the GB10 and
# everything Laya needs but Laya itself. The checkpoints are mounted from the models folder at run time.
ARG BASE=vllm/vllm-openai:v0.30.0
FROM ${BASE}
ARG LAYA=0.3.27
# The recipe's own fingerprint: DGX-kit rebuilds the image when the one on the box carries another.
ARG REV=0
LABEL dgxkit.build=${REV}
RUN pip install --no-cache-dir --no-deps "laya==${LAYA}"
COPY laya_local.py /opt/laya_local.py
# TORCH_DISABLE_NATIVE_JIT: PyTorch otherwise compiles some CUDA ops on the first request and the image has no C compiler.
ENV TORCH_DISABLE_NATIVE_JIT=1 USE_TF=0 HF_HUB_OFFLINE=1 LAYA_PRELOAD=1
ENTRYPOINT ["python3", "/opt/laya_local.py"]
