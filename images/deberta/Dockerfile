ARG BASE=dgx-kit/vllm:nightly-gb10
FROM ${BASE}
ARG REV=0
LABEL dgxkit.build=${REV}
COPY serve.py /opt/deberta_serve.py
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false USE_TF=0
