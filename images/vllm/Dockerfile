# vLLM for the DGX Spark's GB10: an upstream vLLM image with DGX-kit's patches applied on top,
# FlashInfer 0.7.0 in place of the release's older one, and the GB10 build settings.
# Built on the Spark from the Images tab. BASE picks the vLLM release, SERIES picks its patch folder:
#   patches/common/*.patch   applied to every release
#   patches/<SERIES>/*.patch applied to that release only (for example patches/0.29)
# Patches are unified diffs against the installed site-packages (paths like a/vllm/...).
ARG BASE=vllm/vllm-openai:v0.30.0
FROM ${BASE}
ARG SERIES=0.30
# FlashInfer 0.7.0 is the one the GB10 fleet was validated on. The stock image brings an older FlashInfer plus prebuilt
# kernel packages that must match it exactly, so they go too (the nightlies ship the cache as per-architecture flashinfer-jit-cache-sm* packages, so match by prefix; the emptied package folders are removed too, or FlashInfer finds them and fails on a missing __version__); kernels then compile on first use (cached on the host).
# FLASHINFER= (empty) keeps the stock one.
ARG FLASHINFER=0.7.0
RUN if [ -n "$FLASHINFER" ]; then \
      python3 -m pip uninstall -y $(python3 -m pip list --format=freeze | cut -d= -f1 | grep -E '^flashinfer[-_](cubin|jit[-_]cache)') && \
      rm -rf /usr/local/lib/python3.12/dist-packages/flashinfer_cubin /usr/local/lib/python3.12/dist-packages/flashinfer_jit_cache && \
      python3 -m pip install --no-cache-dir --no-deps --break-system-packages "flashinfer-python==${FLASHINFER}"; \
    fi
ENV TORCH_CUDA_ARCH_LIST=12.1a FLASHINFER_CUDA_ARCH_LIST=12.1a PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
COPY patches/ /tmp/dgxkit-patches/
RUN set -e; \
    SITE="$(python3 -c 'import os, vllm; print(os.path.dirname(os.path.dirname(vllm.__file__)))')"; \
    command -v patch >/dev/null || (apt-get update && apt-get install -y --no-install-recommends patch && rm -rf /var/lib/apt/lists/*); \
    n=0; \
    for p in /tmp/dgxkit-patches/common/*.patch /tmp/dgxkit-patches/"${SERIES}"/*.patch; do \
      [ -e "$p" ] || continue; \
      echo "applying $(basename "$p")"; \
      patch -d "$SITE" -p1 --forward --batch < "$p"; \
      n=$((n + 1)); \
    done; \
    echo "$n patches applied to vLLM in $SITE"; \
    printf 'series=%s\npatches=%s\n' "${SERIES}" "$n" > /etc/dgxkit-vllm-build
