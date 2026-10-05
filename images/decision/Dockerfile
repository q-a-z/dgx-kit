# Lev and Bekko, two more decision models (typed questions about a text in, calibrated probabilities out), served over HTTP.
# One image for both, built on the vLLM image DGX-kit already uses: it has a PyTorch that runs on the GB10, transformers 5
# and FastAPI. Only the extras are added; a constraint file stops pip from replacing the PyTorch that is there.
ARG BASE=vllm/vllm-openai:v0.30.0
FROM ${BASE}
# The Lev source, pinned to a commit so the image doesn't change under you.
ARG LEV=adbbc5c34a90e754c22cc5eb86fc68c99597096e
# The recipe's own fingerprint: DGX-kit rebuilds the image when the one on the box carries another.
ARG REV=0
LABEL dgxkit.build=${REV}
# Only what is missing is installed: the base image already has accelerate, psutil, pydantic and tqdm, and Lev's server
# and Bekko's code import without the heavy training extras (datasets, pyarrow, pandas), which are left out. Package
# downloads can be slow on a Spark, so the list is kept short: scikit-learn (and its scipy) is the only large one,
# and sentence-transformers asks for it.
RUN pip freeze | grep -iE '^(torch|torchvision|torchaudio|transformers|tokenizers|safetensors|numpy|huggingface-hub|huggingface_hub)==' > /tmp/keep.txt \
 && PIP_CONSTRAINT=/tmp/keep.txt pip install --no-cache-dir scikit-learn \
 && PIP_CONSTRAINT=/tmp/keep.txt pip install --no-cache-dir --no-deps peft "sentence-transformers==6.1.0" \
 && PIP_CONSTRAINT=/tmp/keep.txt pip install --no-cache-dir --no-deps "https://github.com/Abhinavexists/lev/archive/${LEV}.tar.gz#subdirectory=packages/lev" \
 && python3 -c "import lev.server, lev.cli, peft, sentence_transformers, accelerate; print('decision image: imports ok')"
COPY lev_local.py bekko_serve.py /opt/
# The checkpoints are mounted from the models folder, and nothing here may reach for the Hub.
ENV TORCH_DISABLE_NATIVE_JIT=1 USE_TF=0 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
