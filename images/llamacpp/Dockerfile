# llama.cpp server built for the DGX Spark's GB10 (compute capability 12.1, arm64).
# Built on the Spark itself from the Images tab; the result is tagged dgx-kit/llamacpp:gb10.
ARG CUDA=13.0.1
FROM nvidia/cuda:${CUDA}-devel-ubuntu24.04 AS build
ARG LLAMA_REF=master
RUN apt-get update && apt-get install -y --no-install-recommends git cmake build-essential libcurl4-openssl-dev ca-certificates \
 && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch ${LLAMA_REF} https://github.com/ggml-org/llama.cpp /src
WORKDIR /src
RUN cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121 -DLLAMA_CURL=ON -DGGML_NATIVE=OFF \
      -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined \
 && cmake --build build --config Release -j"$(nproc)" --target llama-server \
 && mkdir -p /app && cp build/bin/llama-server /app/ && find build -name '*.so*' -exec cp -P {} /app/ \;

FROM nvidia/cuda:${CUDA}-runtime-ubuntu24.04
RUN apt-get update && apt-get install -y --no-install-recommends libcurl4 libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*
COPY --from=build /app /app
ENV LD_LIBRARY_PATH=/app:${LD_LIBRARY_PATH}
WORKDIR /app
# Same layout as the upstream server-cuda image, so DGX-kit starts either one the same way.
ENTRYPOINT ["/app/llama-server"]
