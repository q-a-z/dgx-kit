ARG BASE=nvidia/cuda:13.0.2-runtime-ubuntu24.04
FROM ${BASE}
ARG VERSION=v0.8.0
ARG SHA256=5580bae0a9dbdc138a5f60e02aed18fa734a92c905ea074f55d7b331d17f48a8
RUN test "$(uname -m)" = aarch64 \
    && apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl libibverbs1 ibverbs-providers librdmacm1 libgomp1 libssl3t64 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/velogb10
RUN set -eu; \
    archive="velogb10-${VERSION}-gb10-sm121.tar.gz"; \
    base="https://github.com/sf-stav/veloGB10/releases/download/${VERSION}"; \
    mkdir -p /tmp/velo-download /tmp/velo-unpack; \
    cd /tmp/velo-download; \
    curl -fL --retry 3 "${base}/${archive}" -o "$archive"; \
    curl -fL --retry 3 "${base}/SHA256SUMS.txt" -o SHA256SUMS; \
    awk -v name="$archive" '$2 == name || $2 == "*" name {print}' SHA256SUMS > selected.sha256; \
    test -s selected.sha256; \
    sha256sum -c selected.sha256; \
    echo "${SHA256}  ${archive}" | sha256sum -c -; \
    tar -xzf "$archive" -C /tmp/velo-unpack; \
    binary="$(find /tmp/velo-unpack -type f -name gb10_inference -print -quit)"; \
    test -n "$binary"; \
    cp -a "$(dirname "$binary")/." /opt/velogb10/; \
    chmod +x /opt/velogb10/gb10_inference; \
    rm -rf /tmp/velo-download /tmp/velo-unpack
ENV PATH="/opt/velogb10:${PATH}"
ENTRYPOINT ["/opt/velogb10/gb10_inference"]
CMD ["--help"]
